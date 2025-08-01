/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { createHash } from 'crypto';
import { GeminiEventType, ServerGeminiStreamEvent } from '../core/turn.js';
import { logLoopDetected } from '../telemetry/loggers.js';
import { LoopDetectedEvent, LoopType } from '../telemetry/types.js';
import { Config, DEFAULT_GEMINI_FLASH_MODEL } from '../config/config.js';
import { SchemaUnion, Type } from '@google/genai';

const TOOL_CALL_LOOP_THRESHOLD = 5;
const CONTENT_LOOP_THRESHOLD = 10;
const CONTENT_CHUNK_SIZE = 50;
const MAX_HISTORY_LENGTH = 1000;

// 基于句子的备用检测常量
const SENTENCE_CONTENT_LOOP_THRESHOLD = 10;
const SENTENCE_ENDING_PUNCTUATION_REGEX = /[.!?]+(?=\s|$)/;

/**
 * The number of recent conversation turns to include in the history when asking the LLM to check for a loop.
 */
const LLM_LOOP_CHECK_HISTORY_COUNT = 20;

/**
 * The number of turns that must pass in a single prompt before the LLM-based loop check is activated.
 */
const LLM_CHECK_AFTER_TURNS = 30;

/**
 * The default interval, in number of turns, at which the LLM-based loop check is performed.
 * This value is adjusted dynamically based on the LLM's confidence.
 */
const DEFAULT_LLM_CHECK_INTERVAL = 3;

/**
 * The minimum interval for LLM-based loop checks.
 * This is used when the confidence of a loop is high, to check more frequently.
 */
const MIN_LLM_CHECK_INTERVAL = 5;

/**
 * The maximum interval for LLM-based loop checks.
 * This is used when the confidence of a loop is low, to check less frequently.
 */
const MAX_LLM_CHECK_INTERVAL = 15;

/**
 * Service for detecting and preventing infinite loops in AI responses.
 * Monitors tool call repetitions and content sentence repetitions.
 */
export class LoopDetectionService {
  private readonly config: Config;
  private promptId = '';

  // Tool call tracking
  private lastToolCallKey: string | null = null;
  private toolCallRepetitionCount: number = 0;

  // Content streaming tracking
  private streamContentHistory = '';
  private contentStats = new Map<string, number[]>();
  private lastContentIndex = 0;
  private loopDetected = false;
  private inCodeBlock = false;

  // LLM loop track tracking
  private turnsInCurrentPrompt = 0;
  private llmCheckInterval = DEFAULT_LLM_CHECK_INTERVAL;
  private lastCheckTurn = 0;

  // 基于句子的备用检测状态
  private lastRepeatedSentence: string = '';
  private sentenceRepetitionCount: number = 0;
  private partialSentenceContent: string = '';

  constructor(config: Config) {
    this.config = config;
  }

  private getToolCallKey(toolCall: { name: string; args: object }): string {
    const argsString = JSON.stringify(toolCall.args);
    const keyString = `${toolCall.name}:${argsString}`;
    return createHash('sha256').update(keyString).digest('hex');
  }

  /**
   * Processes a stream event and checks for loop conditions.
   * @param event - The stream event to process
   * @returns true if a loop is detected, false otherwise
   */
  addAndCheck(event: ServerGeminiStreamEvent): boolean {
    if (this.loopDetected) {
      return true;
    }

    switch (event.type) {
      case GeminiEventType.ToolCallRequest:
        // content chanting only happens in one single stream, reset if there
        // is a tool call in between
        this.resetContentTracking();
        this.resetSentenceTracking();
        this.loopDetected = this.checkToolCallLoop(event.value);
        break;
      case GeminiEventType.Content:
        this.loopDetected = this.checkContentLoop(event.value);
        break;
      default:
        break;
    }
    return this.loopDetected;
  }

  /**
   * Signals the start of a new turn in the conversation.
   *
   * This method increments the turn counter and, if specific conditions are met,
   * triggers an LLM-based check to detect potential conversation loops. The check
   * is performed periodically based on the `llmCheckInterval`.
   *
   * @param signal - An AbortSignal to allow for cancellation of the asynchronous LLM check.
   * @returns A promise that resolves to `true` if a loop is detected, and `false` otherwise.
   */
  async turnStarted(signal: AbortSignal) {
    this.turnsInCurrentPrompt++;

    if (
      this.turnsInCurrentPrompt >= LLM_CHECK_AFTER_TURNS &&
      this.turnsInCurrentPrompt - this.lastCheckTurn >= this.llmCheckInterval
    ) {
      this.lastCheckTurn = this.turnsInCurrentPrompt;
      return await this.checkForLoopWithLLM(signal);
    }

    return false;
  }

  private checkToolCallLoop(toolCall: { name: string; args: object }): boolean {
    const key = this.getToolCallKey(toolCall);
    if (this.lastToolCallKey === key) {
      this.toolCallRepetitionCount++;
    } else {
      this.lastToolCallKey = key;
      this.toolCallRepetitionCount = 1;
    }
    if (this.toolCallRepetitionCount >= TOOL_CALL_LOOP_THRESHOLD) {
      logLoopDetected(
        this.config,
        new LoopDetectedEvent(
          LoopType.CONSECUTIVE_IDENTICAL_TOOL_CALLS,
          this.promptId,
        ),
      );
      return true;
    }
    return false;
  }

  /**
   * Detects content loops by analyzing streaming text for repetitive patterns.
   *
   * The algorithm works by:
   * 1. Appending new content to the streaming history
   * 2. Truncating history if it exceeds the maximum length
   * 3. Analyzing content chunks for repetitive patterns using hashing
   * 4. Detecting loops when identical chunks appear frequently within a short distance
   * 5. Disabling loop detection within code blocks to prevent false positives,
   *    as repetitive code structures are common and not necessarily loops.
   */
  private checkContentLoop(content: string): boolean {
    // Code blocks can often contain repetitive syntax that is not indicative of a loop.
    // To avoid false positives, we detect when we are inside a code block and
    // temporarily disable loop detection.
    const numFences = (content.match(/```/g) ?? []).length;
    
    // 只在代码块状态改变时才重置跟踪
    const wasInCodeBlock = this.inCodeBlock;
    if (numFences > 0) {
      this.inCodeBlock = numFences % 2 === 0 ? this.inCodeBlock : !this.inCodeBlock;
      
      // 只有当从代码块外进入代码块时才重置
      if (!wasInCodeBlock && this.inCodeBlock) {
        this.resetContentTracking(false); // 不重置历史，只重置统计
      }
    }
    
    // 在代码块内时跳过检测
    if (this.inCodeBlock) {
      return false;
    }

    this.streamContentHistory += content;
    this.partialSentenceContent += content;

    this.truncateAndUpdate();
    
    // 首先尝试基于块的检测
    const blockDetected = this.analyzeContentChunksForLoop();
    if (blockDetected) {
      return true;
    }
    
    // 同时进行基于句子的检测作为备用
    return this.checkSentenceBasedLoop();
  }

  /**
   * Truncates the content history to prevent unbounded memory growth.
   * When truncating, adjusts all stored indices to maintain their relative positions.
   */
  private truncateAndUpdate(): void {
    if (this.streamContentHistory.length <= MAX_HISTORY_LENGTH) {
      return;
    }

    // Calculate how much content to remove from the beginning
    const truncationAmount =
      this.streamContentHistory.length - MAX_HISTORY_LENGTH;
    this.streamContentHistory =
      this.streamContentHistory.slice(truncationAmount);
    this.lastContentIndex = Math.max(
      0,
      this.lastContentIndex - truncationAmount,
    );

    // Update all stored chunk indices to account for the truncation
    for (const [hash, oldIndices] of this.contentStats.entries()) {
      const adjustedIndices = oldIndices
        .map((index) => index - truncationAmount)
        .filter((index) => index >= 0);

      if (adjustedIndices.length > 0) {
        this.contentStats.set(hash, adjustedIndices);
      } else {
        this.contentStats.delete(hash);
      }
    }
  }

  /**
   * Analyzes content in fixed-size chunks to detect repetitive patterns.
   *
   * Uses a sliding window approach:
   * 1. Extract chunks of fixed size (CONTENT_CHUNK_SIZE)
   * 2. Hash each chunk for efficient comparison
   * 3. Track positions where identical chunks appear
   * 4. Detect loops when chunks repeat frequently within a short distance
   */
  private analyzeContentChunksForLoop(): boolean {
    while (this.hasMoreChunksToProcess()) {
      // Extract current chunk of text
      const currentChunk = this.streamContentHistory.substring(
        this.lastContentIndex,
        this.lastContentIndex + CONTENT_CHUNK_SIZE,
      );
      const chunkHash = createHash('sha256').update(currentChunk).digest('hex');

      if (this.isLoopDetectedForChunk(currentChunk, chunkHash)) {
        logLoopDetected(
          this.config,
          new LoopDetectedEvent(
            LoopType.CHANTING_IDENTICAL_SENTENCES,
            this.promptId,
          ),
        );
        return true;
      }

      // Move to next position in the sliding window
      this.lastContentIndex++;
    }

    return false;
  }

  private hasMoreChunksToProcess(): boolean {
    return (
      this.lastContentIndex + CONTENT_CHUNK_SIZE <=
      this.streamContentHistory.length
    );
  }

  /**
   * Determines if a content chunk indicates a loop pattern.
   *
   * Loop detection logic:
   * 1. Check if we've seen this hash before (new chunks are stored for future comparison)
   * 2. Verify actual content matches to prevent hash collisions
   * 3. Track all positions where this chunk appears
   * 4. A loop is detected when the same chunk appears CONTENT_LOOP_THRESHOLD times
   *    within a small average distance (≤ 1.5 * chunk size)
   */
  private isLoopDetectedForChunk(chunk: string, hash: string): boolean {
    const existingIndices = this.contentStats.get(hash);

    if (!existingIndices) {
      this.contentStats.set(hash, [this.lastContentIndex]);
      return false;
    }

    if (!this.isActualContentMatch(chunk, existingIndices[0])) {
      return false;
    }

    existingIndices.push(this.lastContentIndex);

    if (existingIndices.length < CONTENT_LOOP_THRESHOLD) {
      return false;
    }

    // Analyze the most recent occurrences to see if they're clustered closely together
    const recentIndices = existingIndices.slice(-CONTENT_LOOP_THRESHOLD);
    const totalDistance =
      recentIndices[recentIndices.length - 1] - recentIndices[0];
    const averageDistance = totalDistance / (CONTENT_LOOP_THRESHOLD - 1);
    const maxAllowedDistance = CONTENT_CHUNK_SIZE * 1.5;

    return averageDistance <= maxAllowedDistance;
  }

  /**
   * Verifies that two chunks with the same hash actually contain identical content.
   * This prevents false positives from hash collisions.
   */
  private isActualContentMatch(
    currentChunk: string,
    originalIndex: number,
  ): boolean {
    const originalChunk = this.streamContentHistory.substring(
      originalIndex,
      originalIndex + CONTENT_CHUNK_SIZE,
    );
    return originalChunk === currentChunk;
  }

  /**
   * 基于句子的内容循环检测（备用方法）
   * 当基于块的检测无法有效工作时使用
   */
  private checkSentenceBasedLoop(): boolean {
    if (!SENTENCE_ENDING_PUNCTUATION_REGEX.test(this.partialSentenceContent)) {
      return false;
    }

    // 改进的句子分割正则，处理更多边界情况
    const completeSentences =
      this.partialSentenceContent.match(/[^.!?]*[.!?]+/g) || [];
    if (completeSentences.length === 0) {
      return false;
    }

    const lastSentence = completeSentences[completeSentences.length - 1];
    const lastCompleteIndex = this.partialSentenceContent.lastIndexOf(lastSentence);
    const endOfLastSentence = lastCompleteIndex + lastSentence.length;
    this.partialSentenceContent = this.partialSentenceContent.slice(endOfLastSentence);

    for (const sentence of completeSentences) {
      const trimmedSentence = sentence.trim();
      if (trimmedSentence === '' || trimmedSentence.length < 5) {
        continue; // 忽略空句子和太短的句子
      }

      if (this.lastRepeatedSentence === trimmedSentence) {
        this.sentenceRepetitionCount++;
      } else {
        this.lastRepeatedSentence = trimmedSentence;
        this.sentenceRepetitionCount = 1;
      }

      if (this.sentenceRepetitionCount >= SENTENCE_CONTENT_LOOP_THRESHOLD) {
        logLoopDetected(
          this.config,
          new LoopDetectedEvent(
            LoopType.CHANTING_IDENTICAL_SENTENCES,
            this.promptId,
          ),
        );
        return true;
      }
    }
    return false;
  }

  private async checkForLoopWithLLM(signal: AbortSignal) {
    const recentHistory = this.config
      .getGeminiClient()
      .getHistory()
      .slice(-LLM_LOOP_CHECK_HISTORY_COUNT);

    const prompt = `You are a sophisticated AI diagnostic agent specializing in identifying when a conversational AI is stuck in an unproductive state. Your task is to analyze the provided conversation history and determine if the assistant has ceased to make meaningful progress.

An unproductive state is characterized by one or more of the following patterns over the last 5 or more assistant turns:

Repetitive Actions: The assistant repeats the same tool calls or conversational responses a decent number of times. This includes simple loops (e.g., tool_A, tool_A, tool_A) and alternating patterns (e.g., tool_A, tool_B, tool_A, tool_B, ...).

Cognitive Loop: The assistant seems unable to determine the next logical step. It might express confusion, repeatedly ask the same questions, or generate responses that don't logically follow from the previous turns, indicating it's stuck and not advancing the task.

Crucially, differentiate between a true unproductive state and legitimate, incremental progress.
For example, a series of 'tool_A' or 'tool_B' tool calls that make small, distinct changes to the same file (like adding docstrings to functions one by one) is considered forward progress and is NOT a loop. A loop would be repeatedly replacing the same text with the same content, or cycling between a small set of files with no net change.

CRITICAL OUTPUT FORMAT REQUIREMENTS:
You MUST respond with ONLY a valid JSON object. NO additional text, explanations, markdown blocks, or formatting is allowed.
The JSON must contain exactly these two fields with these exact names:
- "reasoning": a string explaining your analysis
- "confidence": a number between 0.0 and 1.0

REQUIRED OUTPUT FORMAT (copy this structure exactly):
{
  "reasoning": "Your detailed analysis of whether the conversation shows repetitive patterns without progress",
  "confidence": 0.5
}

VALIDATION RULES:
- reasoning: MUST be a string, cannot be null or empty
- confidence: MUST be a number between 0.0 and 1.0 (inclusive)
- Use exactly these field names: "reasoning" and "confidence"
- No additional fields allowed
- No text outside the JSON object
- No markdown code blocks like \`\`\`json

EXAMPLES OF CORRECT RESPONSES:
{"reasoning": "The assistant has made 8 consecutive identical tool calls without any variation in parameters or outcomes, indicating a clear unproductive loop.", "confidence": 0.95}

{"reasoning": "The assistant is making different tool calls with varying parameters and each call produces different results, showing clear forward progress.", "confidence": 0.1}

Remember: ONLY return the JSON object. Nothing else.`;
    const systemInstruction = `You are ONLY a diagnostic agent. Your SOLE function is to analyze conversation patterns and return JSON.

CRITICAL: You are NOT a coding assistant. You CANNOT use tools. You CANNOT execute functions. 
You can ONLY analyze text and return a JSON object with "reasoning" and "confidence" fields.

If you try to use tools or return anything other than the specified JSON format, you will fail your task.`;

    const contents = [
      { role: 'system', parts: [{ text: systemInstruction }] },
      ...recentHistory,
      { role: 'user', parts: [{ text: prompt }] },
    ];
    const schema: SchemaUnion = {
      type: Type.OBJECT,
      properties: {
        reasoning: {
          type: Type.STRING,
          description:
            'Your reasoning on if the conversation is looping without forward progress.',
          minLength: 10,
          maxLength: 500,
        },
        confidence: {
          type: Type.NUMBER,
          description:
            'A number between 0.0 and 1.0 representing your confidence that the conversation is in an unproductive state.',
          minimum: 0.0,
          maximum: 1.0,
        },
      },
      required: ['reasoning', 'confidence'],
      additionalProperties: false, // 禁止额外属性
    };
    let result;
    try {
      result = await this.config
        .getGeminiClient()
        .generateJson(contents, schema, signal, DEFAULT_GEMINI_FLASH_MODEL);
    } catch (e) {
      // LLM检测失败，触发基于句子的备用检测
      this.config.getDebugMode() ? console.error('LLM loop detection failed, falling back to sentence-based detection:', e) : console.debug('LLM loop detection failed, using fallback method');
      return this.performFallbackLoopDetection();
    }

    if (typeof result.confidence === 'number') {
      // 验证confidence值在有效范围内
      const confidence = Math.max(0.0, Math.min(1.0, result.confidence));
      
      if (confidence > 0.9) {
        if (typeof result.reasoning === 'string' && result.reasoning) {
          console.warn(result.reasoning);
        }
        logLoopDetected(
          this.config,
          new LoopDetectedEvent(LoopType.LLM_DETECTED_LOOP, this.promptId),
        );
        return true;
      } else {
        this.llmCheckInterval = Math.round(
          MIN_LLM_CHECK_INTERVAL +
            (MAX_LLM_CHECK_INTERVAL - MIN_LLM_CHECK_INTERVAL) *
              (1 - confidence),
        );
      }
    } else {
      // 如果confidence不是数字，尝试从reasoning中提取或设默认值
      if (this.config.getDebugMode()) {
        console.warn('Invalid confidence value received from loop detection LLM:', result);
      }
      
      // 尝试从reasoning字符串中提取confidence（如果模型返回了描述性文本）
      if (typeof result.reasoning === 'string') {
        const confidenceMatch = result.reasoning.match(/confidence[:\s]*([0-9]*\.?[0-9]+)/i);
        if (confidenceMatch) {
          const extractedConfidence = parseFloat(confidenceMatch[1]);
          if (!isNaN(extractedConfidence) && extractedConfidence >= 0 && extractedConfidence <= 1) {
            if (extractedConfidence > 0.9) {
              console.warn('Loop detected based on extracted confidence:', result.reasoning);
              logLoopDetected(
                this.config,
                new LoopDetectedEvent(LoopType.LLM_DETECTED_LOOP, this.promptId),
              );
              return true;
            }
          }
        }
      }
      
      // 最后的备用方案：使用基于句子的检测
      return this.performFallbackLoopDetection();
    }
    return false;
  }

  /**
   * 当LLM检测失败时的备用循环检测方法
   * 基于对话历史进行句子级别的重复检测
   */
  private performFallbackLoopDetection(): boolean {
    const recentHistory = this.config
      .getGeminiClient()
      .getHistory()
      .slice(-10); // 检查最近10轮对话

    // 提取所有助手的文本内容
    const assistantTexts: string[] = [];
    for (const historyItem of recentHistory) {
      if (historyItem.role === 'model' && historyItem.parts) {
        for (const part of historyItem.parts) {
          if ('text' in part && part.text) {
            assistantTexts.push(part.text);
          }
        }
      }
    }

    if (assistantTexts.length < 3) {
      return false; // 历史不够多，无法判断
    }

    // 使用统一的句子检测逻辑
    return this.detectRepeatedContent(assistantTexts.join(' '), 3);
  }

  /**
   * 统一的重复内容检测方法
   * @param content 要检测的内容
   * @param threshold 重复阈值
   * @returns 是否检测到重复
   */
  private detectRepeatedContent(content: string, threshold: number): boolean {
    // 使用改进的句子分割
    const sentences = content.match(/[^.!?]*[.!?]+/g) || [];
    
    // 统计句子重复，使用相似度检测而不是完全匹配
    const sentenceCount = new Map<string, number>();
    for (const sentence of sentences) {
      const trimmed = sentence.trim();
      if (trimmed.length > 10) { // 忽略太短的句子
        // 简单的相似度检测：去除多余空格和标点差异
        const normalized = trimmed.toLowerCase().replace(/\s+/g, ' ').replace(/[.,;:!?]+$/, '');
        sentenceCount.set(normalized, (sentenceCount.get(normalized) || 0) + 1);
      }
    }

    // 检查是否有句子重复超过阈值
    for (const [sentence, count] of sentenceCount) {
      if (count >= threshold) {
        if (this.config.getDebugMode()) {
          console.warn('Fallback loop detection: repeated content detected:', sentence);
        }
        logLoopDetected(
          this.config,
          new LoopDetectedEvent(LoopType.CHANTING_IDENTICAL_SENTENCES, this.promptId),
        );
        return true;
      }
    }

    return false;
  }

  /**
   * Resets all loop detection state.
   */
  reset(promptId: string): void {
    this.promptId = promptId;
    this.resetToolCallCount();
    this.resetContentTracking();
    this.resetLlmCheckTracking();
    this.resetSentenceTracking();
    this.loopDetected = false;
    this.inCodeBlock = false; // 重置代码块状态
  }

  private resetToolCallCount(): void {
    this.lastToolCallKey = null;
    this.toolCallRepetitionCount = 0;
  }

  private resetContentTracking(resetHistory = true): void {
    if (resetHistory) {
      this.streamContentHistory = '';
    }
    this.contentStats.clear();
    this.lastContentIndex = 0;
  }

  private resetLlmCheckTracking(): void {
    this.turnsInCurrentPrompt = 0;
    this.llmCheckInterval = DEFAULT_LLM_CHECK_INTERVAL;
    this.lastCheckTurn = 0;
  }

  private resetSentenceTracking(): void {
    this.lastRepeatedSentence = '';
    this.sentenceRepetitionCount = 0;
    this.partialSentenceContent = '';
  }
}
