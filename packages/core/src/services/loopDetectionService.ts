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
      this.inCodeBlock =
        numFences % 2 === 0 ? this.inCodeBlock : !this.inCodeBlock;

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
    const lastCompleteIndex =
      this.partialSentenceContent.lastIndexOf(lastSentence);
    const endOfLastSentence = lastCompleteIndex + lastSentence.length;
    this.partialSentenceContent =
      this.partialSentenceContent.slice(endOfLastSentence);

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

  /**
   * Enhanced LLM-based self-reflection check that analyzes the conversation
   * for unproductive patterns and suggests corrective actions.
   */
  private async checkForLoopWithLLM(signal: AbortSignal) {
    const recentHistory = this.config
      .getGeminiClient()
      .getHistory()
      .slice(-LLM_LOOP_CHECK_HISTORY_COUNT);

    const prompt = `You are an AI self-analysis system. Analyze this conversation to detect if the assistant is stuck in an unproductive loop.

CONVERSATION HISTORY:
${recentHistory.map((turn, i) => `Turn ${i + 1} (${turn.role}): ${JSON.stringify(turn.parts)}`).join('\n')}

ANALYSIS FRAMEWORK:
1. INTENTION-EXECUTION GAPS: Is the assistant expressing intentions to perform actions (like "让我查看", "Let me read") but not actually executing the corresponding tool calls?

2. REPETITIVE RESPONSES: Is the assistant generating similar text responses without meaningful variation or progress?

3. LACK OF FORWARD PROGRESS: Is the assistant failing to advance toward solving the user's actual request over multiple turns?

4. PATTERN RECOGNITION: Are there any other unproductive behavioral patterns (like repeatedly asking the same questions, cycling between the same few responses, etc.)?

CRITICAL EVALUATION CRITERIA:
- A true loop exists when there's NO meaningful progress toward the user's goal
- Repetitive tool calls that make incremental progress are NOT loops
- Expressing an intention without immediate execution IS a loop indicator
- Responses that are semantically identical across multiple turns indicate a loop

You MUST respond with ONLY this exact JSON format:
{
  "reasoning": "Detailed analysis of conversation patterns and whether they indicate an unproductive loop",
  "confidence": 0.85,
  "suggestedAction": "Specific recommendation for breaking the loop if one exists"
}

Requirements:
- reasoning: String analysis (20-200 characters)
- confidence: Number 0.0-1.0 (confidence that a loop exists)
- suggestedAction: String with specific next step recommendation`;

    const systemInstruction = `You are a specialized diagnostic AI that ONLY analyzes conversation patterns. 
You CANNOT use tools or execute functions. 
You MUST return exactly the specified JSON format with reasoning, confidence, and suggestedAction fields.
Higher confidence (>0.8) means you're certain an unproductive loop exists.`;

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
          description: 'Analysis of conversation patterns and loop detection',
          minLength: 20,
          maxLength: 200,
        },
        confidence: {
          type: Type.NUMBER,
          description: 'Confidence that an unproductive loop exists (0.0-1.0)',
          minimum: 0.0,
          maximum: 1.0,
        },
        suggestedAction: {
          type: Type.STRING,
          description: 'Specific recommendation for breaking the loop',
          minLength: 10,
          maxLength: 100,
        },
      },
      required: ['reasoning', 'confidence', 'suggestedAction'],
      additionalProperties: false,
    };

    let result;
    try {
      result = await this.config
        .getGeminiClient()
        .generateJson(contents, schema, signal, DEFAULT_GEMINI_FLASH_MODEL);
    } catch (e) {
      if (this.config.getDebugMode()) {
        console.error('Enhanced LLM loop detection failed:', e);
      }
      return this.performFallbackLoopDetection();
    }

    // Validate and process the result
    if (
      typeof result.confidence === 'number' &&
      result.confidence >= 0 &&
      result.confidence <= 1
    ) {
      if (result.confidence > 0.8) {
        // High confidence loop detected
        console.warn(
          `Loop detected with ${(result.confidence * 100).toFixed(1)}% confidence: ${result.reasoning}`,
        );
        if (result.suggestedAction) {
          console.warn(`Suggested action: ${result.suggestedAction}`);
        }

        logLoopDetected(
          this.config,
          new LoopDetectedEvent(LoopType.LLM_DETECTED_LOOP, this.promptId),
        );
        return true;
      } else {
        // Adjust check interval based on confidence
        this.llmCheckInterval = Math.round(
          MIN_LLM_CHECK_INTERVAL +
            (MAX_LLM_CHECK_INTERVAL - MIN_LLM_CHECK_INTERVAL) *
              (1 - result.confidence),
        );

        if (this.config.getDebugMode() && result.confidence > 0.5) {
          console.log(
            `Moderate loop risk detected: ${result.reasoning} (confidence: ${result.confidence})`,
          );
        }
      }
    } else {
      if (this.config.getDebugMode()) {
        console.warn('Invalid confidence value from loop detection:', result);
      }
      return this.performFallbackLoopDetection();
    }

    return false;
  }

  /**
   * Analyzes AI response for intention-execution alignment to prevent loops
   * where the AI states it will do something but doesn't actually do it.
   */
  async analyzeIntentionExecutionGap(
    responseText: string,
    hasToolCalls: boolean,
    conversationHistory: Array<{
      role: string;
      parts?: Array<{ text?: string }>;
    }>,
  ): Promise<{ hasGap: boolean; suggestion?: string }> {
    // Quick pattern detection for common intention phrases
    const intentionPatterns = [
      /让我查看|让我读取|让我搜索|让我分析/i,
      /let me check|let me read|let me search|let me analyze/i,
      /i'll check|i'll read|i'll search|i'll analyze/i,
      /now i'll|first i'll|next i'll/i,
    ];

    const hasIntention = intentionPatterns.some((pattern) =>
      pattern.test(responseText),
    );

    if (hasIntention && !hasToolCalls) {
      // Detected intention without execution - this is a strong loop indicator
      const recentHistory = conversationHistory.slice(-3);

      // Use LLM to analyze the specific gap and suggest action
      const analysisPrompt = `Analyze this AI response for intention-execution gaps:

RESPONSE TEXT: "${responseText}"
HAS TOOL CALLS: ${hasToolCalls}
RECENT CONTEXT: ${JSON.stringify(recentHistory)}

The AI expressed an intention to perform an action but didn't execute it. 

Respond with JSON:
{
  "hasGap": true,
  "toolNeeded": "specific tool name that should have been called",
  "parameters": "likely parameters for the tool call",
  "suggestion": "specific corrective action to take"
}`;

      try {
        const result = await this.config.getGeminiClient().generateJson(
          [{ role: 'user', parts: [{ text: analysisPrompt }] }],
          {
            type: Type.OBJECT,
            properties: {
              hasGap: { type: Type.BOOLEAN },
              toolNeeded: { type: Type.STRING },
              parameters: { type: Type.STRING },
              suggestion: { type: Type.STRING, maxLength: 100 },
            },
            required: ['hasGap', 'suggestion'],
          },
          new AbortController().signal,
          DEFAULT_GEMINI_FLASH_MODEL,
        );

        if (result.hasGap) {
          return {
            hasGap: true,
            suggestion:
              (typeof result.suggestion === 'string'
                ? result.suggestion
                : '') || 'Execute the intended action immediately',
          };
        }
      } catch (_error) {
        // Fallback to simple detection
        return {
          hasGap: true,
          suggestion: 'Execute the tool call you mentioned in your response',
        };
      }
    }

    return { hasGap: false };
  }

  /**
   * Enhanced method that combines all loop detection approaches
   */
  async comprehensiveLoopCheck(
    responseText: string,
    hasToolCalls: boolean,
    conversationHistory: Array<{
      role: string;
      parts?: Array<{ text?: string }>;
    }>,
    signal: AbortSignal,
  ): Promise<{ isLoop: boolean; reason?: string; suggestion?: string }> {
    // 1. Check intention-execution gap
    const intentionGap = await this.analyzeIntentionExecutionGap(
      responseText,
      hasToolCalls,
      conversationHistory,
    );

    if (intentionGap.hasGap) {
      return {
        isLoop: true,
        reason: 'Intention-execution gap detected',
        suggestion: intentionGap.suggestion,
      };
    }

    // 2. Check for repetitive content
    const recentResponses = conversationHistory
      .filter((turn) => turn.role === 'model')
      .slice(-3)
      .map((turn) => turn.parts?.[0]?.text || '');

    if (recentResponses.length >= 2) {
      const similarity = this.calculateResponseSimilarity(
        responseText,
        recentResponses,
      );
      if (similarity > 0.8) {
        return {
          isLoop: true,
          reason: 'High response similarity detected',
          suggestion: 'Try a completely different approach to the problem',
        };
      }
    }

    // 3. Perform LLM-based analysis for complex patterns
    const llmResult = await this.checkForLoopWithLLM(signal);
    if (llmResult) {
      return {
        isLoop: true,
        reason: 'LLM detected unproductive conversation pattern',
        suggestion: 'Change approach based on conversation analysis',
      };
    }

    return { isLoop: false };
  }

  /**
   * Simple similarity calculation for responses
   */
  private calculateResponseSimilarity(
    current: string,
    previous: string[],
  ): number {
    if (previous.length === 0) return 0;

    const currentWords = current.toLowerCase().split(/\s+/);
    const maxSimilarity = Math.max(
      ...previous.map((prev) => {
        const prevWords = prev.toLowerCase().split(/\s+/);
        const intersection = currentWords.filter((word) =>
          prevWords.includes(word),
        );
        return (
          intersection.length / Math.max(currentWords.length, prevWords.length)
        );
      }),
    );

    return maxSimilarity;
  }
  private performFallbackLoopDetection(): boolean {
    const recentHistory = this.config.getGeminiClient().getHistory().slice(-10); // 检查最近10轮对话

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
      if (trimmed.length > 10) {
        // 忽略太短的句子
        // 简单的相似度检测：去除多余空格和标点差异
        const normalized = trimmed
          .toLowerCase()
          .replace(/\s+/g, ' ')
          .replace(/[.,;:!?]+$/, '');
        sentenceCount.set(normalized, (sentenceCount.get(normalized) || 0) + 1);
      }
    }

    // 检查是否有句子重复超过阈值
    for (const [sentence, count] of sentenceCount) {
      if (count >= threshold) {
        if (this.config.getDebugMode()) {
          console.warn(
            'Fallback loop detection: repeated content detected:',
            sentence,
          );
        }
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
