/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  CountTokensResponse,
  GenerateContentResponse,
  GenerateContentParameters,
  CountTokensParameters,
  EmbedContentResponse,
  EmbedContentParameters,
  FinishReason,
  Part,
  Content,
  Tool,
  ToolListUnion,
  CallableTool,
  FunctionCall,
  FunctionResponse,
} from '@google/genai';
import { ContentGenerator } from './contentGenerator.js';
import OpenAI from 'openai';
import type {
  ChatCompletion,
  ChatCompletionChunk,
} from 'openai/resources/chat/index.js';
import { logApiResponse } from '../telemetry/loggers.js';
import { ApiResponseEvent } from '../telemetry/types.js';
import { Config } from '../config/config.js';

// OpenAI API type definitions for logging
interface OpenAIToolCall {
  id: string;
  type: 'function';
  function: {
    name: string;
    arguments: string;
  };
}

interface OpenAIMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string | null;
  tool_calls?: OpenAIToolCall[];
  tool_call_id?: string;
}

interface OpenAIUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  prompt_tokens_details?: {
    cached_tokens?: number;
  };
}

interface OpenAIChoice {
  index: number;
  message: OpenAIMessage;
  finish_reason: string;
}

interface OpenAIRequestFormat {
  model: string;
  messages: OpenAIMessage[];
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  tools?: unknown[];
}

interface OpenAIResponseFormat {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: OpenAIChoice[];
  usage?: OpenAIUsage;
}

export class OpenAIContentGenerator implements ContentGenerator {
  private client: OpenAI;
  private model: string;
  private config: Config;
  private streamingToolCalls: Map<
    number,
    {
      id?: string;
      name?: string;
      arguments: string;
    }
  > = new Map();

  constructor(apiKey: string, model: string, baseURL: string, config: Config) {
    this.model = model;
    this.config = config;

    // Configure timeout settings - using progressive timeouts
    const timeoutConfig = {
      // Base timeout for most requests (2 minutes)
      timeout: 120000,
      // Maximum retries for failed requests
      maxRetries: 3,
      // HTTP client options
      httpAgent: undefined, // Let the client use default agent
    };

    // Allow config to override timeout settings
    const contentGeneratorConfig = this.config.getContentGeneratorConfig();
    if (contentGeneratorConfig?.timeout) {
      timeoutConfig.timeout = contentGeneratorConfig.timeout;
    }
    if (contentGeneratorConfig?.maxRetries !== undefined) {
      timeoutConfig.maxRetries = contentGeneratorConfig.maxRetries;
    }

    this.client = new OpenAI({
      apiKey,
      baseURL,
      timeout: timeoutConfig.timeout,
      maxRetries: timeoutConfig.maxRetries,
    });
  }

  /**
   * Check if an error is a timeout error
   */
  private isTimeoutError(error: unknown): boolean {
    if (!error) return false;

    const errorMessage =
      error instanceof Error
        ? error.message.toLowerCase()
        : String(error).toLowerCase();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const errorCode = (error as any)?.code;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const errorType = (error as any)?.type;

    // Check for common timeout indicators
    return (
      errorMessage.includes('timeout') ||
      errorMessage.includes('timed out') ||
      errorMessage.includes('connection timeout') ||
      errorMessage.includes('request timeout') ||
      errorMessage.includes('read timeout') ||
      errorMessage.includes('etimedout') || // Include ETIMEDOUT in message check
      errorMessage.includes('esockettimedout') || // Include ESOCKETTIMEDOUT in message check
      errorCode === 'ETIMEDOUT' ||
      errorCode === 'ESOCKETTIMEDOUT' ||
      errorType === 'timeout' ||
      // OpenAI specific timeout indicators
      errorMessage.includes('request timed out') ||
      errorMessage.includes('deadline exceeded')
    );
  }

  async generateContent(
    request: GenerateContentParameters,
  ): Promise<GenerateContentResponse> {
    const startTime = Date.now();
    const messages = this.convertToOpenAIFormat(request);

    try {
      // Build sampling parameters with clear priority:
      // 1. Request-level parameters (highest priority)
      // 2. Config-level sampling parameters (medium priority)
      // 3. Default values (lowest priority)
      const samplingParams = this.buildSamplingParameters(request);

      const createParams: Parameters<
        typeof this.client.chat.completions.create
      >[0] = {
        model: this.model,
        messages,
        ...samplingParams,
      };

      // Handle JSON schema if provided
      if (request.config?.responseSchema && request.config?.responseMimeType === 'application/json') {
        createParams.response_format = {
          type: 'json_schema',
          json_schema: {
            name: 'response',
            schema: this.convertGeminiSchemaToOpenAI(request.config.responseSchema),
            strict: true
          }
        };
      }

      if (request.config?.tools) {
        createParams.tools = await this.convertGeminiToolsToOpenAI(
          request.config.tools,
        );
      }
      // console.log('createParams', createParams);
      const completion = (await this.client.chat.completions.create(
        createParams,
      )) as ChatCompletion;

      const response = this.convertToGeminiFormat(completion);
      const durationMs = Date.now() - startTime;

      // Log API response event for UI telemetry
      const responseEvent = new ApiResponseEvent(
        this.model,
        durationMs,
        `openai-${Date.now()}`, // Generate a prompt ID
        this.config.getContentGeneratorConfig()?.authType,
        response.usageMetadata,
      );

      logApiResponse(this.config, responseEvent);

      return response;
    } catch (error) {
      const durationMs = Date.now() - startTime;

      // Identify timeout errors specifically
      const isTimeoutError = this.isTimeoutError(error);
      const errorMessage = isTimeoutError
        ? `Request timeout after ${Math.round(durationMs / 1000)}s. Try reducing input length or increasing timeout in config.`
        : error instanceof Error
          ? error.message
          : String(error);

      // Estimate token usage even when there's an error
      // This helps track costs and usage even for failed requests
      let estimatedUsage;
      try {
        const tokenCountResult = await this.countTokens({
          contents: request.contents,
          model: this.model,
        });
        estimatedUsage = {
          promptTokenCount: tokenCountResult.totalTokens,
          candidatesTokenCount: 0, // No completion tokens since request failed
          totalTokenCount: tokenCountResult.totalTokens,
        };
      } catch {
        // If token counting also fails, provide a minimal estimate
        const contentStr = JSON.stringify(request.contents);
        const estimatedTokens = Math.ceil(contentStr.length / 4);
        estimatedUsage = {
          promptTokenCount: estimatedTokens,
          candidatesTokenCount: 0,
          totalTokenCount: estimatedTokens,
        };
      }

      // Log API error event for UI telemetry with estimated usage
      const errorEvent = new ApiResponseEvent(
        this.model,
        durationMs,
        `openai-${Date.now()}`, // Generate a prompt ID
        this.config.getContentGeneratorConfig()?.authType,
        estimatedUsage,
        undefined,
        errorMessage,
      );
      logApiResponse(this.config, errorEvent);

      console.error('OpenAI API Error:', errorMessage);

      // Provide helpful timeout-specific error message
      if (isTimeoutError) {
        throw new Error(
          `${errorMessage}\n\nTroubleshooting tips:\n` +
            `- Reduce input length or complexity\n` +
            `- Increase timeout in config: contentGenerator.timeout\n` +
            `- Check network connectivity\n` +
            `- Consider using streaming mode for long responses`,
        );
      }

      throw new Error(`OpenAI API error: ${errorMessage}`);
    }
  }

  async generateContentStream(
    request: GenerateContentParameters,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const startTime = Date.now();
    const messages = this.convertToOpenAIFormat(request);

    try {
      // Build sampling parameters with clear priority
      const samplingParams = this.buildSamplingParameters(request);

      const createParams: Parameters<
        typeof this.client.chat.completions.create
      >[0] = {
        model: this.model,
        messages,
        ...samplingParams,
        stream: true,
        stream_options: { include_usage: true },
      };

      // Handle JSON schema if provided
      if (request.config?.responseSchema && request.config?.responseMimeType === 'application/json') {
        createParams.response_format = {
          type: 'json_schema',
          json_schema: {
            name: 'response',
            schema: this.convertGeminiSchemaToOpenAI(request.config.responseSchema),
            strict: true
          }
        };
      }

      if (request.config?.tools) {
        createParams.tools = await this.convertGeminiToolsToOpenAI(
          request.config.tools,
        );
      }

      // console.log('createParams', createParams);

      const stream = (await this.client.chat.completions.create(
        createParams,
      )) as AsyncIterable<ChatCompletionChunk>;

      const originalStream = this.streamGenerator(stream);

      // Collect all responses for final logging (don't log during streaming)
      const responses: GenerateContentResponse[] = [];

      // Return a new generator that both yields responses and collects them
      const wrappedGenerator = async function* (this: OpenAIContentGenerator) {
        try {
          for await (const response of originalStream) {
            responses.push(response);
            yield response;
          }

          const durationMs = Date.now() - startTime;

          // Get final usage metadata from the last response that has it
          const finalUsageMetadata = responses
            .slice()
            .reverse()
            .find((r) => r.usageMetadata)?.usageMetadata;

          // Log API response event for UI telemetry
          const responseEvent = new ApiResponseEvent(
            this.model,
            durationMs,
            `openai-stream-${Date.now()}`, // Generate a prompt ID
            this.config.getContentGeneratorConfig()?.authType,
            finalUsageMetadata,
          );

          logApiResponse(this.config, responseEvent);
        } catch (error) {
          const durationMs = Date.now() - startTime;

          // Identify timeout errors specifically for streaming
          const isTimeoutError = this.isTimeoutError(error);
          const errorMessage = isTimeoutError
            ? `Streaming request timeout after ${Math.round(durationMs / 1000)}s. Try reducing input length or increasing timeout in config.`
            : error instanceof Error
              ? error.message
              : String(error);

          // Estimate token usage even when there's an error in streaming
          let estimatedUsage;
          try {
            const tokenCountResult = await this.countTokens({
              contents: request.contents,
              model: this.model,
            });
            estimatedUsage = {
              promptTokenCount: tokenCountResult.totalTokens,
              candidatesTokenCount: 0, // No completion tokens since request failed
              totalTokenCount: tokenCountResult.totalTokens,
            };
          } catch {
            // If token counting also fails, provide a minimal estimate
            const contentStr = JSON.stringify(request.contents);
            const estimatedTokens = Math.ceil(contentStr.length / 4);
            estimatedUsage = {
              promptTokenCount: estimatedTokens,
              candidatesTokenCount: 0,
              totalTokenCount: estimatedTokens,
            };
          }

          // Log API error event for UI telemetry with estimated usage
          const errorEvent = new ApiResponseEvent(
            this.model,
            durationMs,
            `openai-stream-${Date.now()}`, // Generate a prompt ID
            this.config.getContentGeneratorConfig()?.authType,
            estimatedUsage,
            undefined,
            errorMessage,
          );
          logApiResponse(this.config, errorEvent);
          // Provide helpful timeout-specific error message for streaming
          if (isTimeoutError) {
            throw new Error(
              `${errorMessage}\n\nStreaming timeout troubleshooting:\n` +
                `- Reduce input length or complexity\n` +
                `- Increase timeout in config: contentGenerator.timeout\n` +
                `- Check network stability for streaming connections\n` +
                `- Consider using non-streaming mode for very long inputs`,
            );
          }

          throw error;
        }
      }.bind(this);

      return wrappedGenerator();
    } catch (error) {
      const durationMs = Date.now() - startTime;

      // Identify timeout errors specifically for streaming setup
      const isTimeoutError = this.isTimeoutError(error);
      const errorMessage = isTimeoutError
        ? `Streaming setup timeout after ${Math.round(durationMs / 1000)}s. Try reducing input length or increasing timeout in config.`
        : error instanceof Error
          ? error.message
          : String(error);

      // Estimate token usage even when there's an error in streaming setup
      let estimatedUsage;
      try {
        const tokenCountResult = await this.countTokens({
          contents: request.contents,
          model: this.model,
        });
        estimatedUsage = {
          promptTokenCount: tokenCountResult.totalTokens,
          candidatesTokenCount: 0, // No completion tokens since request failed
          totalTokenCount: tokenCountResult.totalTokens,
        };
      } catch {
        // If token counting also fails, provide a minimal estimate
        const contentStr = JSON.stringify(request.contents);
        const estimatedTokens = Math.ceil(contentStr.length / 4);
        estimatedUsage = {
          promptTokenCount: estimatedTokens,
          candidatesTokenCount: 0,
          totalTokenCount: estimatedTokens,
        };
      }

      // Log API error event for UI telemetry with estimated usage
      const errorEvent = new ApiResponseEvent(
        this.model,
        durationMs,
        `openai-stream-${Date.now()}`, // Generate a prompt ID
        this.config.getContentGeneratorConfig()?.authType,
        estimatedUsage,
        undefined,
        errorMessage,
      );
      logApiResponse(this.config, errorEvent);

      console.error('OpenAI API Streaming Error:', errorMessage);

      // Provide helpful timeout-specific error message for streaming setup
      if (isTimeoutError) {
        throw new Error(
          `${errorMessage}\n\nStreaming setup timeout troubleshooting:\n` +
            `- Reduce input length or complexity\n` +
            `- Increase timeout in config: contentGenerator.timeout\n` +
            `- Check network connectivity and firewall settings\n` +
            `- Consider using non-streaming mode for very long inputs`,
        );
      }

      throw new Error(`OpenAI API error: ${errorMessage}`);
    }
  }

  private async *streamGenerator(
    stream: AsyncIterable<ChatCompletionChunk>,
  ): AsyncGenerator<GenerateContentResponse> {
    // Reset the accumulator for each new stream
    this.streamingToolCalls.clear();

    for await (const chunk of stream) {
      yield this.convertStreamChunkToGeminiFormat(chunk);
    }
  }

  /**
   * Combine streaming responses for logging purposes
   */
  private combineStreamResponsesForLogging(
    responses: GenerateContentResponse[],
  ): GenerateContentResponse {
    if (responses.length === 0) {
      return new GenerateContentResponse();
    }

    const lastResponse = responses[responses.length - 1];

    // Find the last response with usage metadata
    const finalUsageMetadata = responses
      .slice()
      .reverse()
      .find((r) => r.usageMetadata)?.usageMetadata;

    // Combine all text content from the stream
    const combinedParts: Part[] = [];
    let combinedText = '';
    const functionCalls: Part[] = [];

    for (const response of responses) {
      if (response.candidates?.[0]?.content?.parts) {
        for (const part of response.candidates[0].content.parts) {
          if ('text' in part && part.text) {
            combinedText += part.text;
          } else if ('functionCall' in part && part.functionCall) {
            functionCalls.push(part);
          }
        }
      }
    }

    // Add combined text if any
    if (combinedText) {
      combinedParts.push({ text: combinedText });
    }

    // Add function calls
    combinedParts.push(...functionCalls);

    // Create combined response
    const combinedResponse = new GenerateContentResponse();
    combinedResponse.candidates = [
      {
        content: {
          parts: combinedParts,
          role: 'model' as const,
        },
        finishReason:
          responses[responses.length - 1]?.candidates?.[0]?.finishReason ||
          FinishReason.FINISH_REASON_UNSPECIFIED,
        index: 0,
        safetyRatings: [],
      },
    ];
    combinedResponse.responseId = lastResponse?.responseId;
    combinedResponse.createTime = lastResponse?.createTime;
    combinedResponse.modelVersion = this.model;
    combinedResponse.promptFeedback = { safetyRatings: [] };
    combinedResponse.usageMetadata = finalUsageMetadata;

    return combinedResponse;
  }

  async countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    // Use tiktoken for accurate token counting
    const content = JSON.stringify(request.contents);
    let totalTokens = 0;

    try {
      const { get_encoding } = await import('tiktoken');
      const encoding = get_encoding('cl100k_base'); // GPT-4 encoding, but estimate for qwen
      totalTokens = encoding.encode(content).length;
      encoding.free();
    } catch (error) {
      console.warn(
        'Failed to load tiktoken, falling back to character approximation:',
        error,
      );
      // Fallback: rough approximation using character count
      totalTokens = Math.ceil(content.length / 4); // Rough estimate: 1 token ≈ 4 characters
    }

    return {
      totalTokens,
    };
  }

  async embedContent(
    request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    // Extract text from contents
    let text = '';
    if (Array.isArray(request.contents)) {
      text = request.contents
        .map((content) => {
          if (typeof content === 'string') return content;
          if ('parts' in content && content.parts) {
            return content.parts
              .map((part) =>
                typeof part === 'string'
                  ? part
                  : 'text' in part
                    ? (part as { text?: string }).text || ''
                    : '',
              )
              .join(' ');
          }
          return '';
        })
        .join(' ');
    } else if (request.contents) {
      if (typeof request.contents === 'string') {
        text = request.contents;
      } else if ('parts' in request.contents && request.contents.parts) {
        text = request.contents.parts
          .map((part: Part) =>
            typeof part === 'string' ? part : 'text' in part ? part.text : '',
          )
          .join(' ');
      }
    }

    try {
      const embedding = await this.client.embeddings.create({
        model: 'text-embedding-ada-002', // Default embedding model
        input: text,
      });

      return {
        embeddings: [
          {
            values: embedding.data[0].embedding,
          },
        ],
      };
    } catch (error) {
      console.error('OpenAI API Embedding Error:', error);
      throw new Error(
        `OpenAI API error: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  private convertGeminiSchemaToOpenAI(
    geminiSchema: Record<string, unknown> | unknown
  ): Record<string, unknown> {
    if (!geminiSchema || typeof geminiSchema !== 'object') {
      return geminiSchema as Record<string, unknown>;
    }

    const converted = JSON.parse(JSON.stringify(geminiSchema));

    const convertTypes = (obj: unknown): unknown => {
      if (typeof obj !== 'object' || obj === null) {
        return obj;
      }

      if (Array.isArray(obj)) {
        return obj.map(convertTypes);
      }

      const result: Record<string, unknown> = {};
      for (const [key, value] of Object.entries(obj)) {
        if (key === 'type' && typeof value === 'string') {
          // Convert Gemini types to OpenAI JSON Schema types
          const lowerValue = value.toLowerCase();
          if (lowerValue === 'integer') {
            result[key] = 'integer';
          } else if (lowerValue === 'number') {
            result[key] = 'number';
          } else if (lowerValue === 'string') {
            result[key] = 'string';
          } else if (lowerValue === 'boolean') {
            result[key] = 'boolean';
          } else if (lowerValue === 'array') {
            result[key] = 'array';
          } else if (lowerValue === 'object') {
            result[key] = 'object';
          } else {
            result[key] = lowerValue;
          }
        } else if (
          key === 'minimum' ||
          key === 'maximum' ||
          key === 'multipleOf'
        ) {
          // Ensure numeric constraints are actual numbers, not strings
          if (typeof value === 'string' && !isNaN(Number(value))) {
            result[key] = Number(value);
          } else {
            result[key] = value;
          }
        } else if (
          key === 'minLength' ||
          key === 'maxLength' ||
          key === 'minItems' ||
          key === 'maxItems'
        ) {
          // Ensure length constraints are integers, not strings
          if (typeof value === 'string' && !isNaN(Number(value))) {
            result[key] = parseInt(value, 10);
          } else {
            result[key] = value;
          }
        } else if (typeof value === 'object') {
          result[key] = convertTypes(value);
        } else {
          result[key] = value;
        }
      }
      return result;
    };

    return convertTypes(converted) as Record<string, unknown>;
  }

  private convertGeminiParametersToOpenAI(
    parameters: Record<string, unknown>,
  ): Record<string, unknown> | undefined {
    if (!parameters || typeof parameters !== 'object') {
      return parameters;
    }

    const converted = JSON.parse(JSON.stringify(parameters));

    const convertTypes = (obj: unknown): unknown => {
      if (typeof obj !== 'object' || obj === null) {
        return obj;
      }

      if (Array.isArray(obj)) {
        return obj.map(convertTypes);
      }

      const result: Record<string, unknown> = {};
      for (const [key, value] of Object.entries(obj)) {
        if (key === 'type' && typeof value === 'string') {
          // Convert Gemini types to OpenAI JSON Schema types
          const lowerValue = value.toLowerCase();
          if (lowerValue === 'integer') {
            result[key] = 'integer';
          } else if (lowerValue === 'number') {
            result[key] = 'number';
          } else {
            result[key] = lowerValue;
          }
        } else if (
          key === 'minimum' ||
          key === 'maximum' ||
          key === 'multipleOf'
        ) {
          // Ensure numeric constraints are actual numbers, not strings
          if (typeof value === 'string' && !isNaN(Number(value))) {
            result[key] = Number(value);
          } else {
            result[key] = value;
          }
        } else if (
          key === 'minLength' ||
          key === 'maxLength' ||
          key === 'minItems' ||
          key === 'maxItems'
        ) {
          // Ensure length constraints are integers, not strings
          if (typeof value === 'string' && !isNaN(Number(value))) {
            result[key] = parseInt(value, 10);
          } else {
            result[key] = value;
          }
        } else if (typeof value === 'object') {
          result[key] = convertTypes(value);
        } else {
          result[key] = value;
        }
      }
      return result;
    };

    return convertTypes(converted) as Record<string, unknown> | undefined;
  }

  private async convertGeminiToolsToOpenAI(
    geminiTools: ToolListUnion,
  ): Promise<OpenAI.Chat.ChatCompletionTool[]> {
    const openAITools: OpenAI.Chat.ChatCompletionTool[] = [];

    for (const tool of geminiTools) {
      let actualTool: Tool;

      // Handle CallableTool vs Tool
      if ('tool' in tool) {
        // This is a CallableTool
        actualTool = await (tool as CallableTool).tool();
      } else {
        // This is already a Tool
        actualTool = tool as Tool;
      }

      if (actualTool.functionDeclarations) {
        for (const func of actualTool.functionDeclarations) {
          if (func.name && func.description) {
            openAITools.push({
              type: 'function',
              function: {
                name: func.name,
                description: func.description,
                parameters: this.convertGeminiParametersToOpenAI(
                  (func.parameters || {}) as Record<string, unknown>,
                ),
              },
            });
          }
        }
      }
    }

    // console.log(
    //   'OpenAI Tools Parameters:',
    //   JSON.stringify(openAITools, null, 2),
    // );
    return openAITools;
  }

  private convertToOpenAIFormat(
    request: GenerateContentParameters,
  ): OpenAI.Chat.ChatCompletionMessageParam[] {
    const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [];

    // Handle system instruction from config
    if (request.config?.systemInstruction) {
      const systemInstruction = request.config.systemInstruction;
      let systemText = '';

      if (Array.isArray(systemInstruction)) {
        systemText = systemInstruction
          .map((content) => {
            if (typeof content === 'string') return content;
            if ('parts' in content) {
              const contentObj = content as Content;
              return (
                contentObj.parts
                  ?.map((p: Part) =>
                    typeof p === 'string' ? p : 'text' in p ? p.text : '',
                  )
                  .join('\n') || ''
              );
            }
            return '';
          })
          .join('\n');
      } else if (typeof systemInstruction === 'string') {
        systemText = systemInstruction;
      } else if (
        typeof systemInstruction === 'object' &&
        'parts' in systemInstruction
      ) {
        const systemContent = systemInstruction as Content;
        systemText =
          systemContent.parts
            ?.map((p: Part) =>
              typeof p === 'string' ? p : 'text' in p ? p.text : '',
            )
            .join('\n') || '';
      }

      if (systemText) {
        messages.push({
          role: 'system' as const,
          content: systemText,
        });
      }
    }

    // Handle contents
    if (Array.isArray(request.contents)) {
      for (const content of request.contents) {
        if (typeof content === 'string') {
          messages.push({ role: 'user' as const, content });
        } else if ('role' in content && 'parts' in content) {
          // Check if this content has function calls or responses
          const functionCalls: FunctionCall[] = [];
          const functionResponses: FunctionResponse[] = [];
          const textParts: string[] = [];

          for (const part of content.parts || []) {
            if (typeof part === 'string') {
              textParts.push(part);
            } else if ('text' in part && part.text) {
              textParts.push(part.text);
            } else if ('functionCall' in part && part.functionCall) {
              functionCalls.push(part.functionCall);
            } else if ('functionResponse' in part && part.functionResponse) {
              functionResponses.push(part.functionResponse);
            }
          }

          // Handle function responses (tool results)
          if (functionResponses.length > 0) {
            for (const funcResponse of functionResponses) {
              messages.push({
                role: 'tool' as const,
                tool_call_id: funcResponse.id || '',
                content:
                  typeof funcResponse.response === 'string'
                    ? funcResponse.response
                    : JSON.stringify(funcResponse.response),
              });
            }
          }
          // Handle model messages with function calls
          else if (content.role === 'model' && functionCalls.length > 0) {
            const toolCalls = functionCalls.map((fc, index) => ({
              id: fc.id || `call_${index}`,
              type: 'function' as const,
              function: {
                name: fc.name || '',
                arguments: JSON.stringify(fc.args || {}),
              },
            }));

            messages.push({
              role: 'assistant' as const,
              content: textParts.join('\n') || null,
              tool_calls: toolCalls,
            });
          }
          // Handle regular text messages
          else {
            const role =
              content.role === 'model'
                ? ('assistant' as const)
                : ('user' as const);
            const text = textParts.join('\n');
            if (text) {
              messages.push({ role, content: text });
            }
          }
        }
      }
    } else if (request.contents) {
      if (typeof request.contents === 'string') {
        messages.push({ role: 'user' as const, content: request.contents });
      } else if ('role' in request.contents && 'parts' in request.contents) {
        const content = request.contents;
        const role =
          content.role === 'model' ? ('assistant' as const) : ('user' as const);
        const text =
          content.parts
            ?.map((p: Part) =>
              typeof p === 'string' ? p : 'text' in p ? p.text : '',
            )
            .join('\n') || '';
        messages.push({ role, content: text });
      }
    }

    // Clean up orphaned tool calls and merge consecutive assistant messages
    const cleanedMessages = this.cleanOrphanedToolCalls(messages);
    return this.mergeConsecutiveAssistantMessages(cleanedMessages);
  }

  /**
   * Clean up orphaned tool calls from message history to prevent OpenAI API errors
   */
  private cleanOrphanedToolCalls(
    messages: OpenAI.Chat.ChatCompletionMessageParam[],
  ): OpenAI.Chat.ChatCompletionMessageParam[] {
    const cleaned: OpenAI.Chat.ChatCompletionMessageParam[] = [];
    const toolCallIds = new Set<string>();
    const toolResponseIds = new Set<string>();

    // First pass: collect all tool call IDs and tool response IDs
    for (const message of messages) {
      if (
        message.role === 'assistant' &&
        'tool_calls' in message &&
        message.tool_calls
      ) {
        for (const toolCall of message.tool_calls) {
          if (toolCall.id) {
            toolCallIds.add(toolCall.id);
          }
        }
      } else if (
        message.role === 'tool' &&
        'tool_call_id' in message &&
        message.tool_call_id
      ) {
        toolResponseIds.add(message.tool_call_id);
      }
    }

    // Second pass: filter out orphaned messages
    for (const message of messages) {
      if (
        message.role === 'assistant' &&
        'tool_calls' in message &&
        message.tool_calls
      ) {
        // Filter out tool calls that don't have corresponding responses
        const validToolCalls = message.tool_calls.filter(
          (toolCall) => toolCall.id && toolResponseIds.has(toolCall.id),
        );

        if (validToolCalls.length > 0) {
          // Keep the message but only with valid tool calls
          const cleanedMessage = { ...message };
          (
            cleanedMessage as OpenAI.Chat.ChatCompletionMessageParam & {
              tool_calls?: OpenAI.Chat.ChatCompletionMessageToolCall[];
            }
          ).tool_calls = validToolCalls;
          cleaned.push(cleanedMessage);
        } else if (
          typeof message.content === 'string' &&
          message.content.trim()
        ) {
          // Keep the message if it has text content, but remove tool calls
          const cleanedMessage = { ...message };
          delete (
            cleanedMessage as OpenAI.Chat.ChatCompletionMessageParam & {
              tool_calls?: OpenAI.Chat.ChatCompletionMessageToolCall[];
            }
          ).tool_calls;
          cleaned.push(cleanedMessage);
        }
        // If no valid tool calls and no content, skip the message entirely
      } else if (
        message.role === 'tool' &&
        'tool_call_id' in message &&
        message.tool_call_id
      ) {
        // Only keep tool responses that have corresponding tool calls
        if (toolCallIds.has(message.tool_call_id)) {
          cleaned.push(message);
        }
      } else {
        // Keep all other messages as-is
        cleaned.push(message);
      }
    }

    // Final validation: ensure every assistant message with tool_calls has corresponding tool responses
    const finalCleaned: OpenAI.Chat.ChatCompletionMessageParam[] = [];
    const finalToolCallIds = new Set<string>();

    // Collect all remaining tool call IDs
    for (const message of cleaned) {
      if (
        message.role === 'assistant' &&
        'tool_calls' in message &&
        message.tool_calls
      ) {
        for (const toolCall of message.tool_calls) {
          if (toolCall.id) {
            finalToolCallIds.add(toolCall.id);
          }
        }
      }
    }

    // Verify all tool calls have responses
    const finalToolResponseIds = new Set<string>();
    for (const message of cleaned) {
      if (
        message.role === 'tool' &&
        'tool_call_id' in message &&
        message.tool_call_id
      ) {
        finalToolResponseIds.add(message.tool_call_id);
      }
    }

    // Remove any remaining orphaned tool calls
    for (const message of cleaned) {
      if (
        message.role === 'assistant' &&
        'tool_calls' in message &&
        message.tool_calls
      ) {
        const finalValidToolCalls = message.tool_calls.filter(
          (toolCall) => toolCall.id && finalToolResponseIds.has(toolCall.id),
        );

        if (finalValidToolCalls.length > 0) {
          const cleanedMessage = { ...message };
          (
            cleanedMessage as OpenAI.Chat.ChatCompletionMessageParam & {
              tool_calls?: OpenAI.Chat.ChatCompletionMessageToolCall[];
            }
          ).tool_calls = finalValidToolCalls;
          finalCleaned.push(cleanedMessage);
        } else if (
          typeof message.content === 'string' &&
          message.content.trim()
        ) {
          const cleanedMessage = { ...message };
          delete (
            cleanedMessage as OpenAI.Chat.ChatCompletionMessageParam & {
              tool_calls?: OpenAI.Chat.ChatCompletionMessageToolCall[];
            }
          ).tool_calls;
          finalCleaned.push(cleanedMessage);
        }
      } else {
        finalCleaned.push(message);
      }
    }

    return finalCleaned;
  }

  /**
   * Merge consecutive assistant messages to combine split text and tool calls
   */
  private mergeConsecutiveAssistantMessages(
    messages: OpenAI.Chat.ChatCompletionMessageParam[],
  ): OpenAI.Chat.ChatCompletionMessageParam[] {
    const merged: OpenAI.Chat.ChatCompletionMessageParam[] = [];

    for (const message of messages) {
      if (message.role === 'assistant' && merged.length > 0) {
        const lastMessage = merged[merged.length - 1];

        // If the last message is also an assistant message, merge them
        if (lastMessage.role === 'assistant') {
          // Combine content
          const combinedContent = [
            typeof lastMessage.content === 'string' ? lastMessage.content : '',
            typeof message.content === 'string' ? message.content : '',
          ]
            .filter(Boolean)
            .join('');

          // Combine tool calls
          const lastToolCalls =
            'tool_calls' in lastMessage ? lastMessage.tool_calls || [] : [];
          const currentToolCalls =
            'tool_calls' in message ? message.tool_calls || [] : [];
          const combinedToolCalls = [...lastToolCalls, ...currentToolCalls];

          // Update the last message with combined data
          (
            lastMessage as OpenAI.Chat.ChatCompletionMessageParam & {
              content: string | null;
              tool_calls?: OpenAI.Chat.ChatCompletionMessageToolCall[];
            }
          ).content = combinedContent || null;
          if (combinedToolCalls.length > 0) {
            (
              lastMessage as OpenAI.Chat.ChatCompletionMessageParam & {
                content: string | null;
                tool_calls?: OpenAI.Chat.ChatCompletionMessageToolCall[];
              }
            ).tool_calls = combinedToolCalls;
          }

          continue; // Skip adding the current message since it's been merged
        }
      }

      // Add the message as-is if no merging is needed
      merged.push(message);
    }

    return merged;
  }

  private convertToGeminiFormat(
    openaiResponse: ChatCompletion,
  ): GenerateContentResponse {
    const choice = openaiResponse.choices[0];
    const response = new GenerateContentResponse();

    const parts: Part[] = [];

    // Handle text content
    if (choice.message.content) {
      parts.push({ text: choice.message.content });
    }

    // Handle tool calls
    if (choice.message.tool_calls) {
      for (const toolCall of choice.message.tool_calls) {
        if (toolCall.function) {
          let args: Record<string, unknown> = {};
          if (toolCall.function.arguments) {
            try {
              args = JSON.parse(toolCall.function.arguments);
            } catch (error) {
              console.error('Failed to parse function arguments:', error);
              args = {};
            }
          }

          parts.push({
            functionCall: {
              id: toolCall.id,
              name: toolCall.function.name,
              args,
            },
          });
        }
      }
    }

    response.responseId = openaiResponse.id;
    response.createTime = openaiResponse.created.toString();

    response.candidates = [
      {
        content: {
          parts,
          role: 'model' as const,
        },
        finishReason: this.mapFinishReason(choice.finish_reason || 'stop'),
        index: 0,
        safetyRatings: [],
      },
    ];

    response.modelVersion = this.model;
    response.promptFeedback = { safetyRatings: [] };

    // Add usage metadata if available
    if (openaiResponse.usage) {
      const usage = openaiResponse.usage as OpenAIUsage;

      const promptTokens = usage.prompt_tokens || 0;
      const completionTokens = usage.completion_tokens || 0;
      const totalTokens = usage.total_tokens || 0;
      const cachedTokens = usage.prompt_tokens_details?.cached_tokens || 0;

      // If we only have total tokens but no breakdown, estimate the split
      // Typically input is ~70% and output is ~30% for most conversations
      let finalPromptTokens = promptTokens;
      let finalCompletionTokens = completionTokens;

      if (totalTokens > 0 && promptTokens === 0 && completionTokens === 0) {
        // Estimate: assume 70% input, 30% output
        finalPromptTokens = Math.round(totalTokens * 0.7);
        finalCompletionTokens = Math.round(totalTokens * 0.3);
      }

      response.usageMetadata = {
        promptTokenCount: finalPromptTokens,
        candidatesTokenCount: finalCompletionTokens,
        totalTokenCount: totalTokens,
        cachedContentTokenCount: cachedTokens,
      };
    }

    return response;
  }

  private convertStreamChunkToGeminiFormat(
    chunk: ChatCompletionChunk,
  ): GenerateContentResponse {
    const choice = chunk.choices?.[0];
    const response = new GenerateContentResponse();

    if (choice) {
      const parts: Part[] = [];

      // Handle text content
      if (choice.delta?.content) {
        parts.push({ text: choice.delta.content });
      }

      // Handle tool calls - only accumulate during streaming, emit when complete
      if (choice.delta?.tool_calls) {
        for (const toolCall of choice.delta.tool_calls) {
          const index = toolCall.index ?? 0;

          // Get or create the tool call accumulator for this index
          let accumulatedCall = this.streamingToolCalls.get(index);
          if (!accumulatedCall) {
            accumulatedCall = { arguments: '' };
            this.streamingToolCalls.set(index, accumulatedCall);
          }

          // Update accumulated data
          if (toolCall.id) {
            accumulatedCall.id = toolCall.id;
          }
          if (toolCall.function?.name) {
            accumulatedCall.name = toolCall.function.name;
          }
          if (toolCall.function?.arguments) {
            accumulatedCall.arguments += toolCall.function.arguments;
          }
        }
      }

      // Only emit function calls when streaming is complete (finish_reason is present)
      if (choice.finish_reason) {
        for (const [, accumulatedCall] of this.streamingToolCalls) {
          // TODO: Add back id once we have a way to generate tool_call_id from the VLLM parser.
          // if (accumulatedCall.id && accumulatedCall.name) {
          if (accumulatedCall.name) {
            let args: Record<string, unknown> = {};
            if (accumulatedCall.arguments) {
              try {
                args = JSON.parse(accumulatedCall.arguments);
              } catch (error) {
                console.error(
                  'Failed to parse final tool call arguments:',
                  error,
                );
              }
            }

            parts.push({
              functionCall: {
                id: accumulatedCall.id,
                name: accumulatedCall.name,
                args,
              },
            });
          }
        }
        // Clear all accumulated tool calls
        this.streamingToolCalls.clear();
      }

      response.candidates = [
        {
          content: {
            parts,
            role: 'model' as const,
          },
          finishReason: choice.finish_reason
            ? this.mapFinishReason(choice.finish_reason)
            : FinishReason.FINISH_REASON_UNSPECIFIED,
          index: 0,
          safetyRatings: [],
        },
      ];
    } else {
      response.candidates = [];
    }

    response.responseId = chunk.id;
    response.createTime = chunk.created.toString();

    response.modelVersion = this.model;
    response.promptFeedback = { safetyRatings: [] };

    // Add usage metadata if available in the chunk
    if (chunk.usage) {
      const usage = chunk.usage as OpenAIUsage;

      const promptTokens = usage.prompt_tokens || 0;
      const completionTokens = usage.completion_tokens || 0;
      const totalTokens = usage.total_tokens || 0;
      const cachedTokens = usage.prompt_tokens_details?.cached_tokens || 0;

      // If we only have total tokens but no breakdown, estimate the split
      // Typically input is ~70% and output is ~30% for most conversations
      let finalPromptTokens = promptTokens;
      let finalCompletionTokens = completionTokens;

      if (totalTokens > 0 && promptTokens === 0 && completionTokens === 0) {
        // Estimate: assume 70% input, 30% output
        finalPromptTokens = Math.round(totalTokens * 0.7);
        finalCompletionTokens = Math.round(totalTokens * 0.3);
      }

      response.usageMetadata = {
        promptTokenCount: finalPromptTokens,
        candidatesTokenCount: finalCompletionTokens,
        totalTokenCount: totalTokens,
        cachedContentTokenCount: cachedTokens,
      };
    }

    return response;
  }

  /**
   * Build sampling parameters with clear priority:
   * 1. Config-level sampling parameters (highest priority)
   * 2. Request-level parameters (medium priority)
   * 3. Default values (lowest priority)
   */
  private buildSamplingParameters(
    request: GenerateContentParameters,
  ): Record<string, unknown> {
    const configSamplingParams =
      this.config.getContentGeneratorConfig()?.samplingParams;

    const params = {
      // Temperature: config > request > default
      temperature:
        configSamplingParams?.temperature !== undefined
          ? configSamplingParams.temperature
          : request.config?.temperature !== undefined
            ? request.config.temperature
            : 0.0,

      // Max tokens: config > request > undefined
      ...(configSamplingParams?.max_tokens !== undefined
        ? { max_tokens: configSamplingParams.max_tokens }
        : request.config?.maxOutputTokens !== undefined
          ? { max_tokens: request.config.maxOutputTokens }
          : {}),

      // Top-p: config > request > default
      top_p:
        configSamplingParams?.top_p !== undefined
          ? configSamplingParams.top_p
          : request.config?.topP !== undefined
            ? request.config.topP
            : 1.0,

      // Top-k: config only (not available in request)
      ...(configSamplingParams?.top_k !== undefined
        ? { top_k: configSamplingParams.top_k }
        : {}),

      // Repetition penalty: config only
      ...(configSamplingParams?.repetition_penalty !== undefined
        ? { repetition_penalty: configSamplingParams.repetition_penalty }
        : {}),

      // Presence penalty: config only
      ...(configSamplingParams?.presence_penalty !== undefined
        ? { presence_penalty: configSamplingParams.presence_penalty }
        : {}),

      // Frequency penalty: config only
      ...(configSamplingParams?.frequency_penalty !== undefined
        ? { frequency_penalty: configSamplingParams.frequency_penalty }
        : {}),
    };

    return params;
  }

  private mapFinishReason(openaiReason: string | null): FinishReason {
    if (!openaiReason) return FinishReason.FINISH_REASON_UNSPECIFIED;
    const mapping: Record<string, FinishReason> = {
      stop: FinishReason.STOP,
      length: FinishReason.MAX_TOKENS,
      content_filter: FinishReason.SAFETY,
      function_call: FinishReason.STOP,
      tool_calls: FinishReason.STOP,
    };
    return mapping[openaiReason] || FinishReason.FINISH_REASON_UNSPECIFIED;
  }

  /**
   * Convert Gemini request format to OpenAI chat completion format for logging
   */
  private async convertGeminiRequestToOpenAI(
    request: GenerateContentParameters,
  ): Promise<OpenAIRequestFormat> {
    const messages: OpenAIMessage[] = [];

    // Handle system instruction
    if (request.config?.systemInstruction) {
      const systemInstruction = request.config.systemInstruction;
      let systemText = '';

      if (Array.isArray(systemInstruction)) {
        systemText = systemInstruction
          .map((content) => {
            if (typeof content === 'string') return content;
            if ('parts' in content) {
              const contentObj = content as Content;
              return (
                contentObj.parts
                  ?.map((p: Part) =>
                    typeof p === 'string' ? p : 'text' in p ? p.text : '',
                  )
                  .join('\n') || ''
              );
            }
            return '';
          })
          .join('\n');
      } else if (typeof systemInstruction === 'string') {
        systemText = systemInstruction;
      } else if (
        typeof systemInstruction === 'object' &&
        'parts' in systemInstruction
      ) {
        const systemContent = systemInstruction as Content;
        systemText =
          systemContent.parts
            ?.map((p: Part) =>
              typeof p === 'string' ? p : 'text' in p ? p.text : '',
            )
            .join('\n') || '';
      }

      if (systemText) {
        messages.push({
          role: 'system',
          content: systemText,
        });
      }
    }

    // Handle contents
    if (Array.isArray(request.contents)) {
      for (const content of request.contents) {
        if (typeof content === 'string') {
          messages.push({ role: 'user', content });
        } else if ('role' in content && 'parts' in content) {
          const functionCalls: FunctionCall[] = [];
          const functionResponses: FunctionResponse[] = [];
          const textParts: string[] = [];

          for (const part of content.parts || []) {
            if (typeof part === 'string') {
              textParts.push(part);
            } else if ('text' in part && part.text) {
              textParts.push(part.text);
            } else if ('functionCall' in part && part.functionCall) {
              functionCalls.push(part.functionCall);
            } else if ('functionResponse' in part && part.functionResponse) {
              functionResponses.push(part.functionResponse);
            }
          }

          // Handle function responses (tool results)
          if (functionResponses.length > 0) {
            for (const funcResponse of functionResponses) {
              messages.push({
                role: 'tool',
                tool_call_id: funcResponse.id || '',
                content:
                  typeof funcResponse.response === 'string'
                    ? funcResponse.response
                    : JSON.stringify(funcResponse.response),
              });
            }
          }
          // Handle model messages with function calls
          else if (content.role === 'model' && functionCalls.length > 0) {
            const toolCalls = functionCalls.map((fc, index) => ({
              id: fc.id || `call_${index}`,
              type: 'function' as const,
              function: {
                name: fc.name || '',
                arguments: JSON.stringify(fc.args || {}),
              },
            }));

            messages.push({
              role: 'assistant',
              content: textParts.join('\n') || null,
              tool_calls: toolCalls,
            });
          }
          // Handle regular text messages
          else {
            const role = content.role === 'model' ? 'assistant' : 'user';
            const text = textParts.join('\n');
            if (text) {
              messages.push({ role, content: text });
            }
          }
        }
      }
    } else if (request.contents) {
      if (typeof request.contents === 'string') {
        messages.push({ role: 'user', content: request.contents });
      } else if ('role' in request.contents && 'parts' in request.contents) {
        const content = request.contents;
        const role = content.role === 'model' ? 'assistant' : 'user';
        const text =
          content.parts
            ?.map((p: Part) =>
              typeof p === 'string' ? p : 'text' in p ? p.text : '',
            )
            .join('\n') || '';
        messages.push({ role, content: text });
      }
    }

    // Clean up orphaned tool calls and merge consecutive assistant messages
    const cleanedMessages = this.cleanOrphanedToolCallsForLogging(messages);
    const mergedMessages =
      this.mergeConsecutiveAssistantMessagesForLogging(cleanedMessages);

    const openaiRequest: OpenAIRequestFormat = {
      model: this.model,
      messages: mergedMessages,
    };

    // Add sampling parameters using the same logic as actual API calls
    const samplingParams = this.buildSamplingParameters(request);
    Object.assign(openaiRequest, samplingParams);

    // Convert tools if present
    if (request.config?.tools) {
      openaiRequest.tools = await this.convertGeminiToolsToOpenAI(
        request.config.tools,
      );
    }

    return openaiRequest;
  }

  /**
   * Clean up orphaned tool calls for logging purposes
   */
  private cleanOrphanedToolCallsForLogging(
    messages: OpenAIMessage[],
  ): OpenAIMessage[] {
    const cleaned: OpenAIMessage[] = [];
    const toolCallIds = new Set<string>();
    const toolResponseIds = new Set<string>();

    // First pass: collect all tool call IDs and tool response IDs
    for (const message of messages) {
      if (message.role === 'assistant' && message.tool_calls) {
        for (const toolCall of message.tool_calls) {
          if (toolCall.id) {
            toolCallIds.add(toolCall.id);
          }
        }
      } else if (message.role === 'tool' && message.tool_call_id) {
        toolResponseIds.add(message.tool_call_id);
      }
    }

    // Second pass: filter out orphaned messages
    for (const message of messages) {
      if (message.role === 'assistant' && message.tool_calls) {
        // Filter out tool calls that don't have corresponding responses
        const validToolCalls = message.tool_calls.filter(
          (toolCall) => toolCall.id && toolResponseIds.has(toolCall.id),
        );

        if (validToolCalls.length > 0) {
          // Keep the message but only with valid tool calls
          const cleanedMessage = { ...message };
          cleanedMessage.tool_calls = validToolCalls;
          cleaned.push(cleanedMessage);
        } else if (
          typeof message.content === 'string' &&
          message.content.trim()
        ) {
          // Keep the message if it has text content, but remove tool calls
          const cleanedMessage = { ...message };
          delete cleanedMessage.tool_calls;
          cleaned.push(cleanedMessage);
        }
        // If no valid tool calls and no content, skip the message entirely
      } else if (message.role === 'tool' && message.tool_call_id) {
        // Only keep tool responses that have corresponding tool calls
        if (toolCallIds.has(message.tool_call_id)) {
          cleaned.push(message);
        }
      } else {
        // Keep all other messages as-is
        cleaned.push(message);
      }
    }

    // Final validation: ensure every assistant message with tool_calls has corresponding tool responses
    const finalCleaned: OpenAIMessage[] = [];
    const finalToolCallIds = new Set<string>();

    // Collect all remaining tool call IDs
    for (const message of cleaned) {
      if (message.role === 'assistant' && message.tool_calls) {
        for (const toolCall of message.tool_calls) {
          if (toolCall.id) {
            finalToolCallIds.add(toolCall.id);
          }
        }
      }
    }

    // Verify all tool calls have responses
    const finalToolResponseIds = new Set<string>();
    for (const message of cleaned) {
      if (message.role === 'tool' && message.tool_call_id) {
        finalToolResponseIds.add(message.tool_call_id);
      }
    }

    // Remove any remaining orphaned tool calls
    for (const message of cleaned) {
      if (message.role === 'assistant' && message.tool_calls) {
        const finalValidToolCalls = message.tool_calls.filter(
          (toolCall) => toolCall.id && finalToolResponseIds.has(toolCall.id),
        );

        if (finalValidToolCalls.length > 0) {
          const cleanedMessage = { ...message };
          cleanedMessage.tool_calls = finalValidToolCalls;
          finalCleaned.push(cleanedMessage);
        } else if (
          typeof message.content === 'string' &&
          message.content.trim()
        ) {
          const cleanedMessage = { ...message };
          delete cleanedMessage.tool_calls;
          finalCleaned.push(cleanedMessage);
        }
      } else {
        finalCleaned.push(message);
      }
    }

    return finalCleaned;
  }

  /**
   * Merge consecutive assistant messages to combine split text and tool calls for logging
   */
  private mergeConsecutiveAssistantMessagesForLogging(
    messages: OpenAIMessage[],
  ): OpenAIMessage[] {
    const merged: OpenAIMessage[] = [];

    for (const message of messages) {
      if (message.role === 'assistant' && merged.length > 0) {
        const lastMessage = merged[merged.length - 1];

        // If the last message is also an assistant message, merge them
        if (lastMessage.role === 'assistant') {
          // Combine content
          const combinedContent = [
            lastMessage.content || '',
            message.content || '',
          ]
            .filter(Boolean)
            .join('');

          // Combine tool calls
          const combinedToolCalls = [
            ...(lastMessage.tool_calls || []),
            ...(message.tool_calls || []),
          ];

          // Update the last message with combined data
          lastMessage.content = combinedContent || null;
          if (combinedToolCalls.length > 0) {
            lastMessage.tool_calls = combinedToolCalls;
          }

          continue; // Skip adding the current message since it's been merged
        }
      }

      // Add the message as-is if no merging is needed
      merged.push(message);
    }

    return merged;
  }

  /**
   * Convert Gemini response format to OpenAI chat completion format for logging
   */
  private convertGeminiResponseToOpenAI(
    response: GenerateContentResponse,
  ): OpenAIResponseFormat {
    const candidate = response.candidates?.[0];
    const content = candidate?.content;

    let messageContent: string | null = null;
    const toolCalls: OpenAIToolCall[] = [];

    if (content?.parts) {
      const textParts: string[] = [];

      for (const part of content.parts) {
        if ('text' in part && part.text) {
          textParts.push(part.text);
        } else if ('functionCall' in part && part.functionCall) {
          toolCalls.push({
            id: part.functionCall.id || `call_${toolCalls.length}`,
            type: 'function' as const,
            function: {
              name: part.functionCall.name || '',
              arguments: JSON.stringify(part.functionCall.args || {}),
            },
          });
        }
      }

      messageContent = textParts.join('');
    }

    const choice: OpenAIChoice = {
      index: 0,
      message: {
        role: 'assistant',
        content: messageContent,
      },
      finish_reason: this.mapGeminiFinishReasonToOpenAI(
        candidate?.finishReason,
      ),
    };

    if (toolCalls.length > 0) {
      choice.message.tool_calls = toolCalls;
    }

    const openaiResponse: OpenAIResponseFormat = {
      id: response.responseId || `chatcmpl-${Date.now()}`,
      object: 'chat.completion',
      created: response.createTime
        ? Number(response.createTime)
        : Math.floor(Date.now() / 1000),
      model: this.model,
      choices: [choice],
    };

    // Add usage metadata if available
    if (response.usageMetadata) {
      openaiResponse.usage = {
        prompt_tokens: response.usageMetadata.promptTokenCount || 0,
        completion_tokens: response.usageMetadata.candidatesTokenCount || 0,
        total_tokens: response.usageMetadata.totalTokenCount || 0,
      };

      if (response.usageMetadata.cachedContentTokenCount) {
        openaiResponse.usage.prompt_tokens_details = {
          cached_tokens: response.usageMetadata.cachedContentTokenCount,
        };
      }
    }

    return openaiResponse;
  }

  /**
   * Map Gemini finish reasons to OpenAI finish reasons
   */
  private mapGeminiFinishReasonToOpenAI(geminiReason?: unknown): string {
    if (!geminiReason) return 'stop';

    switch (geminiReason) {
      case 'STOP':
      case 1: // FinishReason.STOP
        return 'stop';
      case 'MAX_TOKENS':
      case 2: // FinishReason.MAX_TOKENS
        return 'length';
      case 'SAFETY':
      case 3: // FinishReason.SAFETY
        return 'content_filter';
      case 'RECITATION':
      case 4: // FinishReason.RECITATION
        return 'content_filter';
      case 'OTHER':
      case 5: // FinishReason.OTHER
        return 'stop';
      default:
        return 'stop';
    }
  }
}
