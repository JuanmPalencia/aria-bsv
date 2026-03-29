/**
 * aria-bsv/streaming — Token-by-token streaming auditor.
 *
 * Mirrors the Python SDK's ``aria.streaming`` module.
 * Accumulates text chunks from a streaming LLM response into a single
 * {@link AuditRecord} that is recorded once the stream is finished.
 *
 * @example
 * ```ts
 * const auditor = new ARIAStreamingAuditor({ system_id: "my-system" });
 *
 * // Sync (for-loop style)
 * const session = auditor.startStream("gpt-4o", userInput);
 * for await (const chunk of openaiStream) {
 *   session.addChunk(chunk.choices[0]?.delta?.content ?? "");
 * }
 * const recordId = await session.finish();
 *
 * // Auto-finish via withStream helper
 * const recordId = await auditor.withStream("gpt-4o", input, async (s) => {
 *   for await (const chunk of stream) s.addChunk(chunk.text);
 * });
 * ```
 */

import { InferenceAuditor } from "./auditor.js";
import type { AuditRecord, AuditConfig } from "./types.js";

// ---------------------------------------------------------------------------
// StreamConfig
// ---------------------------------------------------------------------------

export interface StreamConfig {
  /** Model identifier. */
  model_id: string;
  /** Input data (will be hashed). */
  input_data: unknown;
  /** Optional confidence score [0, 1]. */
  confidence?: number | null;
  /** Optional caller metadata. */
  metadata?: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// StreamingSession
// ---------------------------------------------------------------------------

/**
 * Accumulates streaming chunks and records the result when finished.
 *
 * Obtain via {@link ARIAStreamingAuditor.startStream}.
 */
export class StreamingSession {
  private readonly _config: StreamConfig;
  private readonly _auditor: InferenceAuditor;
  private readonly _chunks: string[] = [];
  private _finished = false;
  private _record: AuditRecord | null = null;

  /** @internal */
  constructor(config: StreamConfig, auditor: InferenceAuditor) {
    this._config = config;
    this._auditor = auditor;
  }

  /**
   * Append a text chunk. Empty strings are silently ignored.
   */
  addChunk(text: string): void {
    if (text.length > 0) {
      this._chunks.push(text);
    }
  }

  /**
   * Finish the stream and record the full accumulated output.
   *
   * Idempotent — calling finish() a second time returns the same record.
   *
   * @param confidence Override the confidence score set at construction time.
   * @returns The created {@link AuditRecord}.
   */
  async finish(confidence?: number | null): Promise<AuditRecord> {
    if (this._finished && this._record) {
      return this._record;
    }
    this._finished = true;
    const output = this._chunks.join("");
    this._record = await this._auditor.record(
      this._config.model_id,
      this._config.input_data,
      output,
      {
        confidence: confidence ?? this._config.confidence ?? null,
        metadata: this._config.metadata,
      }
    );
    return this._record;
  }

  /** Accumulated text so far (before finish). */
  get accumulated(): string {
    return this._chunks.join("");
  }

  /** Number of chunks received. */
  get chunkCount(): number {
    return this._chunks.length;
  }

  /** True after finish() has been called. */
  get isFinished(): boolean {
    return this._finished;
  }

  /** The produced record (null until finish() resolves). */
  get record(): AuditRecord | null {
    return this._record;
  }
}

// ---------------------------------------------------------------------------
// ARIAStreamingAuditor
// ---------------------------------------------------------------------------

/**
 * Auditor specialised for streaming LLM responses.
 *
 * Wraps {@link InferenceAuditor} and provides a {@link StreamingSession}
 * that accumulates token-by-token output before recording a single
 * {@link AuditRecord}.
 */
export class ARIAStreamingAuditor {
  private readonly _auditor: InferenceAuditor;

  constructor(config: AuditConfig) {
    this._auditor = new InferenceAuditor(config);
  }

  /** Access the underlying {@link InferenceAuditor}. */
  get auditor(): InferenceAuditor {
    return this._auditor;
  }

  /**
   * Start a new streaming session.
   *
   * @param model_id    Model identifier.
   * @param input_data  Input that triggered the stream (will be hashed).
   * @param options     Optional confidence and metadata.
   */
  startStream(
    model_id: string,
    input_data: unknown,
    options: { confidence?: number | null; metadata?: Record<string, unknown> } = {}
  ): StreamingSession {
    const config: StreamConfig = {
      model_id,
      input_data,
      confidence: options.confidence ?? null,
      metadata: options.metadata,
    };
    return new StreamingSession(config, this._auditor);
  }

  /**
   * Convenience helper: create a session, run *fn*, then finish.
   *
   * Ensures finish() is always called, even if *fn* throws.
   *
   * @param model_id   Model identifier.
   * @param input_data Input data.
   * @param fn         Async function that receives the session and accumulates
   *                   chunks.
   * @returns The produced {@link AuditRecord}.
   *
   * @example
   * ```ts
   * const record = await streamingAuditor.withStream("gpt-4o", prompt, async (s) => {
   *   for await (const chunk of stream) s.addChunk(chunk.text);
   * });
   * ```
   */
  async withStream(
    model_id: string,
    input_data: unknown,
    fn: (session: StreamingSession) => Promise<void>,
    options: { confidence?: number | null; metadata?: Record<string, unknown> } = {}
  ): Promise<AuditRecord> {
    const session = this.startStream(model_id, input_data, options);
    try {
      await fn(session);
    } finally {
      if (!session.isFinished) {
        await session.finish();
      }
    }
    return session.record!;
  }

  /**
   * Open a new epoch. Delegates to the underlying auditor.
   */
  async open(
    modelHashes: Record<string, string> = {},
    stateHash?: string
  ): Promise<string> {
    return this._auditor.open(modelHashes, stateHash);
  }

  /**
   * Flush pending records to BSV. Delegates to the underlying auditor.
   */
  async flush(): Promise<string> {
    return this._auditor.flush();
  }

  /**
   * Flush and stop the auto-flush timer.
   */
  async close(): Promise<void> {
    return this._auditor.close();
  }
}

// ---------------------------------------------------------------------------
// OpenAI streaming helper
// ---------------------------------------------------------------------------

/**
 * Wrap an OpenAI `Stream<ChatCompletionChunk>` and audit the full response.
 *
 * @example
 * ```ts
 * const stream = await openai.chat.completions.create({ stream: true, ... });
 * const { record, text } = await auditOpenAIStream(streamingAuditor, "gpt-4o", prompt, stream);
 * ```
 */
export async function auditOpenAIStream(
  auditor: ARIAStreamingAuditor,
  model_id: string,
  input_data: unknown,
  stream: AsyncIterable<{ choices?: Array<{ delta?: { content?: string | null } }> }>,
  options: { confidence?: number | null; metadata?: Record<string, unknown> } = {}
): Promise<{ record: AuditRecord; text: string }> {
  const record = await auditor.withStream(model_id, input_data, async (s) => {
    for await (const chunk of stream) {
      const content = chunk.choices?.[0]?.delta?.content ?? "";
      s.addChunk(content);
    }
  }, options);
  return { record, text: record ? "" : "" };
}

/**
 * Wrap an Anthropic `Stream<MessageStreamEvent>` and audit the full response.
 *
 * @example
 * ```ts
 * const stream = await anthropic.messages.stream({ ... });
 * const { record } = await auditAnthropicStream(streamingAuditor, "claude-opus-4-6", prompt, stream);
 * ```
 */
export async function auditAnthropicStream(
  auditor: ARIAStreamingAuditor,
  model_id: string,
  input_data: unknown,
  stream: AsyncIterable<{ type?: string; delta?: { type?: string; text?: string } }>,
  options: { confidence?: number | null; metadata?: Record<string, unknown> } = {}
): Promise<{ record: AuditRecord; text: string }> {
  const session = auditor.startStream(model_id, input_data, options);
  for await (const event of stream) {
    if (event.type === "content_block_delta" && event.delta?.type === "text_delta") {
      session.addChunk(event.delta.text ?? "");
    }
  }
  const record = await session.finish();
  return { record, text: session.accumulated };
}
