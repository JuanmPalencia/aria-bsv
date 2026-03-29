// aria-bsv/dotnet — Immutable audit record produced by InferenceAuditor.
//
// Field names match the Python/TypeScript SDK snake_case convention so that
// canonical JSON hashes are cross-SDK compatible.

namespace AriaBsv.Core;

/// <summary>
/// Immutable snapshot of a single AI inference captured by
/// <see cref="InferenceAuditor"/>.
/// </summary>
/// <param name="RecordId">Unique identifier: <c>rec_{epochId}_{seq:D6}</c>.</param>
/// <param name="SystemId">Identifier of the AI system that produced this record.</param>
/// <param name="ModelId">Identifier of the model that ran the inference.</param>
/// <param name="InputHash">SHA-256 hex digest of the canonical input.</param>
/// <param name="OutputHash">SHA-256 hex digest of the canonical output.</param>
/// <param name="Confidence">Optional model confidence score (0–1).</param>
/// <param name="LatencyMs">Wall-clock inference latency in milliseconds.</param>
/// <param name="Timestamp">ISO-8601 UTC timestamp of the record creation.</param>
/// <param name="Sequence">Zero-based sequence number within the current epoch.</param>
/// <param name="EpochId">Epoch that contains this record.</param>
/// <param name="Metadata">Caller-supplied key/value metadata (never sent on-chain).</param>
public record AuditRecord(
    string RecordId,
    string SystemId,
    string ModelId,
    string InputHash,
    string OutputHash,
    double? Confidence,
    long LatencyMs,
    string Timestamp,
    int Sequence,
    string EpochId,
    Dictionary<string, object?> Metadata);
