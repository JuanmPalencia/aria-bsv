// aria-bsv/dotnet — Client-side inference auditor (BRC-121).
//
// Mirrors the Python SDK's aria.core.auditor module:
//   - Record()   hashes input + output and appends to the current batch.
//   - Flush()    finalises the epoch: computes the Merkle root over all records
//                and fires OnFlush. Does NOT broadcast to BSV (BSV integration
//                is the responsibility of the host application).
//   - Reset()    clears all state and starts a fresh epoch.
//
// Raw inputs and outputs are NEVER stored — only their SHA-256 hashes.

namespace AriaBsv.Core;

/// <summary>
/// Snapshot of auditor counters returned by <see cref="InferenceAuditor.Stats"/>.
/// </summary>
/// <param name="TotalRecords">Total records created since construction.</param>
/// <param name="TotalFlushes">Number of successful flushes performed.</param>
/// <param name="CurrentBatchSize">Records pending in the current unflushed batch.</param>
public record AuditStats(int TotalRecords, int TotalFlushes, int CurrentBatchSize);

/// <summary>
/// Records AI inferences and batches them into verifiable epochs anchored by
/// a Merkle root (BRC-121 protocol).
/// </summary>
/// <example>
/// <code>
/// var auditor = new InferenceAuditor("my-ai-system", batchSize: 100);
/// string id = auditor.Record("gpt-4o", inputObj, outputObj, 0.97, 42);
/// var (root, records) = auditor.Flush();
/// </code>
/// </example>
public sealed class InferenceAuditor
{
    private readonly string _systemId;
    private readonly int _batchSize;

    private string _epochId;
    private int _epochSeq;
    private MerkleTree _tree;
    private readonly List<AuditRecord> _records = new();

    private int _totalRecords;
    private int _totalFlushes;

    // -------------------------------------------------------------------------
    // Events
    // -------------------------------------------------------------------------

    /// <summary>
    /// Fired at the end of every successful <see cref="Flush"/> with
    /// the computed Merkle root and the flushed records.
    /// </summary>
    public event Action<string, IReadOnlyList<AuditRecord>>? OnFlush;

    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    /// <summary>
    /// Create an auditor for <paramref name="systemId"/>.
    /// </summary>
    /// <param name="systemId">Identifier of the audited AI system.</param>
    /// <param name="batchSize">
    /// Auto-flush threshold: when the pending batch reaches this size,
    /// <see cref="Flush"/> is called automatically. 0 means no auto-flush.
    /// </param>
    public InferenceAuditor(string systemId, int batchSize = 0)
    {
        _systemId  = systemId;
        _batchSize = batchSize;
        _epochId   = GenerateEpochId();
        _tree      = new MerkleTree();
    }

    // -------------------------------------------------------------------------
    // Properties
    // -------------------------------------------------------------------------

    /// <summary>Current epoch identifier.</summary>
    public string EpochId => _epochId;

    /// <summary>Auditor counters.</summary>
    public AuditStats Stats => new(_totalRecords, _totalFlushes, _records.Count);

    // -------------------------------------------------------------------------
    // Recording
    // -------------------------------------------------------------------------

    /// <summary>
    /// Record a single AI inference with no caller metadata.
    /// Raw inputs and outputs are hashed; the original values are not stored.
    /// </summary>
    /// <returns>The unique record ID for this entry.</returns>
    public string Record(
        string modelId,
        object? input,
        object? output,
        double? confidence,
        long latencyMs)
        => RecordWithMeta(modelId, input, output, confidence, latencyMs,
                          new Dictionary<string, object?>());

    /// <summary>
    /// Record a single AI inference with additional caller-supplied metadata.
    /// </summary>
    /// <returns>The unique record ID for this entry.</returns>
    public string RecordWithMeta(
        string modelId,
        object? input,
        object? output,
        double? confidence,
        long latencyMs,
        Dictionary<string, object?> metadata)
    {
        var inputHash  = Hasher.HashObject(input);
        var outputHash = Hasher.HashObject(output);

        var seq      = _epochSeq++;
        var recordId = $"rec_{_epochId}_{seq:D6}";
        var ts       = DateTime.UtcNow.ToString("o");

        var record = new AuditRecord(
            RecordId:   recordId,
            SystemId:   _systemId,
            ModelId:    modelId,
            InputHash:  inputHash,
            OutputHash: outputHash,
            Confidence: confidence,
            LatencyMs:  latencyMs,
            Timestamp:  ts,
            Sequence:   seq,
            EpochId:    _epochId,
            Metadata:   metadata);

        _records.Add(record);
        _totalRecords++;

        // Hash the record's canonical representation and add to the Merkle tree.
        // Uses CanonicalJson so that the leaf hash is cross-SDK reproducible.
        var recordJson = Hasher.CanonicalJson(RecordToDict(record));
        _tree.AddLeaf(recordJson);

        // Auto-flush when batch size is reached
        if (_batchSize > 0 && _records.Count >= _batchSize)
            Flush();

        return recordId;
    }

    // -------------------------------------------------------------------------
    // Flush / Reset
    // -------------------------------------------------------------------------

    /// <summary>
    /// Finalise the current batch: compute the Merkle root, fire
    /// <see cref="OnFlush"/>, reset state, and return the root + records.
    /// Returns an empty root string when the batch is empty.
    /// </summary>
    public (string MerkleRoot, IReadOnlyList<AuditRecord> Records) Flush()
    {
        if (_records.Count == 0)
            return (string.Empty, Array.Empty<AuditRecord>());

        var merkleRoot = _tree.Root();
        var flushedRecords = _records.AsReadOnly();

        _totalFlushes++;
        OnFlush?.Invoke(merkleRoot, flushedRecords);

        // Keep a snapshot before resetting
        var snapshot = _records.ToList().AsReadOnly();
        Reset();
        return (merkleRoot, snapshot);
    }

    /// <summary>
    /// Clear all pending records, reset the Merkle tree, and start a new epoch.
    /// Does NOT increment the flush counter.
    /// </summary>
    public void Reset()
    {
        _records.Clear();
        _tree     = new MerkleTree();
        _epochId  = GenerateEpochId();
        _epochSeq = 0;
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    /// <summary>
    /// Convert an <see cref="AuditRecord"/> to a snake_case dictionary that
    /// matches the serialisation used by the Python and TypeScript SDKs.
    /// </summary>
    private static Dictionary<string, object?> RecordToDict(AuditRecord r) => new()
    {
        ["record_id"]   = r.RecordId,
        ["system_id"]   = r.SystemId,
        ["model_id"]    = r.ModelId,
        ["input_hash"]  = r.InputHash,
        ["output_hash"] = r.OutputHash,
        ["confidence"]  = r.Confidence,    // double? → null or number
        ["latency_ms"]  = r.LatencyMs,     // long → integer
        ["timestamp"]   = r.Timestamp,
        ["sequence"]    = (object)r.Sequence,  // int → integer
        ["epoch_id"]    = r.EpochId,
        ["metadata"]    = r.Metadata,
    };

    private static string GenerateEpochId()
    {
        var ts   = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        var rand = Random.Shared.Next(0, 0xFFFFFF).ToString("x6");
        return $"{ts}_{rand}";
    }
}
