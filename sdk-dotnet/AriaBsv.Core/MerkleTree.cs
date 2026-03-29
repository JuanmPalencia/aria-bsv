// aria-bsv/dotnet — RFC 6962 Merkle tree with second-preimage protection.
//
// Identical algorithm to the Python SDK's aria.core.merkle module, so Merkle
// roots computed by any BRC-121 SDK are interchangeable.
//
// Construction (RFC 6962 / BRC-121):
//   Leaf nodes:     SHA-256(0x00 || leaf_hash_bytes)
//   Internal nodes: SHA-256(0x01 || left_bytes || right_bytes)
//   Odd layer:      the last node is promoted unchanged (not duplicated).

using System.Security.Cryptography;

namespace AriaBsv.Core;

/// <summary>
/// Stateful Merkle tree that accumulates leaves and computes the RFC 6962
/// root on demand.
/// </summary>
public sealed class MerkleTree
{
    private readonly List<string> _leafHashes = new();

    // -------------------------------------------------------------------------
    // Mutation
    // -------------------------------------------------------------------------

    /// <summary>
    /// Hash <paramref name="data"/> with SHA-256 and add the result as a leaf.
    /// The stored leaf hash equals <c>SHA-256(data)</c> as a 64-char lowercase hex string.
    /// </summary>
    public void AddLeaf(byte[] data)
    {
        var hash = Hasher.HashBytes(data);
        _leafHashes.Add(hash);
    }

    // -------------------------------------------------------------------------
    // Query
    // -------------------------------------------------------------------------

    /// <summary>Number of leaves added so far.</summary>
    public int Count => _leafHashes.Count;

    /// <summary>
    /// Compute the Merkle root over all accumulated leaves.
    /// </summary>
    /// <exception cref="InvalidOperationException">When the tree is empty.</exception>
    public string Root()
    {
        if (_leafHashes.Count == 0)
            throw new InvalidOperationException(
                "Cannot compute Merkle root of an empty tree.");

        return ComputeRoot(_leafHashes);
    }

    /// <summary>
    /// Return the SHA-256 leaf hashes in insertion order.
    /// Each hash is a 64-character lowercase hex string.
    /// </summary>
    public IReadOnlyList<string> LeafHashes() => _leafHashes.AsReadOnly();

    /// <summary>
    /// Check whether <paramref name="hexHash"/> matches any stored leaf hash
    /// (case-insensitive comparison).
    /// </summary>
    public bool ContainsLeafHash(string hexHash)
        => _leafHashes.Any(h => string.Equals(h, hexHash, StringComparison.OrdinalIgnoreCase));

    // -------------------------------------------------------------------------
    // Static helpers
    // -------------------------------------------------------------------------

    /// <summary>
    /// Hash each item in <paramref name="items"/> with SHA-256, build a tree,
    /// and return the Merkle root.
    /// </summary>
    /// <exception cref="ArgumentException">When <paramref name="items"/> is empty.</exception>
    public static string ComputeRootFromBytes(IEnumerable<byte[]> items)
    {
        var hashes = items.Select(Hasher.HashBytes).ToList();
        if (hashes.Count == 0)
            throw new ArgumentException(
                "Cannot compute Merkle root of an empty collection.", nameof(items));

        return ComputeRoot(hashes);
    }

    /// <summary>
    /// Build a <see cref="MerkleTree"/> from pre-computed SHA-256 leaf hashes
    /// (64-char lowercase hex strings) without re-hashing the underlying data.
    /// </summary>
    public static MerkleTree BuildFromLeafHashes(IEnumerable<string> hexHashes)
    {
        var tree = new MerkleTree();
        tree._leafHashes.AddRange(hexHashes);
        return tree;
    }

    // -------------------------------------------------------------------------
    // Core algorithm
    // -------------------------------------------------------------------------

    private static string ComputeRoot(List<string> leafHashes)
    {
        // Step 1 — compute RFC 6962 leaf digests: SHA-256(0x00 || leaf_hash_bytes)
        var layer = leafHashes.Select(LeafDigest).ToList();

        // Step 2 — reduce layers until one root remains
        while (layer.Count > 1)
        {
            var next = new List<string>();
            for (int i = 0; i < layer.Count; i += 2)
            {
                if (i + 1 < layer.Count)
                    next.Add(NodeDigest(layer[i], layer[i + 1]));
                else
                    next.Add(layer[i]);   // odd node — promote without hashing
            }
            layer = next;
        }

        return layer[0];
    }

    /// <summary>SHA-256(0x00 || bytes(leafHex))</summary>
    private static string LeafDigest(string leafHex)
    {
        var leafBytes = Hasher.FromHex(leafHex);
        var payload = new byte[1 + leafBytes.Length];
        payload[0] = 0x00;
        leafBytes.CopyTo(payload, 1);
        return Hasher.ToHex(SHA256.HashData(payload));
    }

    /// <summary>SHA-256(0x01 || bytes(leftHex) || bytes(rightHex))</summary>
    private static string NodeDigest(string leftHex, string rightHex)
    {
        var leftBytes  = Hasher.FromHex(leftHex);
        var rightBytes = Hasher.FromHex(rightHex);
        var payload = new byte[1 + leftBytes.Length + rightBytes.Length];
        payload[0] = 0x01;
        leftBytes.CopyTo(payload, 1);
        rightBytes.CopyTo(payload, 1 + leftBytes.Length);
        return Hasher.ToHex(SHA256.HashData(payload));
    }
}
