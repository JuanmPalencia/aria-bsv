// MerkleTreeTests.cs — xUnit tests for AriaBsv.Core.MerkleTree.
//
// Tests cover:
//   - Empty tree guard (Root throws)
//   - Single-leaf root equals RFC 6962 leaf digest
//   - Two-leaf and four-leaf roots (determinism + structure)
//   - Odd-leaf promotion
//   - Second-preimage protection (leaf prefix 0x00 vs node prefix 0x01)
//   - LeafHashes / ContainsLeafHash / Count accessors
//   - ComputeRootFromBytes static helper
//   - BuildFromLeafHashes round-trip

using System.Text;
using AriaBsv.Core;

namespace AriaBsv.Tests;

public class MerkleTreeTests
{
    private static byte[] Bytes(string s) => Encoding.UTF8.GetBytes(s);

    // -------------------------------------------------------------------------
    // Empty tree
    // -------------------------------------------------------------------------

    [Fact]
    public void Root_EmptyTree_ThrowsInvalidOperationException()
    {
        var tree = new MerkleTree();
        Assert.Throws<InvalidOperationException>(() => tree.Root());
    }

    [Fact]
    public void Count_NewTree_IsZero()
    {
        Assert.Equal(0, new MerkleTree().Count);
    }

    // -------------------------------------------------------------------------
    // Single leaf
    // -------------------------------------------------------------------------

    [Fact]
    public void Root_SingleLeaf_Is64LowercaseHexChars()
    {
        var tree = new MerkleTree();
        tree.AddLeaf(Bytes("aria"));
        var root = tree.Root();
        Assert.Equal(64, root.Length);
        Assert.Matches("^[0-9a-f]{64}$", root);
    }

    [Fact]
    public void Root_SingleLeaf_EqualsRfc6962LeafDigest()
    {
        // RFC 6962: root = SHA-256(0x00 || SHA-256(data))
        var data     = Bytes("test");
        var leafHash = Hasher.HashBytes(data);
        var expected = ComputeExpectedLeafDigest(leafHash);

        var tree = new MerkleTree();
        tree.AddLeaf(data);
        Assert.Equal(expected, tree.Root());
    }

    [Fact]
    public void Root_SameInput_Deterministic()
    {
        var data = Bytes("deterministic");
        var t1   = new MerkleTree();
        var t2   = new MerkleTree();
        t1.AddLeaf(data);
        t2.AddLeaf(data);
        Assert.Equal(t1.Root(), t2.Root());
    }

    // -------------------------------------------------------------------------
    // Two leaves
    // -------------------------------------------------------------------------

    [Fact]
    public void Root_TwoLeaves_DiffersFromSingleLeaf()
    {
        var one = new MerkleTree();
        one.AddLeaf(Bytes("a"));

        var two = new MerkleTree();
        two.AddLeaf(Bytes("a"));
        two.AddLeaf(Bytes("b"));

        Assert.NotEqual(one.Root(), two.Root());
    }

    [Fact]
    public void Root_TwoLeaves_DifferentOrder_DifferentHash()
    {
        var ab = new MerkleTree();
        ab.AddLeaf(Bytes("a"));
        ab.AddLeaf(Bytes("b"));

        var ba = new MerkleTree();
        ba.AddLeaf(Bytes("b"));
        ba.AddLeaf(Bytes("a"));

        Assert.NotEqual(ab.Root(), ba.Root());
    }

    [Fact]
    public void Root_TwoLeaves_MatchesManualRfc6962Computation()
    {
        var data1    = Bytes("test");
        var data2    = Bytes("aria");
        var leafHash1 = Hasher.HashBytes(data1);
        var leafHash2 = Hasher.HashBytes(data2);
        var ld1       = ComputeExpectedLeafDigest(leafHash1);
        var ld2       = ComputeExpectedLeafDigest(leafHash2);
        var expected  = ComputeExpectedNodeDigest(ld1, ld2);

        var tree = new MerkleTree();
        tree.AddLeaf(data1);
        tree.AddLeaf(data2);
        Assert.Equal(expected, tree.Root());
    }

    // -------------------------------------------------------------------------
    // Odd-leaf promotion
    // -------------------------------------------------------------------------

    [Fact]
    public void Root_ThreeLeaves_LastLeafPromoted()
    {
        // With 3 leaves [L0, L1, L2]:
        //   layer1 = [LD0, LD1, LD2]
        //   layer2 = [Node(LD0,LD1), LD2 promoted]
        //   root   = Node(Node(LD0,LD1), LD2)
        var t3 = new MerkleTree();
        t3.AddLeaf(Bytes("a"));
        t3.AddLeaf(Bytes("b"));
        t3.AddLeaf(Bytes("c"));
        var root3 = t3.Root();

        var t4 = new MerkleTree();
        t4.AddLeaf(Bytes("a"));
        t4.AddLeaf(Bytes("b"));
        t4.AddLeaf(Bytes("c"));
        t4.AddLeaf(Bytes("d"));
        var root4 = t4.Root();

        Assert.NotEqual(root3, root4);
        Assert.Matches("^[0-9a-f]{64}$", root3);
    }

    // -------------------------------------------------------------------------
    // Second-preimage protection
    // -------------------------------------------------------------------------

    [Fact]
    public void LeafDigest_DiffersFromNodeDigestOfSameData()
    {
        // A leaf digest (0x00 prefix) must differ from a node digest (0x01 prefix)
        // of the same payload — this is the RFC 6962 second-preimage guard.
        var data        = Bytes("payload");
        var leafHash    = Hasher.HashBytes(data);
        var leafDigest  = ComputeExpectedLeafDigest(leafHash);
        var nodeDigest  = ComputeExpectedNodeDigest(leafHash, leafHash);
        Assert.NotEqual(leafDigest, nodeDigest);
    }

    // -------------------------------------------------------------------------
    // LeafHashes / ContainsLeafHash / Count
    // -------------------------------------------------------------------------

    [Fact]
    public void LeafHashes_ReturnsHashesOfAddedData()
    {
        var data = Bytes("leaf-data");
        var tree = new MerkleTree();
        tree.AddLeaf(data);

        var expected = Hasher.HashBytes(data);
        Assert.Single(tree.LeafHashes());
        Assert.Equal(expected, tree.LeafHashes()[0]);
    }

    [Fact]
    public void ContainsLeafHash_MatchesAddedLeaf()
    {
        var data = Bytes("present");
        var tree = new MerkleTree();
        tree.AddLeaf(data);

        var hash = Hasher.HashBytes(data);
        Assert.True(tree.ContainsLeafHash(hash));
        Assert.True(tree.ContainsLeafHash(hash.ToUpperInvariant())); // case-insensitive
    }

    [Fact]
    public void ContainsLeafHash_ReturnsFalseForAbsentHash()
    {
        var tree = new MerkleTree();
        tree.AddLeaf(Bytes("alpha"));
        Assert.False(tree.ContainsLeafHash(Hasher.HashBytes(Bytes("beta"))));
    }

    [Fact]
    public void Count_IncrementsWithEachAddLeaf()
    {
        var tree = new MerkleTree();
        Assert.Equal(0, tree.Count);
        tree.AddLeaf(Bytes("x"));
        Assert.Equal(1, tree.Count);
        tree.AddLeaf(Bytes("y"));
        Assert.Equal(2, tree.Count);
    }

    // -------------------------------------------------------------------------
    // ComputeRootFromBytes
    // -------------------------------------------------------------------------

    [Fact]
    public void ComputeRootFromBytes_Empty_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(
            () => MerkleTree.ComputeRootFromBytes(Array.Empty<byte[]>()));
    }

    [Fact]
    public void ComputeRootFromBytes_TwoItems_MatchesAddLeafResult()
    {
        var items = new[] { Bytes("x"), Bytes("y") };

        var tree = new MerkleTree();
        foreach (var item in items)
            tree.AddLeaf(item);

        Assert.Equal(tree.Root(), MerkleTree.ComputeRootFromBytes(items));
    }

    // -------------------------------------------------------------------------
    // BuildFromLeafHashes
    // -------------------------------------------------------------------------

    [Fact]
    public void BuildFromLeafHashes_RoundTrip_SameRootAsAddLeaf()
    {
        var dataItems = new[] { Bytes("p"), Bytes("q"), Bytes("r") };

        var tree = new MerkleTree();
        foreach (var d in dataItems)
            tree.AddLeaf(d);

        var precomputed = dataItems.Select(Hasher.HashBytes);
        var rebuilt     = MerkleTree.BuildFromLeafHashes(precomputed);

        Assert.Equal(tree.Root(), rebuilt.Root());
    }

    // -------------------------------------------------------------------------
    // Private helpers — manual RFC 6962 computation for assertions
    // -------------------------------------------------------------------------

    private static string ComputeExpectedLeafDigest(string leafHex)
    {
        var leafBytes = Hasher.FromHex(leafHex);
        var payload   = new byte[1 + leafBytes.Length];
        payload[0]    = 0x00;
        leafBytes.CopyTo(payload, 1);
        return Hasher.HashBytes(payload);
    }

    private static string ComputeExpectedNodeDigest(string leftHex, string rightHex)
    {
        var leftBytes  = Hasher.FromHex(leftHex);
        var rightBytes = Hasher.FromHex(rightHex);
        var payload    = new byte[1 + leftBytes.Length + rightBytes.Length];
        payload[0]     = 0x01;
        leftBytes.CopyTo(payload, 1);
        rightBytes.CopyTo(payload, 1 + leftBytes.Length);
        return Hasher.HashBytes(payload);
    }
}
