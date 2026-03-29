package io.aria.bsv.merkle;

import io.aria.bsv.hasher.Hasher;
import org.junit.jupiter.api.Test;

import java.security.MessageDigest;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link MerkleTree}.
 *
 * <p>All tests verify the RFC 6962 construction:
 * <ul>
 *   <li>Leaf:     {@code SHA-256(0x00 || data)}</li>
 *   <li>Internal: {@code SHA-256(0x01 || left || right)}</li>
 *   <li>Odd promotion: unmatched nodes are carried up unchanged</li>
 * </ul>
 */
class MerkleTreeTest {

    // =========================================================================
    // Test helpers — mirror the internal primitives
    // =========================================================================

    private static byte[] hashLeafBytes(byte[] data) throws Exception {
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        md.update((byte) 0x00);
        md.update(data);
        return md.digest();
    }

    private static byte[] hashInternalBytes(byte[] left, byte[] right) throws Exception {
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        md.update((byte) 0x01);
        md.update(left);
        md.update(right);
        return md.digest();
    }

    private static String leafHex(byte[] data) throws Exception {
        return Hasher.bytesToHex(hashLeafBytes(data));
    }

    private static String internalHex(byte[] left, byte[] right) throws Exception {
        return Hasher.bytesToHex(hashInternalBytes(left, right));
    }

    // =========================================================================
    // Single leaf
    // =========================================================================

    @Test
    void singleLeaf_rootEqualsHashLeaf() throws Exception {
        MerkleTree tree = new MerkleTree();
        tree.addLeaf("record_0".getBytes());

        String root     = tree.root();
        String expected = leafHex("record_0".getBytes());
        assertEquals(expected, root,
                "Single-leaf root must equal hash_leaf(data)");
    }

    // =========================================================================
    // Two leaves
    // =========================================================================

    @Test
    void twoLeaves_rootIsHashInternal() throws Exception {
        MerkleTree tree = new MerkleTree();
        tree.addLeaf("a".getBytes());
        tree.addLeaf("b".getBytes());

        byte[] la = hashLeafBytes("a".getBytes());
        byte[] lb = hashLeafBytes("b".getBytes());
        String expected = internalHex(la, lb);

        assertEquals(expected, tree.root());
    }

    // =========================================================================
    // Three leaves — odd-node promotion
    // =========================================================================

    @Test
    void threeLeaves_oddNodePromoted() throws Exception {
        // Level 0: [hash_leaf(a), hash_leaf(b), hash_leaf(c)]
        // Level 1: [hash_internal(hl_a, hl_b), hash_leaf(c)]  ← c promoted
        // Level 2: [hash_internal(level1[0], level1[1])]
        byte[] la = hashLeafBytes("a".getBytes());
        byte[] lb = hashLeafBytes("b".getBytes());
        byte[] lc = hashLeafBytes("c".getBytes());
        byte[] l1 = hashInternalBytes(la, lb);
        String expected = internalHex(l1, lc);

        MerkleTree tree = new MerkleTree();
        tree.addLeaf("a".getBytes());
        tree.addLeaf("b".getBytes());
        tree.addLeaf("c".getBytes());

        assertEquals(expected, tree.root());
    }

    // =========================================================================
    // Order sensitivity
    // =========================================================================

    @Test
    void orderSensitive_differentOrderDifferentRoot() throws Exception {
        MerkleTree treeAB = new MerkleTree();
        treeAB.addLeaf("a".getBytes());
        treeAB.addLeaf("b".getBytes());

        MerkleTree treeBA = new MerkleTree();
        treeBA.addLeaf("b".getBytes());
        treeBA.addLeaf("a".getBytes());

        assertNotEquals(treeAB.root(), treeBA.root(),
                "Leaf order must affect the Merkle root");
    }

    // =========================================================================
    // Determinism
    // =========================================================================

    @Test
    void deterministicRootForSameData() throws Exception {
        MerkleTree t1 = new MerkleTree();
        t1.addLeaf("x".getBytes());
        t1.addLeaf("y".getBytes());
        t1.addLeaf("z".getBytes());

        MerkleTree t2 = new MerkleTree();
        t2.addLeaf("x".getBytes());
        t2.addLeaf("y".getBytes());
        t2.addLeaf("z".getBytes());

        assertEquals(t1.root(), t2.root(),
                "Same data must always produce the same root");
    }

    // =========================================================================
    // Empty tree
    // =========================================================================

    @Test
    void emptyTree_rootThrowsMerkleException() {
        MerkleTree tree = new MerkleTree();
        assertThrows(MerkleException.class, tree::root,
                "root() on empty tree must throw MerkleException");
    }

    @Test
    void emptyTree_sizeIsZero() {
        assertEquals(0, new MerkleTree().size());
    }

    // =========================================================================
    // containsLeafHash
    // =========================================================================

    @Test
    void containsLeafHash_presentAfterAdd() throws Exception {
        MerkleTree tree = new MerkleTree();
        tree.addLeaf("data".getBytes());

        String leafHashHex = leafHex("data".getBytes());
        assertTrue(tree.containsLeafHash(leafHashHex),
                "containsLeafHash must find a hash that was added");
    }

    @Test
    void containsLeafHash_absentForOtherData() throws Exception {
        MerkleTree tree = new MerkleTree();
        tree.addLeaf("a".getBytes());

        String otherHex = leafHex("b".getBytes());
        assertFalse(tree.containsLeafHash(otherHex));
    }

    @Test
    void containsLeafHash_invalidHexReturnsFalse() {
        MerkleTree tree = new MerkleTree();
        assertFalse(tree.containsLeafHash("not-valid-hex"));
        assertFalse(tree.containsLeafHash(null));
        assertFalse(tree.containsLeafHash("tooshort"));
    }

    // =========================================================================
    // size and leafHashes
    // =========================================================================

    @Test
    void size_incrementsWithEachAddLeaf() {
        MerkleTree tree = new MerkleTree();
        assertEquals(0, tree.size());
        tree.addLeaf("a".getBytes());
        assertEquals(1, tree.size());
        tree.addLeaf("b".getBytes());
        assertEquals(2, tree.size());
    }

    @Test
    void leafHashes_returnsHashLeafValuesInOrder() throws Exception {
        MerkleTree tree = new MerkleTree();
        tree.addLeaf("alpha".getBytes());
        tree.addLeaf("beta".getBytes());

        List<String> hashes = tree.leafHashes();
        assertEquals(2, hashes.size());
        assertEquals(leafHex("alpha".getBytes()), hashes.get(0));
        assertEquals(leafHex("beta".getBytes()),  hashes.get(1));
    }

    // =========================================================================
    // Static constructors
    // =========================================================================

    @Test
    void buildFromBytes_equivalentToManualAddLeaf() throws Exception {
        List<byte[]> items = Arrays.asList("x".getBytes(), "y".getBytes());
        MerkleTree built = MerkleTree.buildFromBytes(items);

        MerkleTree manual = new MerkleTree();
        manual.addLeaf("x".getBytes());
        manual.addLeaf("y".getBytes());

        assertEquals(manual.root(), built.root());
    }

    @Test
    void computeRootFromBytes_singleItem() throws Exception {
        String root = MerkleTree.computeRootFromBytes(
                Collections.singletonList("item".getBytes()));
        assertEquals(leafHex("item".getBytes()), root);
    }

    @Test
    void computeRootFromBytes_emptyListThrows() {
        assertThrows(MerkleException.class,
                () -> MerkleTree.computeRootFromBytes(Collections.emptyList()),
                "computeRootFromBytes with empty list must throw MerkleException");
    }

    // =========================================================================
    // Second-preimage attack resistance
    // =========================================================================

    @Test
    void secondPreimage_leafAbNotEqualTwoLeavesAB() throws Exception {
        // A single leaf "ab" must produce a different root than two leaves "a", "b".
        // This ensures the 0x00/0x01 prefix scheme prevents second-preimage attacks.
        MerkleTree twoLeaves = new MerkleTree();
        twoLeaves.addLeaf("a".getBytes());
        twoLeaves.addLeaf("b".getBytes());

        MerkleTree singleLeaf = new MerkleTree();
        singleLeaf.addLeaf("ab".getBytes());

        assertNotEquals(twoLeaves.root(), singleLeaf.root(),
                "Two leaves [a,b] must not equal single leaf [ab] — RFC 6962 second-preimage protection");
    }

    @Test
    void rootIs64LowercaseHexChars() throws Exception {
        MerkleTree tree = new MerkleTree();
        tree.addLeaf("test".getBytes());
        String root = tree.root();
        assertEquals(64, root.length());
        assertEquals(root, root.toLowerCase(java.util.Locale.ROOT));
        assertTrue(root.matches("[0-9a-f]+"));
    }
}
