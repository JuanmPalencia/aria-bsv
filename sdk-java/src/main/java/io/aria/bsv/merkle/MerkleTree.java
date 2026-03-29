package io.aria.bsv.merkle;

import io.aria.bsv.hasher.Hasher;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * RFC 6962 Merkle tree for ARIA BRC-121 audit records.
 *
 * <p>Hashing protocol:
 * <ul>
 *   <li><b>Leaf nodes:</b>     {@code SHA-256(0x00 || data)}</li>
 *   <li><b>Internal nodes:</b> {@code SHA-256(0x01 || left || right)}</li>
 * </ul>
 *
 * <p>The {@code 0x00}/{@code 0x01} prefixes provide second-preimage attack
 * resistance: a leaf can never be confused with an internal node even if their
 * raw byte payloads coincide.
 *
 * <p>This is the ARIA audit tree. It is distinct from the Bitcoin block Merkle
 * tree (which uses double-SHA-256 without prefixes).
 */
public final class MerkleTree {

    private static final byte PREFIX_LEAF     = 0x00;
    private static final byte PREFIX_INTERNAL = 0x01;

    /**
     * Stored leaf hashes. Each element is the 32-byte result of
     * {@code SHA-256(0x00 || rawLeafData)}.
     */
    private final List<byte[]> leafHashes = new ArrayList<>();

    // -------------------------------------------------------------------------
    // Mutation
    // -------------------------------------------------------------------------

    /**
     * Add a leaf from raw bytes.
     *
     * <p>The leaf hash is computed as {@code SHA-256(0x00 || data)} and stored
     * internally. Call {@link #root()} to obtain the current Merkle root.
     *
     * @param data raw leaf data (any length)
     */
    public void addLeaf(byte[] data) {
        leafHashes.add(hashLeaf(data));
    }

    // -------------------------------------------------------------------------
    // Queries
    // -------------------------------------------------------------------------

    /**
     * Compute and return the Merkle root as a 64-character lowercase hex string.
     *
     * @return Merkle root hex
     * @throws MerkleException if the tree has no leaves
     */
    public String root() throws MerkleException {
        if (leafHashes.isEmpty()) {
            throw new MerkleException("merkle tree is empty");
        }
        return Hasher.bytesToHex(computeRoot(leafHashes));
    }

    /**
     * Return all leaf hashes — i.e. {@code SHA-256(0x00 || rawData)} for each
     * leaf added — as lowercase hex strings, in insertion order.
     *
     * @return unmodifiable snapshot of leaf hashes
     */
    public List<String> leafHashes() {
        List<String> result = new ArrayList<>(leafHashes.size());
        for (byte[] h : leafHashes) {
            result.add(Hasher.bytesToHex(h));
        }
        return result;
    }

    /**
     * Return {@code true} if the given 64-character hex string matches any
     * stored leaf hash (i.e. a {@code SHA-256(0x00 || data)} value).
     *
     * <p>Comparison is case-insensitive.
     *
     * @param hexHash 64-character hex leaf hash
     * @return {@code true} if present
     */
    public boolean containsLeafHash(String hexHash) {
        if (hexHash == null || hexHash.length() != 64) {
            return false;
        }
        byte[] target;
        try {
            target = Hasher.hexToBytes(hexHash.toLowerCase(java.util.Locale.ROOT));
        } catch (Exception e) {
            return false;
        }
        for (byte[] h : leafHashes) {
            if (Arrays.equals(h, target)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Return the number of leaves currently in the tree.
     *
     * @return leaf count
     */
    public int size() {
        return leafHashes.size();
    }

    // -------------------------------------------------------------------------
    // Static constructors
    // -------------------------------------------------------------------------

    /**
     * Build a {@code MerkleTree} from a list of raw byte slices.
     *
     * <p>Each item is added via {@link #addLeaf(byte[])} — producing a leaf
     * hash of {@code SHA-256(0x00 || item)}.
     *
     * @param items list of raw data items (must not be null)
     * @return populated tree
     */
    public static MerkleTree buildFromBytes(List<byte[]> items) {
        MerkleTree tree = new MerkleTree();
        for (byte[] item : items) {
            tree.addLeaf(item);
        }
        return tree;
    }

    /**
     * Compute the Merkle root of a list of raw byte slices without retaining
     * the tree structure.
     *
     * @param items list of raw data items
     * @return lowercase hex Merkle root
     * @throws MerkleException if {@code items} is empty
     */
    public static String computeRootFromBytes(List<byte[]> items) throws MerkleException {
        if (items.isEmpty()) {
            throw new MerkleException("merkle tree is empty");
        }
        return buildFromBytes(items).root();
    }

    // -------------------------------------------------------------------------
    // Internal hashing primitives
    // -------------------------------------------------------------------------

    private static byte[] hashLeaf(byte[] data) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            md.update(PREFIX_LEAF);
            md.update(data);
            return md.digest();
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 not available", e);
        }
    }

    private static byte[] hashInternal(byte[] left, byte[] right) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            md.update(PREFIX_INTERNAL);
            md.update(left);
            md.update(right);
            return md.digest();
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 not available", e);
        }
    }

    /**
     * Compute the root over a list of already-hashed leaf values.
     *
     * <p>Odd nodes are promoted to the next level without pairing (standard
     * RFC 6962 / certificate-transparency practice).
     */
    private static byte[] computeRoot(List<byte[]> hashes) {
        List<byte[]> layer = new ArrayList<>(hashes);
        while (layer.size() > 1) {
            List<byte[]> next = new ArrayList<>((layer.size() + 1) / 2);
            for (int i = 0; i < layer.size(); i += 2) {
                if (i + 1 < layer.size()) {
                    next.add(hashInternal(layer.get(i), layer.get(i + 1)));
                } else {
                    // Odd leaf — promote without pairing
                    next.add(layer.get(i));
                }
            }
            layer = next;
        }
        return layer.get(0);
    }
}
