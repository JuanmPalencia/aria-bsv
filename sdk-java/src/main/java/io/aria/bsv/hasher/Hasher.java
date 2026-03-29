package io.aria.bsv.hasher;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

/**
 * SHA-256 hashing utilities for ARIA BRC-121.
 *
 * <p>All public methods produce lowercase hex-encoded SHA-256 digests (64 characters).
 * Object hashing always goes through {@link CanonicalJson} to ensure
 * deterministic, cross-SDK-compatible output.
 */
public final class Hasher {

    private Hasher() {}

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    /**
     * Compute SHA-256 of raw bytes.
     *
     * @param data input bytes
     * @return lowercase hex SHA-256 digest (64 chars)
     */
    public static String hashBytes(byte[] data) {
        return bytesToHex(sha256(data));
    }

    /**
     * Compute SHA-256 of a UTF-8 encoded string.
     *
     * @param s input string
     * @return lowercase hex SHA-256 digest (64 chars)
     */
    public static String hashString(String s) {
        return hashBytes(s.getBytes(StandardCharsets.UTF_8));
    }

    /**
     * Compute SHA-256 of the canonical JSON encoding of {@code v}.
     *
     * <p>The value is serialized via {@link CanonicalJson#toBytes(Object)} (keys
     * sorted, no whitespace) before hashing.
     *
     * @param v any JSON-serializable object
     * @return lowercase hex SHA-256 digest (64 chars)
     * @throws CanonicalJson.CanonicalJsonException if {@code v} cannot be serialized
     */
    public static String hashObject(Object v) {
        return hashBytes(CanonicalJson.toBytes(v));
    }

    /**
     * Return the hash in {@code "sha256:<hex>"} format, matching the Python SDK
     * convention used in DatasetAnchor and EpochOpen records.
     *
     * @param hash a 64-character lowercase hex SHA-256 hash
     * @return {@code "sha256:" + hash}
     */
    public static String prefixedHash(String hash) {
        return "sha256:" + hash;
    }

    /**
     * Compute the canonical hash of {@code v} and return it in
     * {@code "sha256:<hex>"} format.
     *
     * @param v any JSON-serializable object
     * @return {@code "sha256:" + hashObject(v)}
     */
    public static String hashObjectPrefixed(Object v) {
        return prefixedHash(hashObject(v));
    }

    /**
     * Report whether two hex-encoded SHA-256 hashes are equal.
     *
     * <p>Comparison is case-insensitive and constant-time to mitigate timing
     * side-channel attacks.
     *
     * @param a first hash string (may be upper- or lower-case)
     * @param b second hash string (may be upper- or lower-case)
     * @return {@code true} if the hashes represent the same digest
     */
    public static boolean equal(String a, String b) {
        if (a == null || b == null) {
            return a == b;
        }
        String lowerA = a.toLowerCase(java.util.Locale.ROOT);
        String lowerB = b.toLowerCase(java.util.Locale.ROOT);
        if (lowerA.length() != lowerB.length()) {
            return false;
        }
        // Constant-time XOR comparison to avoid early exit
        int result = 0;
        for (int i = 0; i < lowerA.length(); i++) {
            result |= lowerA.charAt(i) ^ lowerB.charAt(i);
        }
        return result == 0;
    }

    // -------------------------------------------------------------------------
    // Package-internal helpers (used by MerkleTree)
    // -------------------------------------------------------------------------

    static byte[] sha256(byte[] data) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            return md.digest(data);
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 algorithm not available", e);
        }
    }

    static String bytesToHex(byte[] bytes) {
        StringBuilder sb = new StringBuilder(bytes.length * 2);
        for (byte b : bytes) {
            sb.append(String.format("%02x", b & 0xff));
        }
        return sb.toString();
    }

    static byte[] hexToBytes(String hex) {
        int len = hex.length();
        if (len % 2 != 0) {
            throw new IllegalArgumentException("Hex string must have even length: " + hex);
        }
        byte[] data = new byte[len / 2];
        for (int i = 0; i < len; i += 2) {
            int hi = Character.digit(hex.charAt(i), 16);
            int lo = Character.digit(hex.charAt(i + 1), 16);
            if (hi < 0 || lo < 0) {
                throw new IllegalArgumentException("Invalid hex character at index " + i);
            }
            data[i / 2] = (byte) ((hi << 4) + lo);
        }
        return data;
    }
}
