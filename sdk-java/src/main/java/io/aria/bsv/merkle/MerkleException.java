package io.aria.bsv.merkle;

/**
 * Thrown by {@link MerkleTree} when an operation cannot be completed, for
 * example when computing a root on an empty tree.
 */
public class MerkleException extends Exception {

    public MerkleException(String message) {
        super(message);
    }
}
