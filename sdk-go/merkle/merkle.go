// Package merkle implements the ARIA audit Merkle tree (RFC 6962).
//
// Leaf nodes:    SHA-256(0x00 || data)
// Internal nodes: SHA-256(0x01 || left || right)
//
// This is the ARIA audit tree — different from the Bitcoin block Merkle tree
// (which uses SHA-256d without prefixes).
package merkle

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
)

const (
	prefixLeaf     = byte(0x00)
	prefixInternal = byte(0x01)
)

// ErrEmptyTree is returned when operations require at least one leaf.
var ErrEmptyTree = errors.New("merkle tree is empty")

// hashLeaf computes SHA-256(0x00 || data).
func hashLeaf(data []byte) []byte {
	h := sha256.New()
	h.Write([]byte{prefixLeaf})
	h.Write(data)
	return h.Sum(nil)
}

// hashInternal computes SHA-256(0x01 || left || right).
func hashInternal(left, right []byte) []byte {
	h := sha256.New()
	h.Write([]byte{prefixInternal})
	h.Write(left)
	h.Write(right)
	return h.Sum(nil)
}

// ComputeRoot computes the Merkle root of a list of pre-hashed leaf values.
// Each leafHash must be a 64-character lowercase hex string (32 bytes).
// Returns the root as a 64-character lowercase hex string.
func ComputeRoot(leafHashes []string) (string, error) {
	if len(leafHashes) == 0 {
		return "", ErrEmptyTree
	}

	layer := make([][]byte, len(leafHashes))
	for i, h := range leafHashes {
		b, err := hex.DecodeString(h)
		if err != nil {
			return "", err
		}
		layer[i] = hashLeaf(b)
	}

	for len(layer) > 1 {
		next := make([][]byte, 0, (len(layer)+1)/2)
		for i := 0; i < len(layer); i += 2 {
			if i+1 < len(layer) {
				next = append(next, hashInternal(layer[i], layer[i+1]))
			} else {
				// Odd node: promote without hashing
				next = append(next, layer[i])
			}
		}
		layer = next
	}

	return hex.EncodeToString(layer[0]), nil
}

// MerkleProof is an inclusion proof for a single leaf.
type MerkleProof struct {
	// LeafHash is the 64-hex-char hash of the leaf data.
	LeafHash string
	// Index is the 0-based position of the leaf in the tree.
	Index int
	// Siblings are the sibling hashes from leaf to root (hex strings).
	Siblings []string
}

// Tree holds an append-only list of leaf hashes.
type Tree struct {
	leaves []string
}

// Add appends a pre-hashed leaf (64-char hex) to the tree.
func (t *Tree) Add(leafHash string) {
	t.leaves = append(t.leaves, leafHash)
}

// Root computes and returns the current Merkle root.
func (t *Tree) Root() (string, error) {
	return ComputeRoot(t.leaves)
}

// Len returns the number of leaves.
func (t *Tree) Len() int {
	return len(t.leaves)
}

// GenerateProof builds an inclusion proof for the leaf at index.
func (t *Tree) GenerateProof(index int) (*MerkleProof, error) {
	if index < 0 || index >= len(t.leaves) {
		return nil, errors.New("index out of range")
	}

	layer := make([][]byte, len(t.leaves))
	for i, h := range t.leaves {
		b, err := hex.DecodeString(h)
		if err != nil {
			return nil, err
		}
		layer[i] = hashLeaf(b)
	}

	idx := index
	var siblings []string

	for len(layer) > 1 {
		next := make([][]byte, 0, (len(layer)+1)/2)
		for i := 0; i < len(layer); i += 2 {
			if i+1 < len(layer) {
				if i == idx || i+1 == idx {
					sibIdx := idx ^ 1
					if sibIdx < len(layer) {
						siblings = append(siblings, hex.EncodeToString(layer[sibIdx]))
					} else {
						siblings = append(siblings, hex.EncodeToString(layer[i]))
					}
				}
				next = append(next, hashInternal(layer[i], layer[i+1]))
			} else {
				if i == idx {
					// odd leaf promotes — no sibling at this level
					siblings = append(siblings, hex.EncodeToString(layer[i]))
				}
				next = append(next, layer[i])
			}
		}
		layer = next
		idx /= 2
	}

	return &MerkleProof{
		LeafHash: t.leaves[index],
		Index:    index,
		Siblings: siblings,
	}, nil
}

// VerifyProof checks that proof.LeafHash is included in expectedRoot.
func VerifyProof(proof *MerkleProof, expectedRoot string) (bool, error) {
	leafBytes, err := hex.DecodeString(proof.LeafHash)
	if err != nil {
		return false, err
	}
	current := hashLeaf(leafBytes)
	idx := proof.Index

	for _, sibHex := range proof.Siblings {
		sib, err := hex.DecodeString(sibHex)
		if err != nil {
			return false, err
		}
		if idx%2 == 0 {
			current = hashInternal(current, sib)
		} else {
			current = hashInternal(sib, current)
		}
		idx /= 2
	}

	computed := hex.EncodeToString(current)
	return computed == expectedRoot, nil
}
