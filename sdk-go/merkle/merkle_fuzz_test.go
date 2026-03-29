// Package merkle — fuzz tests for the ARIA RFC 6962 Merkle tree.
//
// Run with: go test -fuzz=Fuzz -fuzztime=30s ./merkle/
// Seed corpus entries are run as normal unit tests: go test ./merkle/
package merkle

import (
	"encoding/hex"
	"strings"
	"testing"
)

// FuzzAddLeaf verifies that adding arbitrary leaf data always produces a 64-char
// hex root and that the tree is non-empty.
func FuzzAddLeaf(f *testing.F) {
	// Seed corpus
	f.Add([]byte("leaf1"))
	f.Add([]byte(""))
	f.Add([]byte{0x00})
	f.Add([]byte{0xff, 0xfe})
	f.Add([]byte("ARIA BRC-121 inference record"))
	f.Add([]byte(`{"record_id":"rec_abc","confidence":0.92}`))

	f.Fuzz(func(t *testing.T, data []byte) {
		tree := NewTree()
		tree.AddLeaf(data)

		root := tree.Root()
		if root == "" {
			t.Fatal("Root() returned empty string after AddLeaf")
		}
		if len(root) != 64 {
			t.Fatalf("Root() returned %d-char string, want 64: %q", len(root), root)
		}
		if _, err := hex.DecodeString(root); err != nil {
			t.Fatalf("Root() is not valid hex: %q", root)
		}
		if root != strings.ToLower(root) {
			t.Fatalf("Root() is not lowercase: %q", root)
		}
	})
}

// FuzzAddLeaf_Determinism verifies that two trees with the same leaves always
// produce the same root.
func FuzzAddLeaf_Determinism(f *testing.F) {
	f.Add([]byte("a"), []byte("b"))
	f.Add([]byte(""), []byte(""))
	f.Add([]byte{0x01}, []byte{0x02})
	f.Add([]byte("record1"), []byte("record2"))

	f.Fuzz(func(t *testing.T, leaf1, leaf2 []byte) {
		tree1 := NewTree()
		tree1.AddLeaf(leaf1)
		tree1.AddLeaf(leaf2)

		tree2 := NewTree()
		tree2.AddLeaf(leaf1)
		tree2.AddLeaf(leaf2)

		root1 := tree1.Root()
		root2 := tree2.Root()

		if root1 != root2 {
			t.Fatalf("Determinism violated: same leaves produced different roots: %q vs %q", root1, root2)
		}
	})
}

// FuzzAddLeaf_OrderSensitivity verifies that swapping leaves changes the root
// (except in the degenerate case where leaf1 == leaf2).
func FuzzAddLeaf_OrderSensitivity(f *testing.F) {
	f.Add([]byte("a"), []byte("b"))
	f.Add([]byte("record1"), []byte("record2"))
	f.Add([]byte{0x00}, []byte{0x01})

	f.Fuzz(func(t *testing.T, leaf1, leaf2 []byte) {
		// Only check when leaves are different.
		if string(leaf1) == string(leaf2) {
			return
		}

		tree1 := NewTree()
		tree1.AddLeaf(leaf1)
		tree1.AddLeaf(leaf2)

		tree2 := NewTree()
		tree2.AddLeaf(leaf2)
		tree2.AddLeaf(leaf1)

		root1 := tree1.Root()
		root2 := tree2.Root()

		if root1 == root2 {
			t.Fatalf("Order sensitivity violated: swapping leaves produced same root %q for leaves (%q, %q)", root1, leaf1, leaf2)
		}
	})
}

// FuzzComputeRootFromBytes verifies structural invariants when building a tree
// from arbitrary byte slices.
func FuzzComputeRootFromBytes(f *testing.F) {
	f.Add([]byte("single item"))
	f.Add([]byte(""))
	f.Add([]byte{0x00, 0xff})

	f.Fuzz(func(t *testing.T, item []byte) {
		root, err := ComputeRootFromBytes([][]byte{item})
		if err != nil {
			t.Fatalf("ComputeRootFromBytes([item]) returned error: %v", err)
		}
		if len(root) != 64 {
			t.Fatalf("ComputeRootFromBytes returned %d-char root, want 64", len(root))
		}

		// Deterministic
		root2, _ := ComputeRootFromBytes([][]byte{item})
		if root != root2 {
			t.Fatalf("ComputeRootFromBytes not deterministic")
		}
	})
}

// FuzzBuildTree verifies that BuildTree always produces a valid root for
// arbitrary leaf hash slices.
func FuzzBuildTree(f *testing.F) {
	validHash1 := strings.Repeat("a", 64)
	validHash2 := strings.Repeat("b", 64)

	f.Add(validHash1)
	f.Add(validHash2)
	f.Add(strings.Repeat("0", 64))

	f.Fuzz(func(t *testing.T, h string) {
		// Only exercise BuildTree with valid-looking hex strings.
		if len(h) != 64 {
			return
		}
		if _, err := hex.DecodeString(h); err != nil {
			return
		}

		tree := BuildTree([]string{h})
		if tree == nil {
			t.Fatalf("BuildTree returned nil for valid leaf hash")
		}

		root := tree.Root()
		if len(root) != 64 {
			t.Fatalf("BuildTree.Root() returned %d-char string, want 64", len(root))
		}

		// Tree must contain the leaf.
		if !tree.Contains(h) {
			t.Fatalf("BuildTree.Contains(%q) returned false", h)
		}
	})
}
