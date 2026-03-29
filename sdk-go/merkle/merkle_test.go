package merkle_test

import (
	"testing"

	"github.com/JuanmPalencia/aria-bsv/sdk-go/merkle"
)

const zeroHash = "0000000000000000000000000000000000000000000000000000000000000000"
const aaHash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
const bbHash = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
const ccHash = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"

func TestComputeRoot_SingleLeaf(t *testing.T) {
	root, err := merkle.ComputeRoot([]string{zeroHash})
	if err != nil {
		t.Fatal(err)
	}
	if len(root) != 64 {
		t.Errorf("expected 64-char root, got %d", len(root))
	}
}

func TestComputeRoot_EmptyReturnsError(t *testing.T) {
	_, err := merkle.ComputeRoot([]string{})
	if err == nil {
		t.Error("expected error for empty tree")
	}
}

func TestComputeRoot_TwoLeaves(t *testing.T) {
	root, err := merkle.ComputeRoot([]string{aaHash, bbHash})
	if err != nil {
		t.Fatal(err)
	}
	if len(root) != 64 {
		t.Errorf("got root length %d", len(root))
	}
}

func TestComputeRoot_Deterministic(t *testing.T) {
	leaves := []string{aaHash, bbHash, ccHash}
	r1, _ := merkle.ComputeRoot(leaves)
	r2, _ := merkle.ComputeRoot(leaves)
	if r1 != r2 {
		t.Error("not deterministic")
	}
}

func TestComputeRoot_OrderSensitive(t *testing.T) {
	r1, _ := merkle.ComputeRoot([]string{aaHash, bbHash})
	r2, _ := merkle.ComputeRoot([]string{bbHash, aaHash})
	if r1 == r2 {
		t.Error("order should matter")
	}
}

func TestComputeRoot_DifferentLeavesDifferentRoot(t *testing.T) {
	r1, _ := merkle.ComputeRoot([]string{aaHash})
	r2, _ := merkle.ComputeRoot([]string{bbHash})
	if r1 == r2 {
		t.Error("different leaves should give different roots")
	}
}

func TestTree_AddAndRoot(t *testing.T) {
	var tree merkle.Tree
	tree.Add(aaHash)
	tree.Add(bbHash)
	root, err := tree.Root()
	if err != nil {
		t.Fatal(err)
	}
	expected, _ := merkle.ComputeRoot([]string{aaHash, bbHash})
	if root != expected {
		t.Errorf("tree.Root() != ComputeRoot: %s vs %s", root, expected)
	}
}

func TestTree_LenTracked(t *testing.T) {
	var tree merkle.Tree
	if tree.Len() != 0 {
		t.Error("new tree should have 0 leaves")
	}
	tree.Add(aaHash)
	tree.Add(bbHash)
	if tree.Len() != 2 {
		t.Errorf("expected 2, got %d", tree.Len())
	}
}

func TestTree_RootEmptyError(t *testing.T) {
	var tree merkle.Tree
	_, err := tree.Root()
	if err == nil {
		t.Error("expected error for empty tree")
	}
}

func TestGenerateAndVerifyProof_SingleLeaf(t *testing.T) {
	var tree merkle.Tree
	tree.Add(aaHash)
	root, _ := tree.Root()
	proof, err := tree.GenerateProof(0)
	if err != nil {
		t.Fatal(err)
	}
	ok, err := merkle.VerifyProof(proof, root)
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Error("proof should be valid")
	}
}

func TestGenerateAndVerifyProof_TwoLeaves_Index0(t *testing.T) {
	var tree merkle.Tree
	tree.Add(aaHash)
	tree.Add(bbHash)
	root, _ := tree.Root()
	proof, _ := tree.GenerateProof(0)
	ok, _ := merkle.VerifyProof(proof, root)
	if !ok {
		t.Error("proof for index 0 should be valid")
	}
}

func TestGenerateAndVerifyProof_TwoLeaves_Index1(t *testing.T) {
	var tree merkle.Tree
	tree.Add(aaHash)
	tree.Add(bbHash)
	root, _ := tree.Root()
	proof, _ := tree.GenerateProof(1)
	ok, _ := merkle.VerifyProof(proof, root)
	if !ok {
		t.Error("proof for index 1 should be valid")
	}
}

func TestGenerateAndVerifyProof_ThreeLeaves_AllIndices(t *testing.T) {
	leaves := []string{aaHash, bbHash, ccHash}
	var tree merkle.Tree
	for _, l := range leaves {
		tree.Add(l)
	}
	root, _ := tree.Root()
	for i := range leaves {
		proof, err := tree.GenerateProof(i)
		if err != nil {
			t.Fatalf("index %d: %v", i, err)
		}
		ok, err := merkle.VerifyProof(proof, root)
		if err != nil {
			t.Fatalf("index %d verify error: %v", i, err)
		}
		if !ok {
			t.Errorf("proof for index %d should be valid", i)
		}
	}
}

func TestVerifyProof_WrongRoot_Fails(t *testing.T) {
	var tree merkle.Tree
	tree.Add(aaHash)
	tree.Add(bbHash)
	proof, _ := tree.GenerateProof(0)
	ok, _ := merkle.VerifyProof(proof, zeroHash)
	if ok {
		t.Error("wrong root should fail")
	}
}

func TestVerifyProof_TamperedLeaf_Fails(t *testing.T) {
	var tree merkle.Tree
	tree.Add(aaHash)
	tree.Add(bbHash)
	root, _ := tree.Root()
	proof, _ := tree.GenerateProof(0)
	proof.LeafHash = ccHash // tamper
	ok, _ := merkle.VerifyProof(proof, root)
	if ok {
		t.Error("tampered leaf should fail")
	}
}

func TestSecondPreimage_LeafCannotMasqueradeAsInternal(t *testing.T) {
	// Build a 2-leaf tree
	var tree merkle.Tree
	tree.Add(aaHash)
	tree.Add(bbHash)
	root, _ := tree.Root()

	// Build a single-leaf tree whose leaf = root of the 2-leaf tree.
	// Without prefix protection this would collide; with RFC 6962 it must not.
	var singleTree merkle.Tree
	singleTree.Add(root)
	singleRoot, _ := singleTree.Root()

	if singleRoot == root {
		t.Error("second-preimage protection failed")
	}
}

func TestBuildTree_MatchesManualAdd(t *testing.T) {
	var manual merkle.Tree
	manual.Add(aaHash)
	manual.Add(bbHash)
	manual.Add(ccHash)
	built := merkle.BuildTree([]string{aaHash, bbHash, ccHash})

	rootManual, _ := manual.Root()
	rootBuilt, _ := built.Root()
	if rootManual != rootBuilt {
		t.Errorf("BuildTree root mismatch: %s vs %s", rootManual, rootBuilt)
	}
}

func TestBuildTree_Empty(t *testing.T) {
	tree := merkle.BuildTree([]string{})
	if tree.Len() != 0 {
		t.Errorf("expected 0 leaves, got %d", tree.Len())
	}
}

func TestTree_Leaves_ReturnsCopy(t *testing.T) {
	var tree merkle.Tree
	tree.Add(aaHash)
	tree.Add(bbHash)
	leaves := tree.Leaves()
	if len(leaves) != 2 {
		t.Errorf("expected 2 leaves, got %d", len(leaves))
	}
	// Mutate the returned slice — should not affect tree
	leaves[0] = ccHash
	if tree.Leaves()[0] != aaHash {
		t.Error("Leaves() should return a copy, not a reference")
	}
}

func TestTree_Contains(t *testing.T) {
	var tree merkle.Tree
	tree.Add(aaHash)
	tree.Add(bbHash)
	if !tree.Contains(aaHash) {
		t.Error("aaHash should be contained")
	}
	if tree.Contains(ccHash) {
		t.Error("ccHash should not be contained")
	}
}

func TestComputeRootFromBytes_SingleItem(t *testing.T) {
	root, err := merkle.ComputeRootFromBytes([][]byte{[]byte("hello")})
	if err != nil {
		t.Fatal(err)
	}
	if len(root) != 64 {
		t.Errorf("expected 64-char root, got %d", len(root))
	}
}

func TestComputeRootFromBytes_ReturnsErrorForEmpty(t *testing.T) {
	_, err := merkle.ComputeRootFromBytes([][]byte{})
	if err == nil {
		t.Error("expected error for empty input")
	}
}

func TestComputeRootFromBytes_DifferentDataDifferentRoot(t *testing.T) {
	r1, _ := merkle.ComputeRootFromBytes([][]byte{[]byte("a")})
	r2, _ := merkle.ComputeRootFromBytes([][]byte{[]byte("b")})
	if r1 == r2 {
		t.Error("different data should produce different roots")
	}
}

func TestComputeRootFromBytes_OrderSensitive(t *testing.T) {
	r1, _ := merkle.ComputeRootFromBytes([][]byte{[]byte("a"), []byte("b")})
	r2, _ := merkle.ComputeRootFromBytes([][]byte{[]byte("b"), []byte("a")})
	if r1 == r2 {
		t.Error("order should matter")
	}
}
