/**
 * Tests for aria-bsv/merkle — MerkleTree, computeMerkleRoot, proof generation.
 */

import { describe, it, expect } from "vitest";
import {
  EMPTY_ROOT,
  MerkleTree,
  computeMerkleRoot,
  computeMerkleRootSync,
  generateMerkleProof,
  verifyMerkleProof,
} from "./merkle.js";
import { sha256HexSync } from "./hasher.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function fakeLeaf(s: string): string {
  return sha256HexSync(s);
}

// ---------------------------------------------------------------------------
// EMPTY_ROOT
// ---------------------------------------------------------------------------

describe("EMPTY_ROOT", () => {
  it("is 64 zeros", () => {
    expect(EMPTY_ROOT).toBe("0".repeat(64));
  });
});

// ---------------------------------------------------------------------------
// computeMerkleRootSync
// ---------------------------------------------------------------------------

describe("computeMerkleRootSync", () => {
  it("returns EMPTY_ROOT for empty array", () => {
    expect(computeMerkleRootSync([])).toBe(EMPTY_ROOT);
  });

  it("single leaf: root = leaf digest (not the raw leaf)", () => {
    const leaf = fakeLeaf("a");
    const root = computeMerkleRootSync([leaf]);
    expect(root).not.toBe(leaf);  // root is SHA-256(0x00 || leaf)
    expect(root).toHaveLength(64);
  });

  it("same leaves → same root (deterministic)", () => {
    const leaves = ["a", "b", "c"].map(fakeLeaf);
    expect(computeMerkleRootSync(leaves)).toBe(computeMerkleRootSync(leaves));
  });

  it("order matters — different order → different root", () => {
    const l1 = fakeLeaf("x");
    const l2 = fakeLeaf("y");
    expect(computeMerkleRootSync([l1, l2])).not.toBe(
      computeMerkleRootSync([l2, l1])
    );
  });

  it("two leaves — root is not equal to either leaf", () => {
    const l1 = fakeLeaf("a");
    const l2 = fakeLeaf("b");
    const root = computeMerkleRootSync([l1, l2]);
    expect(root).not.toBe(l1);
    expect(root).not.toBe(l2);
    expect(root).toHaveLength(64);
  });

  it("odd number of leaves — last node promoted, not duplicated", () => {
    const leaves = ["a", "b", "c"].map(fakeLeaf);
    const root3 = computeMerkleRootSync(leaves);
    // Adding a 4th leaf that is a duplicate of the 3rd should change the root
    const root4 = computeMerkleRootSync([...leaves, leaves[2]]);
    expect(root3).not.toBe(root4);
  });

  it("four leaves produces consistent root", () => {
    const leaves = ["a", "b", "c", "d"].map(fakeLeaf);
    expect(computeMerkleRootSync(leaves)).toHaveLength(64);
  });

  it("second-preimage protection: leaf root ≠ raw leaf hash", () => {
    // If prefix were absent, root of [h] would equal h for single-leaf trees
    const leaf = fakeLeaf("data");
    const root = computeMerkleRootSync([leaf]);
    expect(root).not.toBe(leaf);
  });
});

// ---------------------------------------------------------------------------
// computeMerkleRoot (async) vs sync
// ---------------------------------------------------------------------------

describe("computeMerkleRoot (async)", () => {
  it("returns EMPTY_ROOT for empty array", async () => {
    expect(await computeMerkleRoot([])).toBe(EMPTY_ROOT);
  });

  it("matches sync result for single leaf", async () => {
    const leaves = [fakeLeaf("hello")];
    expect(await computeMerkleRoot(leaves)).toBe(
      computeMerkleRootSync(leaves)
    );
  });

  it("matches sync result for four leaves", async () => {
    const leaves = ["a", "b", "c", "d"].map(fakeLeaf);
    expect(await computeMerkleRoot(leaves)).toBe(
      computeMerkleRootSync(leaves)
    );
  });

  it("matches sync result for odd leaf count", async () => {
    const leaves = ["a", "b", "c", "d", "e"].map(fakeLeaf);
    expect(await computeMerkleRoot(leaves)).toBe(
      computeMerkleRootSync(leaves)
    );
  });
});

// ---------------------------------------------------------------------------
// MerkleTree class
// ---------------------------------------------------------------------------

describe("MerkleTree", () => {
  it("size starts at 0", () => {
    expect(new MerkleTree().size).toBe(0);
  });

  it("size increments on addLeaf", () => {
    const tree = new MerkleTree();
    tree.addLeaf(fakeLeaf("a"));
    tree.addLeaf(fakeLeaf("b"));
    expect(tree.size).toBe(2);
  });

  it("root() matches computeMerkleRoot", async () => {
    const leaves = ["a", "b", "c"].map(fakeLeaf);
    const tree = new MerkleTree();
    leaves.forEach((l) => tree.addLeaf(l));
    expect(await tree.root()).toBe(await computeMerkleRoot(leaves));
  });

  it("rootSync() matches computeMerkleRootSync", () => {
    const leaves = ["a", "b"].map(fakeLeaf);
    const tree = new MerkleTree();
    leaves.forEach((l) => tree.addLeaf(l));
    expect(tree.rootSync()).toBe(computeMerkleRootSync(leaves));
  });

  it("async and sync roots agree", async () => {
    const tree = new MerkleTree();
    ["x", "y", "z"].map(fakeLeaf).forEach((l) => tree.addLeaf(l));
    expect(await tree.root()).toBe(tree.rootSync());
  });

  it("empty tree root is EMPTY_ROOT", async () => {
    expect(await new MerkleTree().root()).toBe(EMPTY_ROOT);
  });
});

// ---------------------------------------------------------------------------
// Merkle proofs
// ---------------------------------------------------------------------------

describe("generateMerkleProof / verifyMerkleProof", () => {
  it("throws on empty tree", async () => {
    await expect(generateMerkleProof([], 0)).rejects.toThrow();
  });

  it("throws on out-of-range index", async () => {
    const leaves = ["a", "b"].map(fakeLeaf);
    await expect(generateMerkleProof(leaves, 5)).rejects.toThrow(RangeError);
  });

  it("single leaf: proof verifies against root", async () => {
    const leaf = fakeLeaf("solo");
    const root = await computeMerkleRoot([leaf]);
    const proof = await generateMerkleProof([leaf], 0);
    expect(await verifyMerkleProof(leaf, proof, root)).toBe(true);
  });

  it("two leaves: both proofs verify", async () => {
    const leaves = ["left", "right"].map(fakeLeaf);
    const root = await computeMerkleRoot(leaves);
    for (let i = 0; i < 2; i++) {
      const proof = await generateMerkleProof(leaves, i);
      expect(await verifyMerkleProof(leaves[i], proof, root)).toBe(true);
    }
  });

  it("four leaves: all proofs verify", async () => {
    const leaves = ["a", "b", "c", "d"].map(fakeLeaf);
    const root = await computeMerkleRoot(leaves);
    for (let i = 0; i < 4; i++) {
      const proof = await generateMerkleProof(leaves, i);
      expect(await verifyMerkleProof(leaves[i], proof, root)).toBe(true);
    }
  });

  it("five leaves (odd): all proofs verify", async () => {
    const leaves = ["a", "b", "c", "d", "e"].map(fakeLeaf);
    const root = await computeMerkleRoot(leaves);
    for (let i = 0; i < 5; i++) {
      const proof = await generateMerkleProof(leaves, i);
      expect(await verifyMerkleProof(leaves[i], proof, root)).toBe(true);
    }
  });

  it("tampered leaf fails verification", async () => {
    const leaves = ["a", "b", "c"].map(fakeLeaf);
    const root = await computeMerkleRoot(leaves);
    const proof = await generateMerkleProof(leaves, 1);
    const tamperedLeaf = fakeLeaf("tampered");
    expect(await verifyMerkleProof(tamperedLeaf, proof, root)).toBe(false);
  });

  it("wrong root fails verification", async () => {
    const leaves = ["a", "b"].map(fakeLeaf);
    const proof = await generateMerkleProof(leaves, 0);
    const wrongRoot = "0".repeat(64);
    expect(await verifyMerkleProof(leaves[0], proof, wrongRoot)).toBe(false);
  });
});
