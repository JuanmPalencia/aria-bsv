//! aria-bsv-merkle — RFC 6962 Merkle tree for ARIA BRC-121 audit records.
//!
//! # Protocol
//!
//! **Leaf nodes**: `SHA-256(0x00 || data)`
//! **Internal nodes**: `SHA-256(0x01 || left || right)`
//!
//! The `0x00`/`0x01` prefixes provide second-preimage attack resistance,
//! ensuring a leaf cannot be confused with an internal node.
//!
//! This is the ARIA audit tree, not to be confused with the Bitcoin block
//! Merkle tree (which uses SHA-256d without prefixes).
//!
//! # Example
//! ```rust
//! use aria_bsv_merkle::Tree;
//!
//! let mut tree = Tree::new();
//! tree.add_leaf(b"record_0");
//! tree.add_leaf(b"record_1");
//! tree.add_leaf(b"record_2");
//!
//! let root = tree.root().unwrap();
//! assert_eq!(root.len(), 64);  // lowercase hex SHA-256
//! ```

use sha2::{Digest, Sha256};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors returned by the merkle module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MerkleError {
    /// The tree has no leaves; the root is undefined.
    EmptyTree,
    /// A leaf hash string is not valid 64-character lowercase hex.
    InvalidLeafHash(String),
}

impl std::fmt::Display for MerkleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MerkleError::EmptyTree => write!(f, "merkle tree is empty"),
            MerkleError::InvalidLeafHash(s) => write!(f, "invalid leaf hash: {s}"),
        }
    }
}

impl std::error::Error for MerkleError {}

// ---------------------------------------------------------------------------
// Internal hashing primitives
// ---------------------------------------------------------------------------

fn hash_leaf(data: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update([0x00]);
    h.update(data);
    h.finalize().into()
}

fn hash_internal(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update([0x01]);
    h.update(left);
    h.update(right);
    h.finalize().into()
}

fn bytes_to_hex(b: &[u8; 32]) -> String {
    hex::encode(b)
}

fn hex_to_bytes(s: &str) -> Result<[u8; 32], MerkleError> {
    if s.len() != 64 {
        return Err(MerkleError::InvalidLeafHash(s.to_string()));
    }
    let decoded = hex::decode(s).map_err(|_| MerkleError::InvalidLeafHash(s.to_string()))?;
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&decoded);
    Ok(arr)
}

// ---------------------------------------------------------------------------
// Tree
// ---------------------------------------------------------------------------

/// ARIA RFC 6962 Merkle tree.
///
/// Leaves are added incrementally via [`add_leaf`][Tree::add_leaf].
/// The Merkle root is recomputed lazily on each call to [`root`][Tree::root].
pub struct Tree {
    /// Raw leaf data (kept to allow `compute_root_from_bytes`-style APIs).
    leaves_raw: Vec<Vec<u8>>,
    /// SHA-256 hex digests of the leaves (public, for proof generation).
    leaf_hashes: Vec<[u8; 32]>,
}

impl Tree {
    /// Create an empty tree.
    pub fn new() -> Self {
        Tree {
            leaves_raw: Vec::new(),
            leaf_hashes: Vec::new(),
        }
    }

    /// Add a raw byte slice as a new leaf.
    ///
    /// The leaf hash is `SHA-256(0x00 || data)`.
    pub fn add_leaf(&mut self, data: &[u8]) {
        self.leaf_hashes.push(hash_leaf(data));
        self.leaves_raw.push(data.to_vec());
    }

    /// Add a leaf from an existing SHA-256 hex hash string.
    ///
    /// Use this when the leaf data is already hashed (e.g. from an `AuditRecord`'s
    /// `input_hash` field).  The provided string is treated as a leaf value and
    /// wrapped in `hash_leaf` as required by the RFC 6962 protocol.
    pub fn add_leaf_hash(&mut self, hash_hex: &str) -> Result<(), MerkleError> {
        let hash_bytes = hex_to_bytes(hash_hex)?;
        // Treat the 32-byte hash as leaf data, apply hash_leaf on top.
        self.leaf_hashes.push(hash_leaf(&hash_bytes));
        self.leaves_raw.push(hash_bytes.to_vec());
        Ok(())
    }

    /// Number of leaves in the tree.
    pub fn len(&self) -> usize {
        self.leaf_hashes.len()
    }

    /// Returns `true` if the tree has no leaves.
    pub fn is_empty(&self) -> bool {
        self.leaf_hashes.is_empty()
    }

    /// Compute and return the Merkle root as lowercase hex.
    ///
    /// Returns `Err(MerkleError::EmptyTree)` if the tree has no leaves.
    pub fn root(&self) -> Result<String, MerkleError> {
        if self.leaf_hashes.is_empty() {
            return Err(MerkleError::EmptyTree);
        }
        let root = Self::compute_root(&self.leaf_hashes);
        Ok(bytes_to_hex(&root))
    }

    /// Return a copy of all raw leaf data.
    pub fn leaves_raw(&self) -> Vec<Vec<u8>> {
        self.leaves_raw.clone()
    }

    /// Return all leaf hashes as lowercase hex strings.
    pub fn leaf_hashes_hex(&self) -> Vec<String> {
        self.leaf_hashes.iter().map(bytes_to_hex).collect()
    }

    /// Check whether a given SHA-256 hex hash is in the tree as a leaf hash.
    ///
    /// The comparison is against the `hash_leaf(data)` values, not the raw bytes.
    pub fn contains_leaf_hash(&self, hash_hex: &str) -> bool {
        let Ok(target) = hex_to_bytes(hash_hex) else {
            return false;
        };
        self.leaf_hashes.iter().any(|h| h == &target)
    }

    fn compute_root(hashes: &[[u8; 32]]) -> [u8; 32] {
        assert!(!hashes.is_empty());
        if hashes.len() == 1 {
            return hashes[0];
        }

        let mut level: Vec<[u8; 32]> = hashes.to_vec();

        while level.len() > 1 {
            let mut next = Vec::with_capacity((level.len() + 1) / 2);
            let mut i = 0;
            while i < level.len() {
                if i + 1 < level.len() {
                    next.push(hash_internal(&level[i], &level[i + 1]));
                    i += 2;
                } else {
                    // Odd node: promote without pairing (standard practice).
                    next.push(level[i]);
                    i += 1;
                }
            }
            level = next;
        }

        level[0]
    }
}

impl Default for Tree {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Convenience constructors
// ---------------------------------------------------------------------------

/// Build a tree from a slice of raw byte items.
///
/// Each item is hashed as `SHA-256(0x00 || item)` to produce the leaf hash.
///
/// # Errors
/// Returns `MerkleError::EmptyTree` if `items` is empty.
pub fn compute_root_from_bytes(items: &[&[u8]]) -> Result<String, MerkleError> {
    if items.is_empty() {
        return Err(MerkleError::EmptyTree);
    }
    let mut tree = Tree::new();
    for item in items {
        tree.add_leaf(item);
    }
    tree.root()
}

/// Build a tree from a slice of existing SHA-256 hex strings.
///
/// Each hash is wrapped in `hash_leaf` before building the tree.
///
/// # Errors
/// Returns `MerkleError::InvalidLeafHash` if any string is not valid hex of length 64.
/// Returns `MerkleError::EmptyTree` if `leaf_hashes` is empty.
pub fn build_tree(leaf_hashes: &[&str]) -> Result<Tree, MerkleError> {
    if leaf_hashes.is_empty() {
        return Err(MerkleError::EmptyTree);
    }
    let mut tree = Tree::new();
    for h in leaf_hashes {
        tree.add_leaf_hash(h)?;
    }
    Ok(tree)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn hex_leaf(data: &[u8]) -> String {
        bytes_to_hex(&hash_leaf(data))
    }

    fn hex_internal(left: &str, right: &str) -> String {
        let l = hex_to_bytes(left).unwrap();
        let r = hex_to_bytes(right).unwrap();
        bytes_to_hex(&hash_internal(&l, &r))
    }

    // -- Single leaf ----------------------------------------------------------

    #[test]
    fn root_single_leaf_equals_hash_leaf() {
        let mut tree = Tree::new();
        tree.add_leaf(b"record_0");

        let root = tree.root().unwrap();
        let expected = hex_leaf(b"record_0");
        assert_eq!(root, expected);
    }

    // -- Two leaves -----------------------------------------------------------

    #[test]
    fn root_two_leaves() {
        let mut tree = Tree::new();
        tree.add_leaf(b"a");
        tree.add_leaf(b"b");

        let root = tree.root().unwrap();
        let la = hex_leaf(b"a");
        let lb = hex_leaf(b"b");
        let expected = hex_internal(&la, &lb);
        assert_eq!(root, expected);
    }

    // -- Order sensitivity ----------------------------------------------------

    #[test]
    fn root_is_order_sensitive() {
        let mut tree_ab = Tree::new();
        tree_ab.add_leaf(b"a");
        tree_ab.add_leaf(b"b");

        let mut tree_ba = Tree::new();
        tree_ba.add_leaf(b"b");
        tree_ba.add_leaf(b"a");

        assert_ne!(tree_ab.root().unwrap(), tree_ba.root().unwrap());
    }

    // -- Determinism ----------------------------------------------------------

    #[test]
    fn root_is_deterministic() {
        let build = || {
            let mut t = Tree::new();
            t.add_leaf(b"x");
            t.add_leaf(b"y");
            t.add_leaf(b"z");
            t.root().unwrap()
        };
        assert_eq!(build(), build());
    }

    // -- Three leaves (odd promotion) -----------------------------------------

    #[test]
    fn root_three_leaves_promotes_odd() {
        // Level 0: [hash_leaf(a), hash_leaf(b), hash_leaf(c)]
        // Level 1: [hash_internal(hl_a, hl_b), hash_leaf(c)]  ← c promoted
        // Level 2: [hash_internal(level1[0], level1[1])]
        let la = hex_leaf(b"a");
        let lb = hex_leaf(b"b");
        let lc = hex_leaf(b"c");
        let l1_0 = hex_internal(&la, &lb);
        let expected = hex_internal(&l1_0, &lc);

        let mut tree = Tree::new();
        tree.add_leaf(b"a");
        tree.add_leaf(b"b");
        tree.add_leaf(b"c");

        assert_eq!(tree.root().unwrap(), expected);
    }

    // -- Four leaves (perfect binary tree) ------------------------------------

    #[test]
    fn root_four_leaves_perfect_tree() {
        let la = hex_leaf(b"a");
        let lb = hex_leaf(b"b");
        let lc = hex_leaf(b"c");
        let ld = hex_leaf(b"d");
        let l1_01 = hex_internal(&la, &lb);
        let l1_23 = hex_internal(&lc, &ld);
        let expected = hex_internal(&l1_01, &l1_23);

        let mut tree = Tree::new();
        for item in [b"a" as &[u8], b"b", b"c", b"d"] {
            tree.add_leaf(item);
        }

        assert_eq!(tree.root().unwrap(), expected);
    }

    // -- Empty tree -----------------------------------------------------------

    #[test]
    fn root_empty_tree_returns_error() {
        let tree = Tree::new();
        assert!(matches!(tree.root(), Err(MerkleError::EmptyTree)));
    }

    #[test]
    fn is_empty_new_tree() {
        assert!(Tree::new().is_empty());
    }

    #[test]
    fn len_after_adds() {
        let mut tree = Tree::new();
        assert_eq!(tree.len(), 0);
        tree.add_leaf(b"a");
        assert_eq!(tree.len(), 1);
        tree.add_leaf(b"b");
        assert_eq!(tree.len(), 2);
    }

    // -- contains_leaf_hash ---------------------------------------------------

    #[test]
    fn contains_leaf_hash_present() {
        let mut tree = Tree::new();
        tree.add_leaf(b"data");
        let h = hex_leaf(b"data");
        assert!(tree.contains_leaf_hash(&h));
    }

    #[test]
    fn contains_leaf_hash_absent() {
        let mut tree = Tree::new();
        tree.add_leaf(b"a");
        let h = hex_leaf(b"b");
        assert!(!tree.contains_leaf_hash(&h));
    }

    #[test]
    fn contains_leaf_hash_invalid_hex_returns_false() {
        let tree = Tree::new();
        assert!(!tree.contains_leaf_hash("not-hex"));
    }

    // -- compute_root_from_bytes ----------------------------------------------

    #[test]
    fn compute_root_from_bytes_single() {
        let root = compute_root_from_bytes(&[b"item"]).unwrap();
        assert_eq!(root, hex_leaf(b"item"));
    }

    #[test]
    fn compute_root_from_bytes_empty_errors() {
        assert!(matches!(
            compute_root_from_bytes(&[]),
            Err(MerkleError::EmptyTree)
        ));
    }

    #[test]
    fn compute_root_from_bytes_two_items() {
        let root = compute_root_from_bytes(&[b"a", b"b"]).unwrap();
        let la = hex_leaf(b"a");
        let lb = hex_leaf(b"b");
        assert_eq!(root, hex_internal(&la, &lb));
    }

    // -- build_tree -----------------------------------------------------------

    #[test]
    fn build_tree_from_valid_hashes() {
        let h1 = hex_leaf(b"rec1");
        let h2 = hex_leaf(b"rec2");
        let tree = build_tree(&[&h1, &h2]).unwrap();
        assert_eq!(tree.len(), 2);
        assert!(tree.contains_leaf_hash(&h1));
    }

    #[test]
    fn build_tree_empty_errors() {
        assert!(matches!(build_tree(&[]), Err(MerkleError::EmptyTree)));
    }

    #[test]
    fn build_tree_invalid_hash_errors() {
        assert!(matches!(
            build_tree(&["not-64-chars"]),
            Err(MerkleError::InvalidLeafHash(_))
        ));
    }

    // -- Root format ----------------------------------------------------------

    #[test]
    fn root_is_64_char_lowercase_hex() {
        let mut tree = Tree::new();
        tree.add_leaf(b"test");
        let root = tree.root().unwrap();
        assert_eq!(root.len(), 64);
        assert_eq!(root, root.to_lowercase());
        assert!(root.chars().all(|c| c.is_ascii_hexdigit()));
    }

    // -- Second-preimage protection -------------------------------------------

    #[test]
    fn leaf_and_internal_nodes_differ_same_bytes() {
        // If an attacker presents an internal node as a leaf, the hash must differ.
        // hash_leaf(x) != hash_internal(a, b) even if x == a||b
        let data = [b"a" as &[u8], b"b"];
        let leaf_root = compute_root_from_bytes(&data).unwrap();

        // Single leaf containing the raw concatenation bytes "ab"
        let mut single_tree = Tree::new();
        single_tree.add_leaf(b"ab");
        let single_root = single_tree.root().unwrap();

        assert_ne!(leaf_root, single_root);
    }
}
