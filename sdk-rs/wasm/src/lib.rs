//! aria-bsv-wasm — WebAssembly bindings for the ARIA BRC-121 SDK.
//!
//! Provides browser and Node.js compatible bindings for the hasher and
//! merkle crates via wasm-bindgen. Compiled with `wasm-pack build`.
//!
//! # Usage (browser/Node.js)
//! ```js
//! import init, { hashObject, hashBytes, MerkleTree } from './aria_bsv_wasm.js';
//! await init();
//!
//! const h = hashObject(JSON.stringify({"b": 2, "a": 1}));
//! console.log(h); // same as Python/TS/Go/Rust SDKs
//!
//! const tree = new MerkleTree();
//! tree.addLeaf(new TextEncoder().encode("record_0"));
//! console.log(tree.root()); // RFC 6962 merkle root
//! ```

use aria_bsv_hasher as hasher;
use aria_bsv_merkle as merkle;
use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Panic hook for better error messages in browser console
// ---------------------------------------------------------------------------

#[wasm_bindgen(start)]
pub fn init_panic_hook() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

// ---------------------------------------------------------------------------
// Hasher API
// ---------------------------------------------------------------------------

/// Hash raw bytes with SHA-256. Returns lowercase hex string.
///
/// @param {Uint8Array} data - The raw bytes to hash.
/// @returns {string} 64-character lowercase hex SHA-256 digest.
#[wasm_bindgen(js_name = hashBytes)]
pub fn hash_bytes(data: &[u8]) -> String {
    hasher::hash_bytes(data)
}

/// Hash a UTF-8 string with SHA-256. Returns lowercase hex string.
///
/// @param {string} s - The string to hash.
/// @returns {string} 64-character lowercase hex SHA-256 digest.
#[wasm_bindgen(js_name = hashString)]
pub fn hash_string(s: &str) -> String {
    hasher::hash_string(s)
}

/// Canonicalize a JSON string and return its SHA-256 hash.
///
/// The JSON is re-parsed and re-serialized with sorted keys before hashing,
/// matching the ARIA BRC-121 canonical JSON format.
///
/// @param {string} jsonStr - A JSON string to canonicalize and hash.
/// @returns {string} 64-character lowercase hex SHA-256 digest.
/// @throws {Error} If the input is not valid JSON or contains non-finite numbers.
#[wasm_bindgen(js_name = hashObject)]
pub fn hash_object(json_str: &str) -> Result<String, JsValue> {
    let v: serde_json::Value = serde_json::from_str(json_str)
        .map_err(|e| JsValue::from_str(&format!("invalid JSON: {e}")))?;
    hasher::hash_object(&v)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Canonicalize a JSON string and return "sha256:<hex>".
///
/// @param {string} jsonStr - A JSON string to canonicalize and hash.
/// @returns {string} "sha256:" followed by 64-character lowercase hex.
/// @throws {Error} If the input is not valid JSON.
#[wasm_bindgen(js_name = hashObjectPrefixed)]
pub fn hash_object_prefixed(json_str: &str) -> Result<String, JsValue> {
    let v: serde_json::Value = serde_json::from_str(json_str)
        .map_err(|e| JsValue::from_str(&format!("invalid JSON: {e}")))?;
    hasher::hash_object_prefixed(&v)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Prepend "sha256:" to a hash string.
///
/// @param {string} hash - A hash string (typically 64-char hex).
/// @returns {string} "sha256:<hash>"
#[wasm_bindgen(js_name = prefixedHash)]
pub fn prefixed_hash(hash: &str) -> String {
    hasher::prefixed_hash(hash)
}

/// Compare two SHA-256 hex strings case-insensitively.
///
/// @param {string} a - First hash string.
/// @param {string} b - Second hash string.
/// @returns {boolean} True if both strings represent the same hash.
#[wasm_bindgen(js_name = hashEqual)]
pub fn equal(a: &str, b: &str) -> bool {
    hasher::equal(a, b)
}

/// Serialize an object to canonical JSON bytes (sorted keys, no whitespace).
///
/// @param {string} jsonStr - A JSON string to canonicalize.
/// @returns {Uint8Array} Canonical JSON bytes.
/// @throws {Error} If not valid JSON.
#[wasm_bindgen(js_name = canonicalJson)]
pub fn canonical_json(json_str: &str) -> Result<Vec<u8>, JsValue> {
    let v: serde_json::Value = serde_json::from_str(json_str)
        .map_err(|e| JsValue::from_str(&format!("invalid JSON: {e}")))?;
    hasher::canonical_json(&v)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

// ---------------------------------------------------------------------------
// MerkleTree API
// ---------------------------------------------------------------------------

/// RFC 6962 Merkle tree for ARIA audit records.
///
/// Leaf nodes:     SHA-256(0x00 || data)
/// Internal nodes: SHA-256(0x01 || left || right)
///
/// @example
/// ```js
/// const tree = new MerkleTree();
/// tree.addLeaf(new TextEncoder().encode("rec_001"));
/// tree.addLeaf(new TextEncoder().encode("rec_002"));
/// console.log(tree.root()); // 64-char hex Merkle root
/// ```
#[wasm_bindgen]
pub struct MerkleTree {
    inner: merkle::Tree,
}

#[wasm_bindgen]
impl MerkleTree {
    /// Create an empty Merkle tree.
    #[wasm_bindgen(constructor)]
    pub fn new() -> MerkleTree {
        MerkleTree {
            inner: merkle::Tree::new(),
        }
    }

    /// Add a raw byte array as a leaf.
    ///
    /// @param {Uint8Array} data - Raw bytes for this leaf.
    #[wasm_bindgen(js_name = addLeaf)]
    pub fn add_leaf(&mut self, data: &[u8]) {
        self.inner.add_leaf(data);
    }

    /// Add a leaf from an existing SHA-256 hex hash string.
    ///
    /// @param {string} hexHash - 64-character lowercase hex hash.
    /// @throws {Error} If hexHash is not valid 64-char hex.
    #[wasm_bindgen(js_name = addLeafHash)]
    pub fn add_leaf_hash(&mut self, hex_hash: &str) -> Result<(), JsValue> {
        self.inner
            .add_leaf_hash(hex_hash)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Compute and return the Merkle root as lowercase hex.
    ///
    /// @returns {string} 64-character lowercase hex Merkle root.
    /// @throws {Error} If the tree is empty.
    pub fn root(&self) -> Result<String, JsValue> {
        self.inner
            .root()
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Number of leaves in the tree.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns true if the tree has no leaves.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Check whether a leaf hash (hex string) is in the tree.
    ///
    /// @param {string} hexHash - 64-character hex to look up.
    /// @returns {boolean}
    #[wasm_bindgen(js_name = containsLeafHash)]
    pub fn contains_leaf_hash(&self, hex_hash: &str) -> bool {
        self.inner.contains_leaf_hash(hex_hash)
    }

    /// Return all leaf hashes as a JSON array of hex strings.
    ///
    /// @returns {string} JSON array of 64-char hex strings.
    #[wasm_bindgen(js_name = leafHashes)]
    pub fn leaf_hashes(&self) -> String {
        serde_json::to_string(&self.inner.leaf_hashes_hex())
            .unwrap_or_else(|_| "[]".to_string())
    }
}

/// Compute a Merkle root from a list of raw byte arrays (JSON-encoded).
///
/// @param {string} itemsJson - JSON array of base64-encoded byte arrays.
/// @returns {string} 64-character hex Merkle root.
/// @throws {Error} If input is invalid or empty.
#[wasm_bindgen(js_name = computeRootFromBytesB64)]
pub fn compute_root_from_bytes_b64(items_b64_json: &str) -> Result<String, JsValue> {
    let b64_items: Vec<String> = serde_json::from_str(items_b64_json)
        .map_err(|e| JsValue::from_str(&format!("invalid JSON: {e}")))?;

    if b64_items.is_empty() {
        return Err(JsValue::from_str("merkle tree is empty"));
    }

    let mut tree = merkle::Tree::new();
    for b64 in &b64_items {
        let bytes = base64_decode(b64)
            .map_err(|e| JsValue::from_str(&format!("base64 decode error: {e}")))?;
        tree.add_leaf(&bytes);
    }

    tree.root()
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

// ---------------------------------------------------------------------------
// Base64 helper (stdlib only — no external dep)
// ---------------------------------------------------------------------------

fn base64_decode(s: &str) -> Result<Vec<u8>, &'static str> {
    // Standard base64 alphabet
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut decode_map = [255u8; 256];
    for (i, &c) in ALPHABET.iter().enumerate() {
        decode_map[c as usize] = i as u8;
    }

    let s = s.trim_end_matches('=');
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let bytes = s.as_bytes();
    let mut i = 0;

    while i + 3 < bytes.len() {
        let a = decode_map[bytes[i] as usize];
        let b = decode_map[bytes[i + 1] as usize];
        let c = decode_map[bytes[i + 2] as usize];
        let d = decode_map[bytes[i + 3] as usize];
        if a == 255 || b == 255 || c == 255 || d == 255 {
            return Err("invalid base64 character");
        }
        out.push((a << 2) | (b >> 4));
        out.push((b << 4) | (c >> 2));
        out.push((c << 6) | d);
        i += 4;
    }

    match bytes.len() - i {
        2 => {
            let a = decode_map[bytes[i] as usize];
            let b = decode_map[bytes[i + 1] as usize];
            if a == 255 || b == 255 { return Err("invalid base64"); }
            out.push((a << 2) | (b >> 4));
        }
        3 => {
            let a = decode_map[bytes[i] as usize];
            let b = decode_map[bytes[i + 1] as usize];
            let c = decode_map[bytes[i + 2] as usize];
            if a == 255 || b == 255 || c == 255 { return Err("invalid base64"); }
            out.push((a << 2) | (b >> 4));
            out.push((b << 4) | (c >> 2));
        }
        _ => {}
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// WASM tests (run with: wasm-pack test --node)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_bytes_empty() {
        let h = hash_bytes(&[]);
        assert_eq!(h, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    }

    #[test]
    fn test_hash_bytes_abc() {
        let h = hash_bytes(b"abc");
        assert_eq!(h, "ba7816bf8f01cfea414140de5dae2ec73b00361bbef0469328ce1b14f7a1d7b8");
    }

    #[test]
    fn test_hash_object_sorted_keys() {
        let result = hash_object(r#"{"b":2,"a":1}"#).unwrap();
        let expected = hash_bytes(b"{\"a\":1,\"b\":2}");
        assert_eq!(result, expected);
    }

    #[test]
    fn test_hash_object_invalid_json_errors() {
        assert!(hash_object("not json").is_err());
    }

    #[test]
    fn test_prefixed_hash() {
        let h = "a".repeat(64);
        assert_eq!(prefixed_hash(&h), format!("sha256:{h}"));
    }

    #[test]
    fn test_equal_case_insensitive() {
        assert!(equal("AABBCC", "aabbcc"));
        assert!(!equal("aab", "aac"));
    }

    #[test]
    fn test_merkle_tree_empty_root_errors() {
        let tree = MerkleTree::new();
        assert!(tree.root().is_err());
        assert!(tree.is_empty());
    }

    #[test]
    fn test_merkle_tree_single_leaf() {
        let mut tree = MerkleTree::new();
        tree.add_leaf(b"record_0");
        let root = tree.root().unwrap();
        assert_eq!(root.len(), 64);
        assert_eq!(tree.len(), 1);
    }

    #[test]
    fn test_merkle_tree_determinism() {
        let mut t1 = MerkleTree::new();
        t1.add_leaf(b"a");
        t1.add_leaf(b"b");

        let mut t2 = MerkleTree::new();
        t2.add_leaf(b"a");
        t2.add_leaf(b"b");

        assert_eq!(t1.root().unwrap(), t2.root().unwrap());
    }

    #[test]
    fn test_canonical_json_round_trip() {
        let bytes = canonical_json(r#"{"z":3,"a":1,"m":2}"#).unwrap();
        let s = String::from_utf8(bytes).unwrap();
        assert_eq!(s, r#"{"a":1,"m":2,"z":3}"#);
    }
}
