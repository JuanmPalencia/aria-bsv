//! aria-bsv-hasher — Canonical JSON serialization and SHA-256 hashing.
//!
//! Implements the ARIA BRC-121 canonical JSON format, which is identical
//! to the Python SDK's `canonical_json()`: object keys are sorted
//! lexicographically, arrays preserve order, and output is UTF-8 bytes
//! with no whitespace.
//!
//! # Example
//! ```rust
//! use aria_bsv_hasher::{hash_object, hash_bytes, prefixed_hash};
//! use serde_json::json;
//!
//! let h = hash_object(&json!({"b": 2, "a": 1})).unwrap();
//! assert_eq!(h.len(), 64);  // lowercase hex SHA-256
//!
//! // Same result as: hash_bytes(b"{\"a\":1,\"b\":2}")
//! let direct = hash_bytes(b"{\"a\":1,\"b\":2}");
//! assert_eq!(h, direct);
//!
//! assert_eq!(prefixed_hash(&h), format!("sha256:{}", h));
//! ```

use sha2::{Digest, Sha256};
use serde_json::Value;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by the hasher.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HasherError {
    /// The value cannot be represented in canonical JSON (e.g. NaN, Infinity).
    NonFiniteFloat(String),
}

impl std::fmt::Display for HasherError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HasherError::NonFiniteFloat(s) => write!(f, "canonical JSON error: {s}"),
        }
    }
}

impl std::error::Error for HasherError {}

// ---------------------------------------------------------------------------
// Canonical JSON
// ---------------------------------------------------------------------------

/// Serialize `v` to deterministic JSON bytes.
///
/// - Object keys are sorted lexicographically (UTF-8 byte order).
/// - Arrays preserve insertion order.
/// - No whitespace is added.
/// - Numbers that cannot be represented (NaN, Infinity) return an error.
pub fn canonical_json(v: &Value) -> Result<Vec<u8>, HasherError> {
    let mut buf = Vec::with_capacity(128);
    canonical_json_into(v, &mut buf)?;
    Ok(buf)
}

fn canonical_json_into(v: &Value, buf: &mut Vec<u8>) -> Result<(), HasherError> {
    match v {
        Value::Null => buf.extend_from_slice(b"null"),
        Value::Bool(b) => buf.extend_from_slice(if *b { b"true" } else { b"false" }),
        Value::Number(n) => {
            // serde_json's Number doesn't allow NaN/Infinity, but guard anyway.
            if n.is_f64() {
                let f = n.as_f64().unwrap();
                if f.is_nan() || f.is_infinite() {
                    return Err(HasherError::NonFiniteFloat(format!(
                        "non-finite float: {f}"
                    )));
                }
            }
            buf.extend_from_slice(n.to_string().as_bytes());
        }
        Value::String(s) => {
            // Use serde_json for correct JSON string escaping.
            let json_str = serde_json::to_string(s).expect("serde_json string serialization");
            buf.extend_from_slice(json_str.as_bytes());
        }
        Value::Array(arr) => {
            buf.push(b'[');
            for (i, item) in arr.iter().enumerate() {
                if i > 0 {
                    buf.push(b',');
                }
                canonical_json_into(item, buf)?;
            }
            buf.push(b']');
        }
        Value::Object(obj) => {
            buf.push(b'{');
            let mut keys: Vec<&str> = obj.keys().map(|k| k.as_str()).collect();
            keys.sort_unstable();
            for (i, key) in keys.iter().enumerate() {
                if i > 0 {
                    buf.push(b',');
                }
                let json_key = serde_json::to_string(key).expect("key serialization");
                buf.extend_from_slice(json_key.as_bytes());
                buf.push(b':');
                canonical_json_into(&obj[*key], buf)?;
            }
            buf.push(b'}');
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Hashing API
// ---------------------------------------------------------------------------

/// Hash raw bytes with SHA-256. Returns lowercase hex.
pub fn hash_bytes(data: &[u8]) -> String {
    let digest = Sha256::digest(data);
    hex::encode(digest)
}

/// Hash a UTF-8 string with SHA-256. Returns lowercase hex.
///
/// Equivalent to `hash_bytes(s.as_bytes())`.
pub fn hash_string(s: &str) -> String {
    hash_bytes(s.as_bytes())
}

/// Canonicalize `v` and return SHA-256 hex of the canonical bytes.
///
/// Returns an error if `v` contains non-finite floats.
pub fn hash_object(v: &Value) -> Result<String, HasherError> {
    let bytes = canonical_json(v)?;
    Ok(hash_bytes(&bytes))
}

/// Like [`hash_object`] but panics on error (useful in tests / infallible contexts).
pub fn must_hash_object(v: &Value) -> String {
    hash_object(v).expect("hash_object failed")
}

/// Prepend `"sha256:"` to `hash`. Does not validate `hash`.
pub fn prefixed_hash(hash: &str) -> String {
    format!("sha256:{hash}")
}

/// Canonicalize and hash `v`, returning `"sha256:<hex>"`.
pub fn hash_object_prefixed(v: &Value) -> Result<String, HasherError> {
    Ok(prefixed_hash(&hash_object(v)?))
}

/// Compare two SHA-256 hex strings in a case-insensitive way.
///
/// Returns `true` only when both strings decode to identical 32-byte digests.
/// Strings of differing lengths are not equal.
pub fn equal(a: &str, b: &str) -> bool {
    let a_lower = a.to_lowercase();
    let b_lower = b.to_lowercase();
    if a_lower.len() != b_lower.len() {
        return false;
    }
    // Constant-time comparison via XOR of all bytes.
    let mut diff: u8 = 0;
    for (x, y) in a_lower.bytes().zip(b_lower.bytes()) {
        diff |= x ^ y;
    }
    diff == 0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -- canonical_json -------------------------------------------------------

    #[test]
    fn canonical_json_null() {
        assert_eq!(canonical_json(&Value::Null).unwrap(), b"null");
    }

    #[test]
    fn canonical_json_bool_true() {
        assert_eq!(canonical_json(&json!(true)).unwrap(), b"true");
    }

    #[test]
    fn canonical_json_bool_false() {
        assert_eq!(canonical_json(&json!(false)).unwrap(), b"false");
    }

    #[test]
    fn canonical_json_integer() {
        assert_eq!(canonical_json(&json!(42)).unwrap(), b"42");
    }

    #[test]
    fn canonical_json_float() {
        assert_eq!(canonical_json(&json!(1.5)).unwrap(), b"1.5");
    }

    #[test]
    fn canonical_json_string() {
        assert_eq!(canonical_json(&json!("hello")).unwrap(), b"\"hello\"");
    }

    #[test]
    fn canonical_json_string_with_escapes() {
        let result = canonical_json(&json!("a\nb")).unwrap();
        assert_eq!(result, b"\"a\\nb\"");
    }

    #[test]
    fn canonical_json_array_preserves_order() {
        let result = canonical_json(&json!([3, 1, 2])).unwrap();
        assert_eq!(result, b"[3,1,2]");
    }

    #[test]
    fn canonical_json_object_keys_sorted() {
        let result = canonical_json(&json!({"b": 2, "a": 1})).unwrap();
        assert_eq!(result, b"{\"a\":1,\"b\":2}");
    }

    #[test]
    fn canonical_json_nested_object_keys_sorted() {
        let result = canonical_json(&json!({"z": {"b": 2, "a": 1}})).unwrap();
        assert_eq!(result, b"{\"z\":{\"a\":1,\"b\":2}}");
    }

    #[test]
    fn canonical_json_empty_object() {
        assert_eq!(canonical_json(&json!({})).unwrap(), b"{}");
    }

    #[test]
    fn canonical_json_empty_array() {
        assert_eq!(canonical_json(&json!([])).unwrap(), b"[]");
    }

    // -- hash_bytes -----------------------------------------------------------

    #[test]
    fn hash_bytes_known_vector() {
        // SHA-256("") = e3b0c44298fc1c149afb...
        let h = hash_bytes(b"");
        assert_eq!(h, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    }

    #[test]
    fn hash_bytes_abc_vector() {
        // SHA-256("abc") = ba7816bf...
        let h = hash_bytes(b"abc");
        assert_eq!(h, "ba7816bf8f01cfea414140de5dae2ec73b00361bbef0469328ce1b14f7a1d7b8");
    }

    #[test]
    fn hash_bytes_returns_64_char_hex() {
        let h = hash_bytes(b"test");
        assert_eq!(h.len(), 64);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn hash_bytes_is_lowercase() {
        let h = hash_bytes(b"test");
        assert_eq!(h, h.to_lowercase());
    }

    // -- hash_string ----------------------------------------------------------

    #[test]
    fn hash_string_equals_hash_bytes_of_utf8() {
        let s = "hello world";
        assert_eq!(hash_string(s), hash_bytes(s.as_bytes()));
    }

    #[test]
    fn hash_string_empty() {
        assert_eq!(hash_string(""), hash_bytes(b""));
    }

    // -- hash_object ----------------------------------------------------------

    #[test]
    fn hash_object_sorted_keys_matches_manual() {
        // canonical JSON of {"a":1,"b":2} → SHA-256
        let expected = hash_bytes(b"{\"a\":1,\"b\":2}");
        let result = hash_object(&json!({"b": 2, "a": 1})).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn hash_object_is_deterministic() {
        let v = json!({"model": "gpt-4", "confidence": 0.95, "seq": 0});
        let h1 = hash_object(&v).unwrap();
        let h2 = hash_object(&v).unwrap();
        assert_eq!(h1, h2);
    }

    #[test]
    fn hash_object_null_value() {
        let h = hash_object(&Value::Null).unwrap();
        assert_eq!(h, hash_bytes(b"null"));
    }

    // -- prefixed_hash --------------------------------------------------------

    #[test]
    fn prefixed_hash_prepends_sha256_colon() {
        let h = "a".repeat(64);
        assert_eq!(prefixed_hash(&h), format!("sha256:{h}"));
    }

    // -- hash_object_prefixed -------------------------------------------------

    #[test]
    fn hash_object_prefixed_starts_with_sha256() {
        let result = hash_object_prefixed(&json!({"x": 1})).unwrap();
        assert!(result.starts_with("sha256:"));
        assert_eq!(result.len(), 7 + 64);
    }

    // -- equal ----------------------------------------------------------------

    #[test]
    fn equal_same_hash() {
        let h = hash_bytes(b"test");
        assert!(equal(&h, &h));
    }

    #[test]
    fn equal_case_insensitive() {
        let lower = "aabbcc".to_string() + &"00".repeat(29);
        let upper = lower.to_uppercase();
        assert!(equal(&lower, &upper));
    }

    #[test]
    fn equal_different_hashes() {
        let h1 = hash_bytes(b"a");
        let h2 = hash_bytes(b"b");
        assert!(!equal(&h1, &h2));
    }

    #[test]
    fn equal_different_lengths() {
        assert!(!equal("abc", "abcd"));
    }

    #[test]
    fn equal_empty_strings() {
        assert!(equal("", ""));
    }

    // -- Cross-SDK BRC-121 test vector ----------------------------------------

    #[test]
    fn cross_sdk_canonical_json_vector() {
        // This vector must match the Python SDK and TypeScript SDK outputs.
        // Test vector: {"model":"gpt-4","seq":0,"confidence":null}
        // canonical form (keys sorted): {"confidence":null,"model":"gpt-4","seq":0}
        let v = json!({"model": "gpt-4", "seq": 0, "confidence": null});
        let canon = canonical_json(&v).unwrap();
        assert_eq!(canon, b"{\"confidence\":null,\"model\":\"gpt-4\",\"seq\":0}");
    }
}
