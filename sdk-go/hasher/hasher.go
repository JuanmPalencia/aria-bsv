// Package hasher provides canonical JSON serialization and SHA-256 hashing
// compatible with the ARIA BRC-121 protocol.
package hasher

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// CanonicalJSONError is returned when an object cannot be canonicalized.
type CanonicalJSONError struct {
	Msg string
}

func (e *CanonicalJSONError) Error() string {
	return fmt.Sprintf("canonical JSON error: %s", e.Msg)
}

// CanonicalJSON returns the deterministic JSON encoding of v.
// Object keys are sorted lexicographically. Arrays preserve order.
// Panics or returns error on non-serializable types (channels, funcs).
func CanonicalJSON(v interface{}) ([]byte, error) {
	return canonicalize(v)
}

func canonicalize(v interface{}) ([]byte, error) {
	if v == nil {
		return []byte("null"), nil
	}

	switch val := v.(type) {
	case bool:
		if val {
			return []byte("true"), nil
		}
		return []byte("false"), nil

	case float64:
		// Use standard JSON encoding for numbers
		b, err := json.Marshal(val)
		if err != nil {
			return nil, err
		}
		return b, nil

	case float32:
		return canonicalize(float64(val))

	case int:
		return []byte(fmt.Sprintf("%d", val)), nil
	case int8:
		return []byte(fmt.Sprintf("%d", val)), nil
	case int16:
		return []byte(fmt.Sprintf("%d", val)), nil
	case int32:
		return []byte(fmt.Sprintf("%d", val)), nil
	case int64:
		return []byte(fmt.Sprintf("%d", val)), nil
	case uint:
		return []byte(fmt.Sprintf("%d", val)), nil
	case uint8:
		return []byte(fmt.Sprintf("%d", val)), nil
	case uint16:
		return []byte(fmt.Sprintf("%d", val)), nil
	case uint32:
		return []byte(fmt.Sprintf("%d", val)), nil
	case uint64:
		return []byte(fmt.Sprintf("%d", val)), nil

	case string:
		b, err := json.Marshal(val)
		if err != nil {
			return nil, err
		}
		return b, nil

	case []interface{}:
		return canonicalizeArray(val)

	case map[string]interface{}:
		return canonicalizeObject(val)

	default:
		// Fallback: marshal with standard JSON then re-canonicalize
		b, err := json.Marshal(val)
		if err != nil {
			return nil, &CanonicalJSONError{Msg: fmt.Sprintf("cannot marshal type %T", v)}
		}
		var decoded interface{}
		if err := json.Unmarshal(b, &decoded); err != nil {
			return nil, err
		}
		return canonicalize(decoded)
	}
}

func canonicalizeArray(arr []interface{}) ([]byte, error) {
	if len(arr) == 0 {
		return []byte("[]"), nil
	}
	var sb strings.Builder
	sb.WriteByte('[')
	for i, item := range arr {
		if i > 0 {
			sb.WriteByte(',')
		}
		b, err := canonicalize(item)
		if err != nil {
			return nil, err
		}
		sb.Write(b)
	}
	sb.WriteByte(']')
	return []byte(sb.String()), nil
}

func canonicalizeObject(obj map[string]interface{}) ([]byte, error) {
	keys := make([]string, 0, len(obj))
	for k := range obj {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	if len(keys) == 0 {
		return []byte("{}"), nil
	}

	var sb strings.Builder
	sb.WriteByte('{')
	for i, k := range keys {
		if i > 0 {
			sb.WriteByte(',')
		}
		keyBytes, err := json.Marshal(k)
		if err != nil {
			return nil, err
		}
		sb.Write(keyBytes)
		sb.WriteByte(':')
		valBytes, err := canonicalize(obj[k])
		if err != nil {
			return nil, err
		}
		sb.Write(valBytes)
	}
	sb.WriteByte('}')
	return []byte(sb.String()), nil
}

// HashObject computes SHA-256 of the canonical JSON encoding of v.
// Returns a lowercase hex string (64 chars).
func HashObject(v interface{}) (string, error) {
	data, err := CanonicalJSON(v)
	if err != nil {
		return "", err
	}
	return HashBytes(data), nil
}

// HashBytes computes SHA-256 of raw bytes. Returns a lowercase hex string.
func HashBytes(data []byte) string {
	h := sha256.Sum256(data)
	return hex.EncodeToString(h[:])
}

// HashString computes SHA-256 of a UTF-8 string. Returns a lowercase hex string.
func HashString(s string) string {
	return HashBytes([]byte(s))
}

// MustHashObject is like HashObject but panics on error.
// Use only when the input is guaranteed to be JSON-serializable.
func MustHashObject(v interface{}) string {
	h, err := HashObject(v)
	if err != nil {
		panic(fmt.Sprintf("hasher.MustHashObject: %v", err))
	}
	return h
}

// PrefixedHash returns the hash in "sha256:<hex>" format, matching the
// Python SDK's hash_object() output used in DatasetAnchor records.
func PrefixedHash(hash string) string {
	return "sha256:" + hash
}

// HashObjectPrefixed computes the canonical hash and returns it in
// "sha256:<hex>" format.
func HashObjectPrefixed(v interface{}) (string, error) {
	h, err := HashObject(v)
	if err != nil {
		return "", err
	}
	return PrefixedHash(h), nil
}

// Equal reports whether two hex-encoded SHA-256 hashes are identical.
// Comparison is case-insensitive and constant-time against timing attacks.
func Equal(a, b string) bool {
	if len(a) != len(b) {
		return false
	}
	// Decode both; fall back to string compare on decode error
	aBytes, errA := hex.DecodeString(a)
	bBytes, errB := hex.DecodeString(b)
	if errA != nil || errB != nil {
		return a == b
	}
	// constant-time byte comparison
	var diff byte
	for i := range aBytes {
		diff |= aBytes[i] ^ bBytes[i]
	}
	return diff == 0
}
