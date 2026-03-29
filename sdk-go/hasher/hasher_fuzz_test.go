// Package hasher — fuzz tests for canonical JSON and SHA-256 hashing.
//
// Run with: go test -fuzz=Fuzz -fuzztime=30s ./hasher/
// Seed corpus entries are run as normal unit tests: go test ./hasher/
package hasher

import (
	"encoding/hex"
	"strings"
	"testing"
)

// FuzzHashBytes verifies that HashBytes always returns a valid 64-char lowercase
// hex string and that identical inputs produce identical outputs (determinism).
func FuzzHashBytes(f *testing.F) {
	// Seed corpus
	f.Add([]byte(""))
	f.Add([]byte("hello"))
	f.Add([]byte{0x00})
	f.Add([]byte{0xff, 0xfe, 0xfd})
	f.Add([]byte("ARIA BRC-121 protocol"))
	f.Add([]byte(`{"model":"gpt-4","confidence":0.95}`))

	f.Fuzz(func(t *testing.T, data []byte) {
		h := HashBytes(data)

		// Must be exactly 64 hex characters.
		if len(h) != 64 {
			t.Fatalf("HashBytes returned %d-char string, want 64: %q", len(h), h)
		}

		// Must be lowercase hex.
		if h != strings.ToLower(h) {
			t.Fatalf("HashBytes returned non-lowercase hex: %q", h)
		}
		if _, err := hex.DecodeString(h); err != nil {
			t.Fatalf("HashBytes returned non-hex string: %q", h)
		}

		// Determinism: same input → same output.
		h2 := HashBytes(data)
		if h != h2 {
			t.Fatalf("HashBytes is not deterministic: %q != %q", h, h2)
		}
	})
}

// FuzzHashString verifies determinism and format for arbitrary string inputs.
func FuzzHashString(f *testing.F) {
	f.Add("")
	f.Add("hello world")
	f.Add("null")
	f.Add("\x00\x01\x02")
	f.Add("unicode: 日本語")

	f.Fuzz(func(t *testing.T, s string) {
		h := HashString(s)

		if len(h) != 64 {
			t.Fatalf("HashString returned %d-char string, want 64", len(h))
		}
		if _, err := hex.DecodeString(h); err != nil {
			t.Fatalf("HashString returned non-hex string: %q", h)
		}

		// Must equal HashBytes of the same UTF-8 bytes.
		expected := HashBytes([]byte(s))
		if h != expected {
			t.Fatalf("HashString(%q) = %q, want %q (from HashBytes)", s, h, expected)
		}
	})
}

// FuzzHashObject_StringValue verifies that a map with one string value always
// produces a 64-char hex hash and is deterministic.
func FuzzHashObject_StringValue(f *testing.F) {
	f.Add("key", "value")
	f.Add("model_id", "gpt-4o")
	f.Add("", "")
	f.Add("prompt", "What is 2+2?")

	f.Fuzz(func(t *testing.T, key, value string) {
		obj := map[string]interface{}{key: value}

		h1, err1 := HashObject(obj)
		h2, err2 := HashObject(obj)

		// Both calls must succeed or both fail consistently.
		if (err1 == nil) != (err2 == nil) {
			t.Fatalf("HashObject non-deterministic error: %v vs %v", err1, err2)
		}
		if err1 != nil {
			return // accept consistent errors (e.g. for NaN values)
		}

		if len(h1) != 64 {
			t.Fatalf("HashObject returned %d-char hash, want 64", len(h1))
		}
		if h1 != h2 {
			t.Fatalf("HashObject not deterministic for obj=%v: %q vs %q", obj, h1, h2)
		}
	})
}

// FuzzPrefixedHash verifies that PrefixedHash always produces "sha256:<64hex>".
func FuzzPrefixedHash(f *testing.F) {
	f.Add("")
	f.Add("abc")
	f.Add(strings.Repeat("a", 64))
	f.Add("not-a-valid-hash")

	f.Fuzz(func(t *testing.T, rawHash string) {
		result := PrefixedHash(rawHash)

		if !strings.HasPrefix(result, "sha256:") {
			t.Fatalf("PrefixedHash(%q) = %q, want sha256: prefix", rawHash, result)
		}
		if result != "sha256:"+rawHash {
			t.Fatalf("PrefixedHash(%q) = %q, want %q", rawHash, result, "sha256:"+rawHash)
		}
	})
}

// FuzzEqual verifies that Equal is always symmetric.
func FuzzEqual(f *testing.F) {
	f.Add("aabbcc", "aabbcc")
	f.Add("AABBCC", "aabbcc")
	f.Add("", "")
	f.Add("abc", "def")

	f.Fuzz(func(t *testing.T, a, b string) {
		ab := Equal(a, b)
		ba := Equal(b, a)

		// Symmetry: Equal(a,b) == Equal(b,a)
		if ab != ba {
			t.Fatalf("Equal is not symmetric: Equal(%q,%q)=%v but Equal(%q,%q)=%v", a, b, ab, b, a, ba)
		}

		// Reflexivity: Equal(a,a) == true whenever a is valid lowercase hex of even length.
		if _, err := hex.DecodeString(a); err == nil && a == strings.ToLower(a) {
			if !Equal(a, a) {
				t.Fatalf("Equal(%q,%q) should be true (reflexivity)", a, a)
			}
		}
	})
}
