package hasher_test

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"testing"

	"github.com/JuanmPalencia/aria-bsv/sdk-go/hasher"
)

func TestCanonicalJSON_NilIsNull(t *testing.T) {
	b, err := hasher.CanonicalJSON(nil)
	if err != nil {
		t.Fatal(err)
	}
	if string(b) != "null" {
		t.Errorf("expected null, got %s", b)
	}
}

func TestCanonicalJSON_BoolTrue(t *testing.T) {
	b, _ := hasher.CanonicalJSON(true)
	if string(b) != "true" {
		t.Errorf("got %s", b)
	}
}

func TestCanonicalJSON_BoolFalse(t *testing.T) {
	b, _ := hasher.CanonicalJSON(false)
	if string(b) != "false" {
		t.Errorf("got %s", b)
	}
}

func TestCanonicalJSON_Integer(t *testing.T) {
	b, _ := hasher.CanonicalJSON(42)
	if string(b) != "42" {
		t.Errorf("got %s", b)
	}
}

func TestCanonicalJSON_NegativeInt(t *testing.T) {
	b, _ := hasher.CanonicalJSON(-7)
	if string(b) != "-7" {
		t.Errorf("got %s", b)
	}
}

func TestCanonicalJSON_String(t *testing.T) {
	b, _ := hasher.CanonicalJSON("hello")
	if string(b) != `"hello"` {
		t.Errorf("got %s", b)
	}
}

func TestCanonicalJSON_StringWithSpecialChars(t *testing.T) {
	b, _ := hasher.CanonicalJSON(`say "hi"`)
	if !strings.Contains(string(b), `\"`) {
		t.Errorf("expected escaped quotes, got %s", b)
	}
}

func TestCanonicalJSON_EmptyObject(t *testing.T) {
	b, _ := hasher.CanonicalJSON(map[string]interface{}{})
	if string(b) != "{}" {
		t.Errorf("got %s", b)
	}
}

func TestCanonicalJSON_KeysSorted(t *testing.T) {
	obj := map[string]interface{}{"z": 1, "a": 2, "m": 3}
	b, _ := hasher.CanonicalJSON(obj)
	s := string(b)
	aIdx := strings.Index(s, `"a"`)
	mIdx := strings.Index(s, `"m"`)
	zIdx := strings.Index(s, `"z"`)
	if !(aIdx < mIdx && mIdx < zIdx) {
		t.Errorf("keys not sorted: %s", s)
	}
}

func TestCanonicalJSON_DeterministicAcrossCallsWithSameObject(t *testing.T) {
	obj := map[string]interface{}{"b": 2, "a": 1}
	b1, _ := hasher.CanonicalJSON(obj)
	b2, _ := hasher.CanonicalJSON(obj)
	if string(b1) != string(b2) {
		t.Error("not deterministic")
	}
}

func TestCanonicalJSON_NestedObject(t *testing.T) {
	obj := map[string]interface{}{
		"outer": map[string]interface{}{"z": 1, "a": 2},
	}
	b, _ := hasher.CanonicalJSON(obj)
	s := string(b)
	aIdx := strings.Index(s, `"a"`)
	zIdx := strings.Index(s, `"z"`)
	if !(aIdx < zIdx) {
		t.Errorf("nested keys not sorted: %s", s)
	}
}

func TestCanonicalJSON_ArrayPreservesOrder(t *testing.T) {
	arr := []interface{}{3, 1, 2}
	b, _ := hasher.CanonicalJSON(arr)
	if string(b) != "[3,1,2]" {
		t.Errorf("got %s", b)
	}
}

func TestCanonicalJSON_EmptyArray(t *testing.T) {
	b, _ := hasher.CanonicalJSON([]interface{}{})
	if string(b) != "[]" {
		t.Errorf("got %s", b)
	}
}

func TestHashBytes_KnownVector(t *testing.T) {
	// SHA-256("") = e3b0c442...
	h := hasher.HashBytes([]byte{})
	sum := sha256.Sum256([]byte{})
	expected := hex.EncodeToString(sum[:])
	if h != expected {
		t.Errorf("got %s, want %s", h, expected)
	}
}

func TestHashBytes_ABCVector(t *testing.T) {
	h := hasher.HashBytes([]byte("abc"))
	expected := "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
	if h != expected {
		t.Errorf("got %s", h)
	}
}

func TestHashObject_SameAsHashBytes(t *testing.T) {
	obj := map[string]interface{}{"key": "value"}
	h, err := hasher.HashObject(obj)
	if err != nil {
		t.Fatal(err)
	}
	canonical, _ := hasher.CanonicalJSON(obj)
	expected := hasher.HashBytes(canonical)
	if h != expected {
		t.Errorf("HashObject != HashBytes(CanonicalJSON)")
	}
}

func TestHashObject_KeyOrderIndependent(t *testing.T) {
	obj1 := map[string]interface{}{"b": 2, "a": 1}
	obj2 := map[string]interface{}{"a": 1, "b": 2}
	h1, _ := hasher.HashObject(obj1)
	h2, _ := hasher.HashObject(obj2)
	if h1 != h2 {
		t.Errorf("key order should not affect hash: %s != %s", h1, h2)
	}
}

func TestHashObject_DifferentObjectsDifferentHash(t *testing.T) {
	h1, _ := hasher.HashObject(map[string]interface{}{"a": 1})
	h2, _ := hasher.HashObject(map[string]interface{}{"a": 2})
	if h1 == h2 {
		t.Error("different objects should have different hashes")
	}
}
