/**
 * Tests for aria-bsv/hasher — canonicalJson, hashObject, SHA-256 helpers.
 */

import { describe, it, expect } from "vitest";
import {
  canonicalJson,
  hashObject,
  hashObjectSync,
  sha256HexSync,
  ARIASerializationError,
} from "./hasher.js";

// ---------------------------------------------------------------------------
// canonicalJson
// ---------------------------------------------------------------------------

describe("canonicalJson", () => {
  it("serialises null", () => {
    expect(canonicalJson(null)).toBe("null");
  });

  it("serialises true", () => {
    expect(canonicalJson(true)).toBe("true");
  });

  it("serialises false", () => {
    expect(canonicalJson(false)).toBe("false");
  });

  it("serialises integer", () => {
    expect(canonicalJson(42)).toBe("42");
  });

  it("serialises negative integer", () => {
    expect(canonicalJson(-7)).toBe("-7");
  });

  it("serialises float", () => {
    expect(canonicalJson(3.14)).toBe("3.14");
  });

  it("serialises string", () => {
    expect(canonicalJson("hello")).toBe('"hello"');
  });

  it("serialises empty string", () => {
    expect(canonicalJson("")).toBe('""');
  });

  it("serialises empty array", () => {
    expect(canonicalJson([])).toBe("[]");
  });

  it("serialises array preserving order", () => {
    expect(canonicalJson([3, 1, 2])).toBe("[3, 1, 2]");
  });

  it("serialises nested array", () => {
    expect(canonicalJson([[1, 2], [3]])).toBe("[[1, 2], [3]]");
  });

  it("serialises empty object", () => {
    expect(canonicalJson({})).toBe("{}");
  });

  it("sorts object keys", () => {
    expect(canonicalJson({ b: 2, a: 1 })).toBe('{"a": 1, "b": 2}');
  });

  it("sorts nested object keys", () => {
    expect(canonicalJson({ z: { y: 1, x: 2 } })).toBe('{"z": {"x": 2, "y": 1}}');
  });

  it("is deterministic for same input", () => {
    const obj = { foo: [1, 2], bar: { baz: true } };
    expect(canonicalJson(obj)).toBe(canonicalJson(obj));
  });

  it("different key ordering produces same output", () => {
    expect(canonicalJson({ x: 1, y: 2 })).toBe(canonicalJson({ y: 2, x: 1 }));
  });

  it("throws on NaN", () => {
    expect(() => canonicalJson(NaN)).toThrow(ARIASerializationError);
  });

  it("throws on Infinity", () => {
    expect(() => canonicalJson(Infinity)).toThrow(ARIASerializationError);
  });

  it("throws on -Infinity", () => {
    expect(() => canonicalJson(-Infinity)).toThrow(ARIASerializationError);
  });

  it("throws on function", () => {
    // eslint-disable-next-line @typescript-eslint/no-empty-function
    expect(() => canonicalJson(() => {})).toThrow(ARIASerializationError);
  });

  it("throws on symbol", () => {
    expect(() => canonicalJson(Symbol("x"))).toThrow(ARIASerializationError);
  });

  it("throws on bigint", () => {
    expect(() => canonicalJson(BigInt(42))).toThrow(ARIASerializationError);
  });

  it("throws on undefined", () => {
    expect(() => canonicalJson(undefined)).toThrow(ARIASerializationError);
  });
});

// ---------------------------------------------------------------------------
// sha256HexSync — known test vectors (RFC 6234 / NIST FIPS 180-4)
// ---------------------------------------------------------------------------

describe("sha256HexSync", () => {
  it('SHA-256("") is the empty-string digest', () => {
    expect(sha256HexSync("")).toBe(
      "e3b0c44298fc1c149afbf4c8996fb924" +
      "27ae41e4649b934ca495991b7852b855"
    );
  });

  it('SHA-256("abc") matches NIST vector', () => {
    expect(sha256HexSync("abc")).toBe(
      "ba7816bf8f01cfea414140de5dae2223" +
      "b00361a396177a9cb410ff61f20015ad"
    );
  });

  it("returns 64-char lowercase hex", () => {
    const h = sha256HexSync("test");
    expect(h).toHaveLength(64);
    expect(h).toMatch(/^[0-9a-f]+$/);
  });

  it("different inputs produce different hashes", () => {
    expect(sha256HexSync("foo")).not.toBe(sha256HexSync("bar"));
  });
});

// ---------------------------------------------------------------------------
// hashObjectSync
// ---------------------------------------------------------------------------

describe("hashObjectSync", () => {
  it("returns a 64-char hex string", () => {
    const h = hashObjectSync({ foo: "bar" });
    expect(h).toHaveLength(64);
    expect(h).toMatch(/^[0-9a-f]{64}$/);
  });

  it("is deterministic", () => {
    expect(hashObjectSync({ x: 1 })).toBe(hashObjectSync({ x: 1 }));
  });

  it("same canonical JSON → same hash regardless of key insertion order", () => {
    expect(hashObjectSync({ a: 1, b: 2 })).toBe(hashObjectSync({ b: 2, a: 1 }));
  });

  it("different values → different hashes", () => {
    expect(hashObjectSync({ x: 1 })).not.toBe(hashObjectSync({ x: 2 }));
  });

  it('hash of null is SHA-256 of "null"', () => {
    // SHA-256("null") first 8 chars = "74234e98"
    expect(hashObjectSync(null).slice(0, 8)).toBe("74234e98");
  });
});

// ---------------------------------------------------------------------------
// hashObject (async)
// ---------------------------------------------------------------------------

describe("hashObject", () => {
  it("returns same result as hashObjectSync", async () => {
    const obj = { model: "gpt-4o", tokens: 100 };
    expect(await hashObject(obj)).toBe(hashObjectSync(obj));
  });

  it("returns 64-char hex", async () => {
    const h = await hashObject([1, 2, 3]);
    expect(h).toHaveLength(64);
    expect(h).toMatch(/^[0-9a-f]{64}$/);
  });

  it("async and sync agree on nested object", async () => {
    const obj = { a: [1, null, true], b: { c: "hello" } };
    expect(await hashObject(obj)).toBe(hashObjectSync(obj));
  });
});
