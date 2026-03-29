package io.aria.bsv.hasher;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link Hasher} and {@link CanonicalJson}.
 *
 * <p>Includes BRC-121 cross-SDK test vectors:
 * <ul>
 *   <li>SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855</li>
 *   <li>SHA-256("abc") = ba7816bf8f01cfea414140de5dae2ec73b00361bbef0469328ce1b14f7a1d7b8</li>
 *   <li>canonical({"b":2,"a":1}) = {"a":1,"b":2}</li>
 * </ul>
 */
class HasherTest {

    // =========================================================================
    // hashBytes — known SHA-256 vectors
    // =========================================================================

    @Test
    void hashBytes_emptyInputKnownVector() {
        String hash = Hasher.hashBytes(new byte[0]);
        assertEquals(
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                hash,
                "SHA-256 of empty bytes must match NIST vector");
    }

    @Test
    void hashBytes_abcKnownVector() {
        String hash = Hasher.hashBytes("abc".getBytes(java.nio.charset.StandardCharsets.UTF_8));
        assertEquals(
                "ba7816bf8f01cfea414140de5dae2ec73b00361bbef0469328ce1b14f7a1d7b8",
                hash,
                "SHA-256 of 'abc' must match NIST vector");
    }

    @Test
    void hashBytes_returns64LowercaseHexChars() {
        String hash = Hasher.hashBytes("hello".getBytes(java.nio.charset.StandardCharsets.UTF_8));
        assertEquals(64, hash.length());
        assertEquals(hash, hash.toLowerCase(java.util.Locale.ROOT));
        assertTrue(hash.matches("[0-9a-f]+"));
    }

    // =========================================================================
    // hashString
    // =========================================================================

    @Test
    void hashString_equivalentToHashBytesUtf8() {
        String s = "hello world";
        assertEquals(
                Hasher.hashString(s),
                Hasher.hashBytes(s.getBytes(java.nio.charset.StandardCharsets.UTF_8)));
    }

    @Test
    void hashString_emptyStringKnownVector() {
        assertEquals(
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                Hasher.hashString(""));
    }

    @Test
    void hashString_abcKnownVector() {
        assertEquals(
                "ba7816bf8f01cfea414140de5dae2ec73b00361bbef0469328ce1b14f7a1d7b8",
                Hasher.hashString("abc"));
    }

    // =========================================================================
    // hashObject — canonical JSON key ordering (BRC-121 cross-SDK vector)
    // =========================================================================

    @Test
    void hashObject_canonicalJsonKeysSorted() {
        // BRC-121 cross-SDK vector: {"b":2,"a":1} → canonical bytes = {"a":1,"b":2}
        Map<String, Object> unsorted = new LinkedHashMap<>();
        unsorted.put("b", 2);
        unsorted.put("a", 1);
        String canonical = CanonicalJson.toString(unsorted);
        assertEquals("{\"a\":1,\"b\":2}", canonical,
                "Keys must be sorted lexicographically");
    }

    @Test
    void hashObject_keyOrderIndependent() {
        Map<String, Object> m1 = new LinkedHashMap<>();
        m1.put("b", 2);
        m1.put("a", 1);

        Map<String, Object> m2 = new LinkedHashMap<>();
        m2.put("a", 1);
        m2.put("b", 2);

        assertEquals(Hasher.hashObject(m1), Hasher.hashObject(m2),
                "Different insertion order must produce the same hash");
    }

    @Test
    void hashObject_differentObjectsDifferentHash() {
        Map<String, Object> m1 = Collections.singletonMap("a", 1);
        Map<String, Object> m2 = Collections.singletonMap("a", 2);
        assertNotEquals(Hasher.hashObject(m1), Hasher.hashObject(m2));
    }

    @Test
    void hashObject_nullValueSerializesAsNull() {
        String json = CanonicalJson.toString(null);
        assertEquals("null", json);
    }

    @Test
    void hashObject_booleanTrue() {
        assertEquals("true", CanonicalJson.toString(Boolean.TRUE));
    }

    @Test
    void hashObject_booleanFalse() {
        assertEquals("false", CanonicalJson.toString(Boolean.FALSE));
    }

    @Test
    void hashObject_arrayPreservesOrder() {
        List<Object> arr = Arrays.asList(3, 1, 2);
        assertEquals("[3,1,2]", CanonicalJson.toString(arr));
    }

    @Test
    void hashObject_emptyObject() {
        assertEquals("{}", CanonicalJson.toString(new HashMap<>()));
    }

    @Test
    void hashObject_emptyArray() {
        assertEquals("[]", CanonicalJson.toString(Collections.emptyList()));
    }

    @Test
    void hashObject_nestedObjectKeysSorted() {
        Map<String, Object> inner = new LinkedHashMap<>();
        inner.put("z", 1);
        inner.put("a", 2);
        Map<String, Object> outer = new LinkedHashMap<>();
        outer.put("outer", inner);

        String json = CanonicalJson.toString(outer);
        int aIdx = json.indexOf("\"a\"");
        int zIdx = json.indexOf("\"z\"");
        assertTrue(aIdx < zIdx, "Nested object keys must be sorted: " + json);
    }

    @Test
    void hashObject_stringWithSpecialCharacters() {
        String json = CanonicalJson.toString("say \"hi\"");
        assertTrue(json.contains("\\\""),
                "Quotes inside strings must be escaped: " + json);
    }

    @Test
    void hashObject_deterministicAcrossCalls() {
        Map<String, Object> obj = new LinkedHashMap<>();
        obj.put("b", 2);
        obj.put("a", 1);
        assertEquals(Hasher.hashObject(obj), Hasher.hashObject(obj),
                "Same object must hash identically across calls");
    }

    // =========================================================================
    // prefixedHash / hashObjectPrefixed
    // =========================================================================

    @Test
    void prefixedHash_prependsSha256Colon() {
        String hash = "ba7816bf8f01cfea414140de5dae2ec73b00361bbef0469328ce1b14f7a1d7b8";
        assertEquals("sha256:" + hash, Hasher.prefixedHash(hash));
    }

    @Test
    void hashObjectPrefixed_hasSha256Prefix() {
        String prefixed = Hasher.hashObjectPrefixed(Collections.singletonMap("x", 1));
        assertTrue(prefixed.startsWith("sha256:"),
                "hashObjectPrefixed must start with 'sha256:': " + prefixed);
        assertEquals(7 + 64, prefixed.length(),
                "Total length must be 7 + 64 chars: " + prefixed);
    }

    // =========================================================================
    // equal — case-insensitive constant-time comparison
    // =========================================================================

    @Test
    void equal_sameHashIsEqual() {
        String h = Hasher.hashString("test");
        assertTrue(Hasher.equal(h, h));
    }

    @Test
    void equal_differentHashesNotEqual() {
        String h1 = Hasher.hashString("a");
        String h2 = Hasher.hashString("b");
        assertFalse(Hasher.equal(h1, h2));
    }

    @Test
    void equal_caseInsensitive() {
        String lower = Hasher.hashString("hello");
        String upper = lower.toUpperCase(java.util.Locale.ROOT);
        assertTrue(Hasher.equal(lower, upper),
                "equal() must be case-insensitive");
    }

    @Test
    void equal_differentLengthsNotEqual() {
        assertFalse(Hasher.equal("abc", "ab"));
    }

    @Test
    void equal_nullsHandledGracefully() {
        assertFalse(Hasher.equal(null, "abc"));
        assertFalse(Hasher.equal("abc", null));
        assertTrue(Hasher.equal(null, null));
    }
}
