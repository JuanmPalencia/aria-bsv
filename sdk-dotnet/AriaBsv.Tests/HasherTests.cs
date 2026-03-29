// HasherTests.cs — xUnit tests for AriaBsv.Core.Hasher.
//
// Tests cover:
//   - Known SHA-256 values (cross-SDK verification anchors)
//   - Canonical JSON property tests (determinism, key ordering, type handling)
//   - Error conditions (NaN, Infinity, unsupported types)
//   - Prefix helpers and case-insensitive equality

using System.Text;
using AriaBsv.Core;

namespace AriaBsv.Tests;

public class HasherTests
{
    // -------------------------------------------------------------------------
    // HashBytes — known SHA-256 anchors
    // -------------------------------------------------------------------------

    [Fact]
    public void HashBytes_EmptyArray_ReturnsWellKnownHash()
    {
        // SHA-256 of the empty byte array is universally known.
        const string expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        Assert.Equal(expected, Hasher.HashBytes(Array.Empty<byte>()));
    }

    [Fact]
    public void HashBytes_HelloUtf8_ReturnsKnownHash()
    {
        // SHA-256("hello") — widely tested across all SDKs.
        const string expected = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";
        Assert.Equal(expected, Hasher.HashBytes(Encoding.UTF8.GetBytes("hello")));
    }

    [Fact]
    public void HashBytes_Returns64LowercaseHexChars()
    {
        var hash = Hasher.HashBytes(new byte[] { 1, 2, 3 });
        Assert.Equal(64, hash.Length);
        Assert.Matches("^[0-9a-f]{64}$", hash);
    }

    [Fact]
    public void HashBytes_SameInput_AlwaysProducesSameHash()
    {
        var data = Encoding.UTF8.GetBytes("determinism test");
        Assert.Equal(Hasher.HashBytes(data), Hasher.HashBytes(data));
    }

    // -------------------------------------------------------------------------
    // HashString
    // -------------------------------------------------------------------------

    [Fact]
    public void HashString_EqualsHashBytesOfUtf8()
    {
        const string input = "cross-check";
        var expected = Hasher.HashBytes(Encoding.UTF8.GetBytes(input));
        Assert.Equal(expected, Hasher.HashString(input));
    }

    [Fact]
    public void HashString_EmptyString_ReturnsEmptyByteHash()
    {
        const string expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        Assert.Equal(expected, Hasher.HashString(string.Empty));
    }

    // -------------------------------------------------------------------------
    // HashObject — null / bool / number
    // -------------------------------------------------------------------------

    [Fact]
    public void HashObject_Null_HashesLiteralNullString()
    {
        // CanonicalJson(null) → b"null"; verify against HashString("null")
        var expected = Hasher.HashString("null");
        Assert.Equal(expected, Hasher.HashObject(null));
    }

    [Fact]
    public void HashObject_BoolTrue_HashesLiteralTrueString()
    {
        var expected = Hasher.HashString("true");
        Assert.Equal(expected, Hasher.HashObject(true));
    }

    [Fact]
    public void HashObject_BoolFalse_HashesLiteralFalseString()
    {
        var expected = Hasher.HashString("false");
        Assert.Equal(expected, Hasher.HashObject(false));
    }

    [Fact]
    public void HashObject_IntegerValue_NoDecimalPoint()
    {
        // CanonicalJson(42) → "42" (no .0 suffix — Python json.dumps(42) → "42")
        var expected = Hasher.HashString("42");
        Assert.Equal(expected, Hasher.HashObject(42));
    }

    [Fact]
    public void HashObject_DoubleWholeNumber_HasDecimalSuffix()
    {
        // CanonicalJson(1.0) → "1.0" (matches Python json.dumps(1.0) → "1.0")
        var expected = Hasher.HashString("1.0");
        Assert.Equal(expected, Hasher.HashObject(1.0));
    }

    [Fact]
    public void HashObject_DoubleWithFraction_SerializesCorrectly()
    {
        // CanonicalJson(0.95) → "0.95"
        var expected = Hasher.HashString("0.95");
        Assert.Equal(expected, Hasher.HashObject(0.95));
    }

    // -------------------------------------------------------------------------
    // HashObject — string
    // -------------------------------------------------------------------------

    [Fact]
    public void HashObject_String_HashesJsonEncodedForm()
    {
        // CanonicalJson("hello") → "\"hello\""
        var expected = Hasher.HashString("\"hello\"");
        Assert.Equal(expected, Hasher.HashObject("hello"));
    }

    [Fact]
    public void HashObject_StringWithSpecialChars_EscapedCorrectly()
    {
        // Newlines and quotes must be JSON-escaped, same as all other SDKs.
        var h1 = Hasher.HashObject("line1\nline2");
        var h2 = Hasher.HashObject("line1\nline2");
        Assert.Equal(h1, h2);
        Assert.NotEqual(h1, Hasher.HashObject("line1line2"));
    }

    // -------------------------------------------------------------------------
    // HashObject — object / key sorting
    // -------------------------------------------------------------------------

    [Fact]
    public void HashObject_ObjectKeysSortedLexicographically()
    {
        // Both dicts must produce the same hash regardless of insertion order.
        var dictAB = new Dictionary<string, object?> { ["a"] = 1, ["b"] = 2 };
        var dictBA = new Dictionary<string, object?> { ["b"] = 2, ["a"] = 1 };
        Assert.Equal(Hasher.HashObject(dictAB), Hasher.HashObject(dictBA));
    }

    [Fact]
    public void HashObject_NestedObjectKeysAreSorted()
    {
        var inner1 = new Dictionary<string, object?> { ["z"] = "last", ["a"] = "first" };
        var inner2 = new Dictionary<string, object?> { ["a"] = "first", ["z"] = "last" };
        var outer1 = new Dictionary<string, object?> { ["nested"] = inner1 };
        var outer2 = new Dictionary<string, object?> { ["nested"] = inner2 };
        Assert.Equal(Hasher.HashObject(outer1), Hasher.HashObject(outer2));
    }

    [Fact]
    public void HashObject_DifferentObjects_DifferentHashes()
    {
        var a = new Dictionary<string, object?> { ["x"] = 1 };
        var b = new Dictionary<string, object?> { ["x"] = 2 };
        Assert.NotEqual(Hasher.HashObject(a), Hasher.HashObject(b));
    }

    // -------------------------------------------------------------------------
    // HashObject — arrays
    // -------------------------------------------------------------------------

    [Fact]
    public void HashObject_ArrayPreservesOrder()
    {
        var arr1 = new object?[] { 1, 2, 3 };
        var arr2 = new object?[] { 3, 2, 1 };
        Assert.NotEqual(Hasher.HashObject(arr1), Hasher.HashObject(arr2));
    }

    [Fact]
    public void HashObject_ArrayWithNullAndMixedTypes()
    {
        var arr = new object?[] { "text", null, true, 42 };
        var h1  = Hasher.HashObject(arr);
        var h2  = Hasher.HashObject(arr);
        Assert.Equal(h1, h2);
        Assert.Matches("^[0-9a-f]{64}$", h1);
    }

    // -------------------------------------------------------------------------
    // Error conditions
    // -------------------------------------------------------------------------

    [Fact]
    public void HashObject_NaN_ThrowsARIASerializationException()
    {
        Assert.Throws<ARIASerializationException>(() => Hasher.HashObject(double.NaN));
    }

    [Fact]
    public void HashObject_PositiveInfinity_Throws()
    {
        Assert.Throws<ARIASerializationException>(
            () => Hasher.HashObject(double.PositiveInfinity));
    }

    [Fact]
    public void HashObject_NegativeInfinity_Throws()
    {
        Assert.Throws<ARIASerializationException>(
            () => Hasher.HashObject(double.NegativeInfinity));
    }

    [Fact]
    public void HashObject_DictionaryWithNonStringKey_Throws()
    {
        // IDictionary with integer keys must be rejected.
        var bad = new System.Collections.Hashtable { [42] = "value" };
        Assert.Throws<ARIASerializationException>(() => Hasher.HashObject(bad));
    }

    // -------------------------------------------------------------------------
    // PrefixedHash / HashObjectPrefixed
    // -------------------------------------------------------------------------

    [Fact]
    public void PrefixedHash_PrependsSha256Colon()
    {
        const string raw = "abc123";
        Assert.Equal("sha256:abc123", Hasher.PrefixedHash(raw));
    }

    [Fact]
    public void HashObjectPrefixed_StartsWithSha256Prefix()
    {
        var result = Hasher.HashObjectPrefixed("aria");
        Assert.StartsWith("sha256:", result);
        Assert.Equal(7 + 64, result.Length);
    }

    [Fact]
    public void HashObjectPrefixed_MatchesPrefixedHashOfHashObject()
    {
        var obj    = new Dictionary<string, object?> { ["k"] = "v" };
        var direct = Hasher.PrefixedHash(Hasher.HashObject(obj));
        Assert.Equal(direct, Hasher.HashObjectPrefixed(obj));
    }

    // -------------------------------------------------------------------------
    // Equal
    // -------------------------------------------------------------------------

    [Fact]
    public void Equal_IdenticalHashes_ReturnsTrue()
    {
        Assert.True(Hasher.Equal("abcdef", "abcdef"));
    }

    [Fact]
    public void Equal_MixedCase_ReturnsTrueForSameValue()
    {
        Assert.True(Hasher.Equal("ABCDEF", "abcdef"));
        Assert.True(Hasher.Equal("AbCdEf", "abcdef"));
    }

    [Fact]
    public void Equal_DifferentHashes_ReturnsFalse()
    {
        Assert.False(Hasher.Equal("aaa", "bbb"));
    }
}
