// aria-bsv/dotnet — Canonical JSON hashing for ARIA (BRC-121).
//
// Produces identical hashes to the Python SDK's aria.core.hasher module.
// Rules:
//   - Object keys sorted recursively (ordinal/lexicographic, same as Python sort_keys=True).
//   - No NaN or Infinity values (throws ARIASerializationException).
//   - null → JSON null.
//   - Arrays preserve insertion order.
//   - Strings are UTF-8 and JSON-escaped.
//   - Result is SHA-256 of the canonical JSON bytes, lowercase hex.

using System.Collections;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace AriaBsv.Core;

/// <summary>
/// Thrown when a value cannot be represented in ARIA canonical JSON
/// (e.g. NaN, Infinity, unsupported types).
/// </summary>
public class ARIASerializationException : Exception
{
    public ARIASerializationException(string message) : base(message) { }
}

/// <summary>
/// Static helpers for canonical JSON serialisation and SHA-256 hashing.
/// All methods are thread-safe.
/// </summary>
public static class Hasher
{
    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    /// <summary>
    /// Serialise <paramref name="v"/> to canonical JSON and return UTF-8 bytes.
    /// Keys in objects are sorted lexicographically (ordinal), matching
    /// Python's json.dumps(sort_keys=True).
    /// </summary>
    /// <exception cref="ARIASerializationException">
    /// If <paramref name="v"/> contains NaN, Infinity, or an unsupported type.
    /// </exception>
    public static byte[] CanonicalJson(object? v)
    {
        var json = SerializeValue(v);
        return Encoding.UTF8.GetBytes(json);
    }

    /// <summary>
    /// Return the SHA-256 hex digest (64 lowercase hex chars) of raw bytes.
    /// </summary>
    public static string HashBytes(byte[] data)
    {
        var hash = SHA256.HashData(data);
        return ToHex(hash);
    }

    /// <summary>
    /// Return the SHA-256 hex digest of a UTF-8 string.
    /// Equivalent to HashBytes(Encoding.UTF8.GetBytes(s)).
    /// </summary>
    public static string HashString(string s)
        => HashBytes(Encoding.UTF8.GetBytes(s));

    /// <summary>
    /// Return the SHA-256 hex digest of <paramref name="v"/>'s canonical JSON.
    /// Returns a 64-character lowercase hex string (no "sha256:" prefix).
    /// </summary>
    public static string HashObject(object? v)
    {
        var bytes = CanonicalJson(v);
        return HashBytes(bytes);
    }

    /// <summary>
    /// Prepend the "sha256:" scheme to a raw hex hash string.
    /// </summary>
    public static string PrefixedHash(string hash) => $"sha256:{hash}";

    /// <summary>
    /// Hash <paramref name="v"/> and return the result with "sha256:" prefix.
    /// </summary>
    public static string HashObjectPrefixed(object? v) => PrefixedHash(HashObject(v));

    /// <summary>
    /// Compare two hex hash strings case-insensitively.
    /// </summary>
    public static bool Equal(string a, string b)
        => string.Equals(a, b, StringComparison.OrdinalIgnoreCase);

    // -------------------------------------------------------------------------
    // Serialisation (internal)
    // -------------------------------------------------------------------------

    internal static string SerializeValue(object? value)
    {
        if (value is null)
            return "null";

        if (value is bool b)
            return b ? "true" : "false";

        if (value is string s)
            return JsonSerializer.Serialize(s);   // produces "\"...\""

        if (value is JsonElement je)
            return SerializeJsonElement(je);

        // Integer numeric types — no decimal point (matches Python int serialisation)
        if (value is byte or sbyte or short or ushort or int or uint or long or ulong)
            return Convert.ToInt64(value).ToString(CultureInfo.InvariantCulture);

        if (value is double d)
            return SerializeDouble(d);

        if (value is float f)
            return SerializeDouble((double)f);

        if (value is decimal dec)
        {
            // Round-trip representation; add .0 if whole number to match Python float style
            var decStr = dec.ToString("G", CultureInfo.InvariantCulture);
            if (!decStr.Contains('.') && !decStr.Contains('E') && !decStr.Contains('e'))
                decStr += ".0";
            return decStr;
        }

        // IDictionary must be checked before IEnumerable (Dictionary implements both)
        if (value is IDictionary dict)
            return SerializeDictionary(dict);

        // IEnumerable covers arrays, List<T>, etc. (string handled above)
        if (value is IEnumerable enumerable)
            return SerializeEnumerable(enumerable);

        // Fallback: serialise via System.Text.Json (handles POCOs, records, etc.)
        // then canonicalise the resulting JsonElement so keys are sorted.
        try
        {
            var fallbackElement = JsonSerializer.SerializeToElement(value);
            return SerializeJsonElement(fallbackElement);
        }
        catch (JsonException ex)
        {
            throw new ARIASerializationException(
                $"Cannot serialise type '{value.GetType().Name}' to ARIA canonical JSON: {ex.Message}");
        }
    }

    private static string SerializeDouble(double d)
    {
        if (double.IsNaN(d) || double.IsInfinity(d))
            throw new ARIASerializationException(
                $"Non-finite number not allowed in ARIA canonical JSON: {d}");

        // Use "R" for shortest round-trip representation.
        var s = d.ToString("R", CultureInfo.InvariantCulture);
        // Add ".0" when the value looks like an integer (matches Python json.dumps for floats).
        if (!s.Contains('.') && !s.Contains('E') && !s.Contains('e'))
            s += ".0";
        return s;
    }

    private static string SerializeDictionary(IDictionary dict)
    {
        var entries = new List<(string Key, string Value)>();
        foreach (DictionaryEntry entry in dict)
        {
            if (entry.Key is not string key)
                throw new ARIASerializationException(
                    $"All object keys must be strings, got '{entry.Key?.GetType().Name}'");
            entries.Add((key, SerializeValue(entry.Value)));
        }

        entries.Sort((x, y) => string.Compare(x.Key, y.Key, StringComparison.Ordinal));

        var pairs = entries.Select(e => $"{JsonSerializer.Serialize(e.Key)}: {e.Value}");
        return "{" + string.Join(", ", pairs) + "}";
    }

    private static string SerializeEnumerable(IEnumerable enumerable)
    {
        var items = new List<string>();
        foreach (var item in enumerable)
            items.Add(SerializeValue(item));
        return "[" + string.Join(", ", items) + "]";
    }

    private static string SerializeJsonElement(JsonElement je)
    {
        return je.ValueKind switch
        {
            JsonValueKind.Null    => "null",
            JsonValueKind.True    => "true",
            JsonValueKind.False   => "false",
            JsonValueKind.String  => JsonSerializer.Serialize(je.GetString()),
            JsonValueKind.Number  => SerializeJsonNumber(je),
            JsonValueKind.Array   => "[" + string.Join(", ", je.EnumerateArray().Select(SerializeJsonElement)) + "]",
            JsonValueKind.Object  => SerializeJsonObject(je),
            _                     => throw new ARIASerializationException(
                                        $"Unsupported JsonValueKind: {je.ValueKind}")
        };
    }

    private static string SerializeJsonObject(JsonElement je)
    {
        var pairs = je.EnumerateObject()
            .OrderBy(p => p.Name, StringComparer.Ordinal)
            .Select(p => $"{JsonSerializer.Serialize(p.Name)}: {SerializeJsonElement(p.Value)}");
        return "{" + string.Join(", ", pairs) + "}";
    }

    private static string SerializeJsonNumber(JsonElement je)
    {
        // Prefer integer representation when the value is a whole number
        if (je.TryGetInt64(out var l))
            return l.ToString(CultureInfo.InvariantCulture);

        var d = je.GetDouble();
        return SerializeDouble(d);
    }

    // -------------------------------------------------------------------------
    // Hex helpers
    // -------------------------------------------------------------------------

    internal static string ToHex(byte[] bytes)
        => Convert.ToHexString(bytes).ToLowerInvariant();

    internal static byte[] FromHex(string hex)
        => Convert.FromHexString(hex);
}
