package io.aria.bsv.hasher;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Canonical JSON serialization for ARIA BRC-121.
 *
 * <p>Rules:
 * <ul>
 *   <li>Object keys sorted lexicographically</li>
 *   <li>Arrays preserve insertion order</li>
 *   <li>No whitespace</li>
 *   <li>Output is UTF-8 encoded bytes</li>
 * </ul>
 *
 * <p>Handles Java native types (Map, List, String, Number, Boolean, null)
 * directly; arbitrary objects are first converted to a JsonElement via Gson.
 */
public final class CanonicalJson {

    private static final Gson GSON = new Gson();

    private CanonicalJson() {}

    /**
     * Serialize any Java object to canonical JSON as UTF-8 bytes.
     *
     * @param value the value to serialize — may be null, Boolean, Number,
     *              String, {@code List<?>}, {@code Map<String,?>}, or any
     *              Gson-serializable object
     * @return canonical JSON bytes (UTF-8)
     * @throws CanonicalJsonException if the value cannot be serialized
     */
    public static byte[] toBytes(Object value) {
        return serialize(value).getBytes(StandardCharsets.UTF_8);
    }

    /**
     * Serialize any Java object to a canonical JSON string.
     *
     * @param value the value to serialize
     * @return canonical JSON string
     * @throws CanonicalJsonException if the value cannot be serialized
     */
    public static String toString(Object value) {
        return serialize(value);
    }

    // -------------------------------------------------------------------------
    // Core recursive serializer
    // -------------------------------------------------------------------------

    private static String serialize(Object value) {
        if (value == null) {
            return "null";
        }
        if (value instanceof Boolean) {
            return value.toString(); // "true" or "false"
        }
        if (value instanceof Number) {
            return serializeNumber((Number) value);
        }
        if (value instanceof String) {
            return serializeString((String) value);
        }
        if (value instanceof List<?>) {
            return serializeList((List<?>) value);
        }
        if (value instanceof Map<?, ?>) {
            return serializeMap((Map<?, ?>) value);
        }
        // Fallback: convert to JsonElement via Gson, then recursively canonicalize
        JsonElement el = GSON.toJsonTree(value);
        return serializeJsonElement(el);
    }

    @SuppressWarnings("unchecked")
    private static String serializeMap(Map<?, ?> map) {
        List<String> keys = new ArrayList<>(map.size());
        for (Object k : map.keySet()) {
            if (!(k instanceof String)) {
                throw new CanonicalJsonException(
                        "Map keys must be Strings, got: " + k.getClass().getName());
            }
            keys.add((String) k);
        }
        Collections.sort(keys);

        StringBuilder sb = new StringBuilder("{");
        for (int i = 0; i < keys.size(); i++) {
            if (i > 0) sb.append(',');
            String key = keys.get(i);
            sb.append(serializeString(key));
            sb.append(':');
            sb.append(serialize(((Map<String, ?>) map).get(key)));
        }
        sb.append('}');
        return sb.toString();
    }

    private static String serializeList(List<?> list) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < list.size(); i++) {
            if (i > 0) sb.append(',');
            sb.append(serialize(list.get(i)));
        }
        sb.append(']');
        return sb.toString();
    }

    private static String serializeNumber(Number n) {
        // Delegate to Gson's JsonPrimitive for consistent cross-type output.
        // Integer/Long → "1", Double/Float → "1.5", etc.
        return GSON.toJsonTree(n).toString();
    }

    private static String serializeString(String s) {
        // Gson handles proper JSON escaping (\", \\, \n, \t, etc.)
        return GSON.toJson(s);
    }

    // -------------------------------------------------------------------------
    // Gson JsonElement canonicalizer (fallback path)
    // -------------------------------------------------------------------------

    private static String serializeJsonElement(JsonElement el) {
        if (el.isJsonNull()) {
            return "null";
        }
        if (el.isJsonPrimitive()) {
            JsonPrimitive p = el.getAsJsonPrimitive();
            if (p.isBoolean()) {
                return Boolean.toString(p.getAsBoolean());
            }
            if (p.isNumber()) {
                // Use the element's own toString which Gson formats correctly
                return el.toString();
            }
            if (p.isString()) {
                return serializeString(p.getAsString());
            }
        }
        if (el.isJsonArray()) {
            JsonArray arr = el.getAsJsonArray();
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < arr.size(); i++) {
                if (i > 0) sb.append(',');
                sb.append(serializeJsonElement(arr.get(i)));
            }
            sb.append(']');
            return sb.toString();
        }
        if (el.isJsonObject()) {
            JsonObject obj = el.getAsJsonObject();
            List<String> keys = new ArrayList<>(obj.keySet());
            Collections.sort(keys);
            StringBuilder sb = new StringBuilder("{");
            for (int i = 0; i < keys.size(); i++) {
                if (i > 0) sb.append(',');
                String key = keys.get(i);
                sb.append(serializeString(key));
                sb.append(':');
                sb.append(serializeJsonElement(obj.get(key)));
            }
            sb.append('}');
            return sb.toString();
        }
        throw new CanonicalJsonException(
                "Unsupported JsonElement type: " + el.getClass().getName());
    }

    // -------------------------------------------------------------------------
    // Exception
    // -------------------------------------------------------------------------

    /**
     * Thrown when an object cannot be canonicalized.
     */
    public static class CanonicalJsonException extends RuntimeException {
        public CanonicalJsonException(String message) {
            super(message);
        }
    }
}
