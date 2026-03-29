"""
aria.multimodal — Multimodal input hashing for ARIA.

Provides canonical hashing for binary and structured inputs beyond plain text:
images, audio files, video frames, and embedding vectors.

Usage::

    from aria.multimodal import hash_image_bytes, hash_audio_bytes, hash_embedding
    from aria.multimodal import MultimodalRecord

    # Hash an image for audit input
    img_hash = hash_image_bytes(image_bytes)
    receipt = auditor.record(
        model_id="vision-model",
        input_data=MultimodalRecord.image(image_bytes, caption="cat photo"),
        output_data={"label": "cat", "confidence": 0.95},
        confidence=0.95,
    )

All hash strings returned by this module use the format "sha256:{64 hex chars}",
consistent with aria.core.hasher conventions.

No external dependencies — pure Python stdlib (hashlib) + aria.core.hasher.
"""

from __future__ import annotations

import hashlib
from typing import Any

from aria.core.hasher import hash_object


# ---------------------------------------------------------------------------
# Primitive hashing functions
# ---------------------------------------------------------------------------


def hash_bytes(data: bytes) -> str:
    """Return the SHA-256 hex digest of raw bytes.

    This is the low-level primitive used by all other hash_* functions in this
    module.  Callers that want to hash arbitrary binary payloads (e.g. a
    serialised model checkpoint) should call this directly.

    Args:
        data: Raw bytes to hash.

    Returns:
        str in the format ``"sha256:{64 lowercase hex chars}"``.
    """
    digest = hashlib.sha256(data).hexdigest()
    return f"sha256:{digest}"


def hash_image_bytes(data: bytes, *, format: str = "unknown") -> str:  # noqa: A002
    """Return the SHA-256 hash of raw image bytes.

    The *format* parameter is accepted for API symmetry and documentation
    purposes but does **not** affect the hash — the hash is always computed
    over the raw bytes exactly as supplied.  Callers should pass the correct
    format string (e.g. ``"jpeg"``, ``"png"``) so that MultimodalRecord
    metadata is accurate.

    Args:
        data:   Raw image bytes (JPEG, PNG, WebP, …).
        format: Human-readable format label.  Default ``"unknown"``.

    Returns:
        str in the format ``"sha256:{64 lowercase hex chars}"``.
    """
    return hash_bytes(data)


def hash_audio_bytes(data: bytes, *, sample_rate: int | None = None) -> str:
    """Return the SHA-256 hash of raw audio bytes.

    The *sample_rate* parameter is accepted for API symmetry but does **not**
    affect the hash.  The hash is computed over the raw bytes as supplied.

    Args:
        data:        Raw audio bytes (WAV, MP3, FLAC, …).
        sample_rate: Sample rate in Hz (metadata only; does not change hash).

    Returns:
        str in the format ``"sha256:{64 lowercase hex chars}"``.
    """
    return hash_bytes(data)


def hash_embedding(vector: list[float]) -> str:
    """Return the SHA-256 hash of a floating-point embedding vector.

    The vector is serialised via ``aria.core.hasher.hash_object`` so that the
    same canonical-JSON rules apply (no NaN, no Infinity, order preserved).
    Two vectors that differ only in element order will produce different hashes.

    Args:
        vector: List of float values representing the embedding.

    Returns:
        str in the format ``"sha256:{64 lowercase hex chars}"``.

    Raises:
        ARIASerializationError: if the vector contains NaN or Infinity.
    """
    return hash_object(vector)


def hash_video_frames(frames: list[bytes]) -> str:
    """Return a single SHA-256 hash representing a sequence of video frames.

    Each frame is hashed individually; the resulting 32-byte digests are
    concatenated in order and hashed once more.  This construction means:

    - An empty frame list produces a distinct, deterministic hash.
    - Reordering frames produces a different hash.
    - Changing any single frame changes the final hash.

    Args:
        frames: List of raw frame bytes, in presentation order.

    Returns:
        str in the format ``"sha256:{64 lowercase hex chars}"``.
    """
    combined = b"".join(
        hashlib.sha256(frame).digest() for frame in frames
    )
    digest = hashlib.sha256(combined).hexdigest()
    return f"sha256:{digest}"


# ---------------------------------------------------------------------------
# MultimodalRecord — convenience constructors for audit input dicts
# ---------------------------------------------------------------------------


class MultimodalRecord:
    """Factory for structured audit input dicts for multimodal models.

    Each static method returns a plain ``dict`` that can be passed directly as
    the ``input_data`` argument to ``InferenceAuditor.record()`` or
    ``ARIAQuick.record()``.  The dict is JSON-serialisable and carries only
    the hash of the raw bytes — never the raw bytes themselves — so it is safe
    to include in an AuditRecord.

    Example::

        from aria.multimodal import MultimodalRecord

        payload = MultimodalRecord.image(raw_jpeg, format="jpeg", caption="chest X-ray")
        auditor.record("radiology-v2", payload, {"findings": "no anomaly"}, confidence=0.98)
    """

    @staticmethod
    def image(
        data: bytes,
        *,
        caption: str = "",
        format: str = "unknown",  # noqa: A002
    ) -> dict[str, Any]:
        """Build an audit input dict for an image input.

        Args:
            data:    Raw image bytes.
            caption: Optional human-readable description (truncated to 200 chars).
            format:  Image format label (e.g. ``"jpeg"``, ``"png"``).

        Returns:
            dict with keys: ``modality``, ``input_hash``, ``format``,
            ``caption``, ``size_bytes``.
        """
        return {
            "modality": "image",
            "input_hash": hash_image_bytes(data, format=format),
            "format": format,
            "caption": caption[:200],
            "size_bytes": len(data),
        }

    @staticmethod
    def audio(
        data: bytes,
        *,
        sample_rate: int | None = None,
        duration_secs: float | None = None,
    ) -> dict[str, Any]:
        """Build an audit input dict for an audio input.

        Args:
            data:          Raw audio bytes.
            sample_rate:   Sample rate in Hz, if known.
            duration_secs: Audio duration in seconds, if known.

        Returns:
            dict with keys: ``modality``, ``input_hash``, ``sample_rate``,
            ``duration_secs``, ``size_bytes``.
        """
        return {
            "modality": "audio",
            "input_hash": hash_audio_bytes(data, sample_rate=sample_rate),
            "sample_rate": sample_rate,
            "duration_secs": duration_secs,
            "size_bytes": len(data),
        }

    @staticmethod
    def embedding(
        vector: list[float],
        *,
        model: str = "",
    ) -> dict[str, Any]:
        """Build an audit input dict for an embedding vector.

        Args:
            vector: Floating-point embedding values.
            model:  Name of the embedding model that produced the vector.

        Returns:
            dict with keys: ``modality``, ``input_hash``, ``dimensions``,
            ``model``.

        Raises:
            ARIASerializationError: if the vector contains NaN or Infinity.
        """
        return {
            "modality": "embedding",
            "input_hash": hash_embedding(vector),
            "dimensions": len(vector),
            "model": model,
        }

    @staticmethod
    def video(
        frames: list[bytes],
        *,
        fps: float | None = None,
    ) -> dict[str, Any]:
        """Build an audit input dict for a video represented as raw frames.

        Args:
            frames: List of raw frame bytes in presentation order.
            fps:    Frames per second, if known.

        Returns:
            dict with keys: ``modality``, ``input_hash``, ``frame_count``,
            ``fps``.
        """
        return {
            "modality": "video",
            "input_hash": hash_video_frames(frames),
            "frame_count": len(frames),
            "fps": fps,
        }
