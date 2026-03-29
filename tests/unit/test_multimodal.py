"""Tests for aria.multimodal — multimodal input hashing and MultimodalRecord."""

from __future__ import annotations

import hashlib

import pytest

from aria.core.errors import ARIASerializationError
from aria.multimodal import (
    MultimodalRecord,
    hash_audio_bytes,
    hash_bytes,
    hash_embedding,
    hash_image_bytes,
    hash_video_frames,
)


# ---------------------------------------------------------------------------
# hash_bytes
# ---------------------------------------------------------------------------

class TestHashBytes:
    def test_returns_sha256_prefix(self):
        result = hash_bytes(b"hello")
        assert result.startswith("sha256:")

    def test_hash_value_correct(self):
        expected = hashlib.sha256(b"hello").hexdigest()
        assert hash_bytes(b"hello") == f"sha256:{expected}"

    def test_empty_bytes(self):
        result = hash_bytes(b"")
        assert result.startswith("sha256:")
        assert len(result) == len("sha256:") + 64

    def test_deterministic(self):
        assert hash_bytes(b"abc") == hash_bytes(b"abc")

    def test_different_inputs_different_hashes(self):
        assert hash_bytes(b"abc") != hash_bytes(b"xyz")


# ---------------------------------------------------------------------------
# hash_image_bytes
# ---------------------------------------------------------------------------

class TestHashImageBytes:
    def test_same_as_hash_bytes(self):
        data = b"\xff\xd8\xff"  # JPEG magic bytes
        assert hash_image_bytes(data) == hash_bytes(data)

    def test_format_param_does_not_affect_hash(self):
        data = b"some image data"
        assert hash_image_bytes(data, format="jpeg") == hash_image_bytes(data, format="png")

    def test_format_default_unknown(self):
        # Just checks no error is raised with default
        hash_image_bytes(b"data")

    def test_empty_image(self):
        result = hash_image_bytes(b"")
        assert result.startswith("sha256:")


# ---------------------------------------------------------------------------
# hash_audio_bytes
# ---------------------------------------------------------------------------

class TestHashAudioBytes:
    def test_same_as_hash_bytes(self):
        data = b"RIFF" + b"\x00" * 36  # WAV-like bytes
        assert hash_audio_bytes(data) == hash_bytes(data)

    def test_sample_rate_does_not_affect_hash(self):
        data = b"audio data"
        assert hash_audio_bytes(data, sample_rate=44100) == hash_audio_bytes(data, sample_rate=22050)

    def test_empty_audio(self):
        result = hash_audio_bytes(b"")
        assert result.startswith("sha256:")


# ---------------------------------------------------------------------------
# hash_embedding
# ---------------------------------------------------------------------------

class TestHashEmbedding:
    def test_returns_sha256_prefixed_string(self):
        result = hash_embedding([0.1, 0.2, 0.3])
        assert result.startswith("sha256:")

    def test_deterministic(self):
        v = [0.1, 0.2, 0.3, 0.4]
        assert hash_embedding(v) == hash_embedding(v)

    def test_order_sensitive(self):
        assert hash_embedding([1.0, 2.0, 3.0]) != hash_embedding([3.0, 2.0, 1.0])

    def test_different_vectors_different_hashes(self):
        assert hash_embedding([0.1]) != hash_embedding([0.2])

    def test_empty_vector(self):
        result = hash_embedding([])
        assert result.startswith("sha256:")

    def test_nan_raises(self):
        with pytest.raises(ARIASerializationError):
            hash_embedding([float("nan")])

    def test_infinity_raises(self):
        with pytest.raises(ARIASerializationError):
            hash_embedding([float("inf")])


# ---------------------------------------------------------------------------
# hash_video_frames
# ---------------------------------------------------------------------------

class TestHashVideoFrames:
    def test_returns_sha256_prefix(self):
        frames = [b"frame1", b"frame2"]
        result = hash_video_frames(frames)
        assert result.startswith("sha256:")

    def test_empty_frames(self):
        result = hash_video_frames([])
        assert result.startswith("sha256:")

    def test_deterministic(self):
        frames = [b"f1", b"f2", b"f3"]
        assert hash_video_frames(frames) == hash_video_frames(frames)

    def test_order_sensitive(self):
        frames = [b"f1", b"f2"]
        assert hash_video_frames([b"f1", b"f2"]) != hash_video_frames([b"f2", b"f1"])

    def test_single_frame(self):
        result = hash_video_frames([b"only_frame"])
        assert result.startswith("sha256:")

    def test_different_frames_different_hash(self):
        assert hash_video_frames([b"a"]) != hash_video_frames([b"b"])


# ---------------------------------------------------------------------------
# MultimodalRecord.image
# ---------------------------------------------------------------------------

class TestMultimodalRecordImage:
    def test_modality_is_image(self):
        rec = MultimodalRecord.image(b"data")
        assert rec["modality"] == "image"

    def test_input_hash_present(self):
        rec = MultimodalRecord.image(b"data")
        assert rec["input_hash"].startswith("sha256:")

    def test_size_bytes_correct(self):
        data = b"x" * 100
        rec = MultimodalRecord.image(data)
        assert rec["size_bytes"] == 100

    def test_caption_truncated_at_200(self):
        long_caption = "a" * 300
        rec = MultimodalRecord.image(b"data", caption=long_caption)
        assert len(rec["caption"]) == 200

    def test_format_stored(self):
        rec = MultimodalRecord.image(b"data", format="jpeg")
        assert rec["format"] == "jpeg"

    def test_default_caption_empty(self):
        rec = MultimodalRecord.image(b"data")
        assert rec["caption"] == ""

    def test_default_format_unknown(self):
        rec = MultimodalRecord.image(b"data")
        assert rec["format"] == "unknown"


# ---------------------------------------------------------------------------
# MultimodalRecord.audio
# ---------------------------------------------------------------------------

class TestMultimodalRecordAudio:
    def test_modality_is_audio(self):
        rec = MultimodalRecord.audio(b"audio")
        assert rec["modality"] == "audio"

    def test_input_hash_present(self):
        rec = MultimodalRecord.audio(b"audio")
        assert rec["input_hash"].startswith("sha256:")

    def test_size_bytes_correct(self):
        data = b"y" * 50
        rec = MultimodalRecord.audio(data)
        assert rec["size_bytes"] == 50

    def test_sample_rate_stored(self):
        rec = MultimodalRecord.audio(b"audio", sample_rate=44100)
        assert rec["sample_rate"] == 44100

    def test_duration_stored(self):
        rec = MultimodalRecord.audio(b"audio", duration_secs=3.5)
        assert rec["duration_secs"] == 3.5

    def test_defaults_none(self):
        rec = MultimodalRecord.audio(b"audio")
        assert rec["sample_rate"] is None
        assert rec["duration_secs"] is None


# ---------------------------------------------------------------------------
# MultimodalRecord.embedding
# ---------------------------------------------------------------------------

class TestMultimodalRecordEmbedding:
    def test_modality_is_embedding(self):
        rec = MultimodalRecord.embedding([0.1, 0.2])
        assert rec["modality"] == "embedding"

    def test_input_hash_present(self):
        rec = MultimodalRecord.embedding([0.1, 0.2])
        assert rec["input_hash"].startswith("sha256:")

    def test_dimensions_correct(self):
        rec = MultimodalRecord.embedding([0.1, 0.2, 0.3])
        assert rec["dimensions"] == 3

    def test_model_stored(self):
        rec = MultimodalRecord.embedding([0.1], model="text-embedding-3")
        assert rec["model"] == "text-embedding-3"

    def test_nan_in_vector_raises(self):
        with pytest.raises(ARIASerializationError):
            MultimodalRecord.embedding([float("nan")])


# ---------------------------------------------------------------------------
# MultimodalRecord.video
# ---------------------------------------------------------------------------

class TestMultimodalRecordVideo:
    def test_modality_is_video(self):
        rec = MultimodalRecord.video([b"f1", b"f2"])
        assert rec["modality"] == "video"

    def test_input_hash_present(self):
        rec = MultimodalRecord.video([b"f1"])
        assert rec["input_hash"].startswith("sha256:")

    def test_frame_count_correct(self):
        rec = MultimodalRecord.video([b"f1", b"f2", b"f3"])
        assert rec["frame_count"] == 3

    def test_fps_stored(self):
        rec = MultimodalRecord.video([b"f1"], fps=24.0)
        assert rec["fps"] == 24.0

    def test_fps_default_none(self):
        rec = MultimodalRecord.video([b"f1"])
        assert rec["fps"] is None

    def test_empty_frames(self):
        rec = MultimodalRecord.video([])
        assert rec["frame_count"] == 0
        assert rec["input_hash"].startswith("sha256:")
