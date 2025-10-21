"""
Utilities for the IndexTTS2 FastAPI server.
Shared logic for request/response handling and validation.
"""

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ConfigDict,
)
from typing import Optional, List, Literal
from enum import Enum
import base64
import os
import tempfile
import soundfile as sf
import numpy as np


class EmotionMode(str, Enum):
    """Emotion control modes."""
    SAME_AS_SPEAKER = "same_as_speaker"  # Mode 0: Use speaker audio
    REFERENCE_AUDIO = "reference_audio"  # Mode 1: Separate emotion audio
    EMOTION_VECTOR = "emotion_vector"    # Mode 2: Manual 8D vector
    TEXT_DESCRIPTION = "text_description" # Mode 3: Text-based emotion


class TTSRequest(BaseModel):
    """Comprehensive TTS request model covering all webui features."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    # Required fields
    text: str = Field(..., description="Text to synthesize", min_length=1)
    speaker_audio: str = Field(
        ...,
        description="Base64-encoded speaker reference audio (WAV/MP3/FLAC)",
    )

    # Emotion control
    emotion_mode: EmotionMode = Field(
        default=EmotionMode.SAME_AS_SPEAKER,
        description="Emotion control mode"
    )
    emotion_audio: Optional[str] = Field(
        default=None,
        description="Base64-encoded emotion reference audio (required if emotion_mode=reference_audio)"
    )
    emotion_vector: Optional[List[float]] = Field(
        default=None,
        description="8D emotion vector: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm] (required if emotion_mode=emotion_vector)"
    )
    emotion_text: Optional[str] = Field(
        default=None,
        description="Text description of emotion"
    )
    emotion_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Emotion strength/alpha (0.0-1.0)"
    )
    emotion_random: bool = Field(
        default=False,
        description="Apply random noise to emotion vector"
    )

    # Generation parameters (GPT)
    do_sample: bool = Field(
        default=True,
        description="Enable sampling (vs greedy decoding)"
    )
    temperature: float = Field(
        default=0.8,
        ge=0.1,
        le=2.0,
        description="Sampling temperature (0.1-2.0)"
    )
    top_p: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling threshold (0.0-1.0)"
    )
    top_k: int = Field(
        default=30,
        ge=0,
        le=100,
        description="Top-K sampling parameter (0-100)"
    )
    num_beams: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Beam search width (1-10)"
    )
    repetition_penalty: float = Field(
        default=10.0,
        ge=0.1,
        le=20.0,
        description="Token repetition penalty (0.1-20.0)"
    )
    length_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Length penalty for beam search (-2.0 to 2.0)"
    )
    max_mel_tokens: int = Field(
        default=1500,
        ge=50,
        le=3000,
        description="Maximum mel-spectrogram tokens to generate"
    )

    # Segmentation parameters
    max_text_tokens_per_segment: int = Field(
        default=120,
        ge=20,
        le=500,
        description="Maximum text tokens per generation segment"
    )
    interval_silence: int = Field(
        default=200,
        ge=0,
        le=2000,
        description="Silence duration between segments in milliseconds (0-2000ms)"
    )

    # Output format
    output_audio_format: Literal["wav", "mp3", "flac"] = Field(
        default="wav",
        description="Output audio format"
    )

    @field_validator("emotion_vector")
    @classmethod
    def validate_emotion_vector(cls, v):
        """Validate emotion vector has exactly 8 dimensions."""
        if v is not None and len(v) != 8:
            raise ValueError("emotion_vector must have exactly 8 values")
        return v

    @field_validator("emotion_audio")
    @classmethod
    def validate_emotion_audio(cls, v, info):
        """Validate emotion_audio is provided when required."""
        if info.data.get("emotion_mode") == EmotionMode.REFERENCE_AUDIO and v is None:
            raise ValueError("emotion_audio is required when emotion_mode=reference_audio")
        return v

    @field_validator("emotion_text")
    @classmethod
    def validate_emotion_text(cls, v, info):
        """Validate emotion_text is provided when required."""
        if info.data.get("emotion_mode") == EmotionMode.TEXT_DESCRIPTION and not v:
            raise ValueError("emotion_text is required when emotion_mode=text_description")
        return v

    @model_validator(mode="before")
    @classmethod
    def _inject_legacy_prompt_audio(cls, data):
        """Support legacy `prompt_audio` field name."""
        if isinstance(data, dict) and "speaker_audio" not in data and "prompt_audio" in data:
            data = dict(data)
            data["speaker_audio"] = data.pop("prompt_audio")
        return data


class TTSResponse(BaseModel):
    """TTS response model."""
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    audio: str = Field(
        ...,
        description="Base64-encoded audio data",
    )
    format: Literal["wav", "mp3", "flac"] = Field(
        default="wav",
        description="Audio format that was generated",
    )
    duration: float = Field(..., description="Audio duration in seconds")
    inference_time_seconds: float = Field(
        default=0.0,
        description="Time taken for inference in seconds",
    )
    sample_rate: int = Field(default=22050, description="Sample rate in Hz")
    num_segments: int = Field(..., description="Number of text segments processed")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")

    @model_validator(mode="before")
    @classmethod
    def _inject_legacy_response_fields(cls, data):
        """Support legacy response field names."""
        if not isinstance(data, dict):
            return data

        updated = False
        mutable = dict(data)

        if "audio" not in mutable and "audio_base64" in mutable:
            mutable["audio"] = mutable.pop("audio_base64")
            updated = True
        if "duration" not in mutable and "audio_duration_seconds" in mutable:
            mutable["duration"] = mutable.pop("audio_duration_seconds")
            updated = True

        return mutable if updated else data


class SegmentPreviewRequest(BaseModel):
    """Request for previewing text segmentation."""
    text: str = Field(..., description="Text to segment", min_length=1)
    max_text_tokens_per_segment: int = Field(
        default=120,
        ge=20,
        le=500,
        description="Maximum text tokens per segment"
    )


class SegmentPreviewResponse(BaseModel):
    """Response for text segmentation preview."""
    segments: List[str] = Field(..., description="List of text segments")
    num_segments: int = Field(..., description="Total number of segments")
    tokens_per_segment: List[int] = Field(..., description="Token count for each segment")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether TTS model is loaded")
    device: str = Field(..., description="Device being used (cuda/cpu/mps/xpu)")
    version: str = Field(..., description="IndexTTS version")


def decode_audio_base64(audio_b64: str, output_path: Optional[str] = None) -> str:
    """
    Decode base64-encoded audio and save to a temporary file.

    Args:
        audio_b64: Base64-encoded audio data
        output_path: Optional specific output path, otherwise uses temp file

    Returns:
        Path to the saved audio file
    """
    audio_bytes = base64.b64decode(audio_b64)

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

    with open(output_path, "wb") as f:
        f.write(audio_bytes)

    return output_path


def encode_audio_base64(audio_path: str, format: str = "wav") -> tuple[str, float, int]:
    """
    Encode audio file to base64.

    Args:
        audio_path: Path to audio file
        format: Output format (wav/mp3/flac)

    Returns:
        Tuple of (base64-encoded audio, duration in seconds, sample rate)

    Note:
        For MP3 format, the file should already be encoded at high quality (320kbps)
        by convert_audio_file() before calling this function.
    """
    if format not in {"wav", "mp3", "flac"}:
        raise ValueError(f"Unsupported audio format: {format}")

    if format in {"wav", "flac"}:
        with sf.SoundFile(audio_path) as f:
            sample_rate = f.samplerate
            frames = f.frames
        duration = frames / sample_rate if sample_rate else 0.0
    elif format == "mp3":
        try:
            from mutagen.mp3 import MP3  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "MP3 responses require the 'mutagen' package. Install the API extras with "
                "`uv sync --extra api`."
            ) from exc

        audio_info = MP3(audio_path)
        sample_rate = int(audio_info.info.sample_rate)
        duration = float(audio_info.info.length)

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    return audio_b64, duration, sample_rate


def convert_audio_file(source_path: str, target_format: str) -> str:
    """
    Convert an audio file to the desired format.

    Args:
        source_path: Path to the source audio file (expected to be WAV)
        target_format: Desired output format ("wav", "mp3", "flac")

    Returns:
        Path to the converted audio file. If the target format is WAV, returns the source path.
    """
    if target_format == "wav":
        return source_path

    fd, target_path = tempfile.mkstemp(suffix=f".{target_format}")
    os.close(fd)

    if target_format == "flac":
        audio_data, sample_rate = sf.read(source_path)
        sf.write(target_path, audio_data, sample_rate, format="FLAC")
    elif target_format == "mp3":
        try:
            from pydub import AudioSegment  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "MP3 responses require the 'pydub' package and an FFmpeg installation. "
                "Install the API extras with `uv sync --extra api` and ensure FFmpeg is available."
            ) from exc
        segment = AudioSegment.from_file(source_path)
        # Use 320kbps bitrate for high-quality MP3 output
        segment.export(target_path, format="mp3", bitrate="320k")
    else:
        raise ValueError(f"Unsupported target format: {target_format}")

    return target_path


def normalize_emotion_vector(vec: List[float], apply_bias: bool = True) -> np.ndarray:
    """
    Normalize emotion vector to unit length.

    Args:
        vec: 8D emotion vector
        apply_bias: Whether to apply bias before normalization

    Returns:
        Normalized emotion vector as numpy array
    """
    vec = np.array(vec, dtype=np.float32)

    if apply_bias:
        # Apply bias similar to webui_utils.py
        vec = vec + 0.1

    # Normalize to unit length
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec
