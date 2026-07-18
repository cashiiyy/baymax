"""
BAYMAX AI – Faster Whisper Speech-to-Text Engine
=================================================
Wraps faster-whisper for low-latency, GPU-accelerated transcription.
Supports single-file transcription and streaming from audio buffers.

Usage:
    from app.stt.whisper_engine import WhisperEngine
    engine = WhisperEngine()
    result = engine.transcribe(audio_array)
    print(result.text)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from app.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class TranscriptionSegment:
    """A single Whisper transcription segment."""
    text: str
    start: float      # Start time in seconds
    end: float        # End time in seconds
    avg_logprob: float = 0.0   # Log-probability score
    no_speech_prob: float = 0.0


@dataclass
class TranscriptionResult:
    """
    Full result from a Whisper transcription pass.

    Attributes:
        text:       Full concatenated transcript.
        language:   Detected or specified language code.
        segments:   Individual timestamped segments.
        duration_s: Audio duration in seconds.
        confidence: Mean confidence score across segments (0–1).
    """
    text: str
    language: str
    segments: List[TranscriptionSegment] = field(default_factory=list)
    duration_s: float = 0.0
    confidence: float = 0.0

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class WhisperEngine:
    """
    Faster Whisper speech recognition engine.

    Uses CTranslate2 backend for fast CPU/GPU inference.
    Supports streaming segment-by-segment output.

    Attributes:
        model_size:    Whisper model variant ('medium', 'large-v3', etc.)
        device:        Compute device ('cuda' or 'cpu').
        compute_type:  Quantization type ('int8', 'float16', 'float32').
        language:      Target language code or None for auto-detect.
    """

    def __init__(
        self,
        model_size: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        language: Optional[str] = None,
    ) -> None:
        from config import settings
        from app.utils.gpu import get_device

        self.model_size = model_size or settings.WHISPER_MODEL
        self.device = device or get_device()
        self.compute_type = compute_type or settings.WHISPER_COMPUTE_TYPE
        self.language = language or settings.WHISPER_LANGUAGE
        self._model = None  # Lazy load

        # Adjust compute type for CPU
        if self.device == "cpu" and self.compute_type == "float16":
            self.compute_type = "int8"
            log.info("CPU device detected – switching compute_type to int8")

        log.info(
            "WhisperEngine configured | model={} device={} compute={}",
            self.model_size,
            self.device,
            self.compute_type,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        beam_size: int = 5,
        vad_filter: bool = True,
        word_timestamps: bool = False,
    ) -> TranscriptionResult:
        """
        Transcribe a numpy audio array.

        Args:
            audio:           Float32 audio samples (mono, any sample rate).
            sample_rate:     Sample rate of the input audio.
            beam_size:       Beam search width (higher = more accurate but slower).
            vad_filter:      Apply Silero VAD filter to skip silence.
            word_timestamps: Include word-level timestamps.

        Returns:
            TranscriptionResult with full text and segments.
        """
        self._ensure_model_loaded()

        # Resample to 16kHz if needed
        audio_16k = self._resample(audio, sample_rate, 16000)

        # Ensure float32 in [-1, 1]
        audio_16k = audio_16k.astype(np.float32)
        if audio_16k.max() > 1.0:
            audio_16k = audio_16k / 32768.0

        log.debug(
            "Transcribing | duration={:.2f}s | vad={}",
            len(audio_16k) / 16000,
            vad_filter,
        )

        segments_gen, info = self._model.transcribe(
            audio_16k,
            language=self.language if self.language else None,
            beam_size=beam_size,
            vad_filter=vad_filter,
            vad_parameters={
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 200,
            },
            word_timestamps=word_timestamps,
        )

        segments: List[TranscriptionSegment] = []
        full_text_parts: List[str] = []

        for seg in segments_gen:
            text = seg.text.strip()
            if not text:
                continue
            segments.append(
                TranscriptionSegment(
                    text=text,
                    start=seg.start,
                    end=seg.end,
                    avg_logprob=getattr(seg, "avg_logprob", 0.0),
                    no_speech_prob=getattr(seg, "no_speech_prob", 0.0),
                )
            )
            full_text_parts.append(text)

        full_text = " ".join(full_text_parts)
        confidence = self._compute_confidence(segments)

        result = TranscriptionResult(
            text=full_text,
            language=info.language,
            segments=segments,
            duration_s=len(audio_16k) / 16000,
            confidence=confidence,
        )

        log.info(
            "Transcription complete | lang={} | confidence={:.2f} | text='{}'",
            result.language,
            result.confidence,
            full_text[:80],
        )
        return result

    def transcribe_file(self, file_path: str) -> TranscriptionResult:
        """
        Transcribe an audio file from disk.

        Args:
            file_path: Path to the audio file (WAV, MP3, etc.).

        Returns:
            TranscriptionResult.
        """
        import soundfile as sf

        audio, sr = sf.read(file_path, dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)  # Stereo → mono
        return self.transcribe(audio, sample_rate=sr)

    def is_loaded(self) -> bool:
        """Return True if the model is currently loaded in memory."""
        return self._model is not None

    def unload(self) -> None:
        """Free model from memory."""
        self._model = None
        log.info("WhisperEngine model unloaded")

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return
        log.info("Loading Faster Whisper model: {}", self.model_size)
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            download_root=None,  # Uses default HF cache
        )
        log.info("WhisperEngine loaded | model={}", self.model_size)

    @staticmethod
    def _resample(
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int,
    ) -> np.ndarray:
        """Resample audio from orig_sr to target_sr using scipy."""
        if orig_sr == target_sr:
            return audio
        try:
            from scipy.signal import resample_poly
            from math import gcd

            g = gcd(orig_sr, target_sr)
            up, down = target_sr // g, orig_sr // g
            return resample_poly(audio, up, down).astype(np.float32)
        except ImportError:
            log.warning("scipy not available for resampling – using raw audio")
            return audio

    @staticmethod
    def _compute_confidence(segments: List[TranscriptionSegment]) -> float:
        """Compute mean confidence from segment log probabilities."""
        if not segments:
            return 0.0
        import math

        valid = [
            s for s in segments
            if s.avg_logprob != 0.0 and not math.isnan(s.avg_logprob)
        ]
        if not valid:
            return 0.5  # Unknown
        # Convert log-prob to probability (clamp to [0, 1])
        probs = [min(1.0, max(0.0, math.exp(s.avg_logprob))) for s in valid]
        return round(sum(probs) / len(probs), 4)
