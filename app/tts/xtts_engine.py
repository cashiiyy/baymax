"""
BAYMAX AI – Coqui XTTS Voice Engine
======================================
Generates natural speech using Coqui TTS XTTS v2 with voice cloning.
Uses a reference audio clip of BAYMAX's voice to clone the voice style.

Supports:
  - Full utterance synthesis
  - Streaming chunk output (sentence-by-sentence)
  - Automatic fallback if voice reference is missing

Usage:
    from app.tts.xtts_engine import XTTSEngine
    engine = XTTSEngine()
    audio = engine.synthesize("Hello, I am BAYMAX.")
"""

from __future__ import annotations

import io
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, List, Optional

import numpy as np

from app.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class SpeechAudio:
    """
    Synthesized speech output.

    Attributes:
        audio_data:  Raw PCM float32 numpy array (mono).
        sample_rate: Audio sample rate (Hz).
        text:        Original text that was synthesized.
        duration_s:  Audio duration in seconds.
        elapsed_s:   Time to generate.
    """
    audio_data: np.ndarray
    sample_rate: int
    text: str
    duration_s: float = 0.0
    elapsed_s: float = 0.0

    def to_bytes(self, format: str = "wav") -> bytes:
        """
        Convert audio to WAV bytes suitable for streaming via HTTP.

        Args:
            format: Audio format (currently only 'wav' supported).

        Returns:
            Audio bytes.
        """
        import soundfile as sf

        buffer = io.BytesIO()
        sf.write(
            buffer,
            self.audio_data,
            self.sample_rate,
            format=format.upper(),
            subtype="PCM_16",
        )
        return buffer.getvalue()

    def __post_init__(self) -> None:
        if self.duration_s == 0.0 and len(self.audio_data) > 0:
            self.duration_s = len(self.audio_data) / self.sample_rate


class XTTSEngine:
    """
    Coqui XTTS v2 voice synthesis engine with BAYMAX voice cloning.

    Features:
        - Lazy model loading
        - Voice cloning from reference audio
        - Sentence-level streaming for low latency
        - Fallback to default TTS voice if reference unavailable

    Attributes:
        model_name:     XTTS model identifier.
        language:       Target language code.
        voice_ref_path: Path to BAYMAX reference audio WAV file.
        sample_rate:    Output audio sample rate.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        language: Optional[str] = None,
        voice_ref_path: Optional[Path] = None,
        sample_rate: Optional[int] = None,
    ) -> None:
        from config import settings

        self.model_name = model_name or settings.TTS_MODEL
        self.language = language or settings.TTS_LANGUAGE
        self.voice_ref_path = voice_ref_path or settings.BAYMAX_VOICE_REF
        self.sample_rate = sample_rate or settings.TTS_SAMPLE_RATE

        self._tts = None       # Lazy load
        self._gpt_cond = None  # Pre-computed speaker conditioning
        self._speaker_emb = None

        log.info(
            "XTTSEngine configured | model={} lang={} voice_ref={}",
            self.model_name,
            self.language,
            self.voice_ref_path,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def synthesize(
        self,
        text: str,
        speed: float = 1.0,
    ) -> SpeechAudio:
        """
        Synthesize the complete text into speech.

        Args:
            text:  Input text to speak.
            speed: Speaking speed multiplier (1.0 = normal).

        Returns:
            SpeechAudio with the complete audio data.
        """
        self._ensure_model_loaded()
        text = text.strip()
        if not text:
            return SpeechAudio(
                audio_data=np.zeros(100, dtype=np.float32),
                sample_rate=self.sample_rate,
                text="",
            )

        t_start = time.time()
        log.info("Synthesizing: '{}'", text[:80])

        try:
            outputs = self._tts.inference(
                text=text,
                language=self.language,
                gpt_cond_latent=self._gpt_cond,
                speaker_embedding=self._speaker_emb,
                speed=speed,
                enable_text_splitting=True,
            )
            audio_array = np.array(outputs["wav"], dtype=np.float32)

        except Exception as exc:
            log.error("XTTS synthesis failed: {}", exc)
            audio_array = self._generate_silence(2.0)

        elapsed = time.time() - t_start
        log.info(
            "Synthesis complete | dur={:.2f}s | latency={:.2f}s",
            len(audio_array) / self.sample_rate,
            elapsed,
        )

        return SpeechAudio(
            audio_data=audio_array,
            sample_rate=self.sample_rate,
            text=text,
            elapsed_s=elapsed,
        )

    def synthesize_streaming(
        self,
        text: str,
        speed: float = 1.0,
    ) -> Generator[SpeechAudio, None, None]:
        """
        Sentence-level streaming synthesis for lower latency.

        Splits text into sentences and synthesizes each one individually,
        yielding audio as each sentence is completed.

        Args:
            text:  Full text to speak.
            speed: Speaking speed.

        Yields:
            SpeechAudio chunks per sentence.
        """
        self._ensure_model_loaded()
        sentences = self._split_sentences(text)
        log.info(
            "Streaming synthesis | {} sentences", len(sentences)
        )

        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            chunk = self.synthesize(sentence, speed=speed)
            log.debug(
                "Sentence {}/{} synthesized | dur={:.2f}s",
                i + 1,
                len(sentences),
                chunk.duration_s,
            )
            yield chunk

    def is_loaded(self) -> bool:
        """Return True if XTTS model is loaded in memory."""
        return self._tts is not None

    def unload(self) -> None:
        """Release model from memory."""
        self._tts = None
        self._gpt_cond = None
        self._speaker_emb = None
        log.info("XTTSEngine unloaded")

    # ── Private Methods ───────────────────────────────────────────────────────

    def _ensure_model_loaded(self) -> None:
        """Load XTTS model and compute voice conditioning."""
        if self._tts is not None:
            return

        log.info("Loading XTTS v2 model: {}", self.model_name)
        from TTS.api import TTS  # type: ignore[import]
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tts = TTS(self.model_name, progress_bar=False).to(device)

        # Load voice reference for cloning
        self._load_voice_reference()
        log.info("XTTSEngine loaded | device={}", device)

    def _load_voice_reference(self) -> None:
        """Pre-compute speaker conditioning from reference audio."""
        if not Path(self.voice_ref_path).exists():
            log.warning(
                "Voice reference not found: {}. "
                "Using default speaker. "
                "Add a BAYMAX voice clip at this path for voice cloning.",
                self.voice_ref_path,
            )
            # Use built-in default speaker conditioning
            try:
                self._gpt_cond, self._speaker_emb = (
                    self._tts.synthesizer.tts_model.get_conditioning_latents(
                        audio_path=[]
                    )
                )
            except Exception:
                self._gpt_cond = None
                self._speaker_emb = None
            return

        log.info("Loading voice reference: {}", self.voice_ref_path)
        try:
            self._gpt_cond, self._speaker_emb = (
                self._tts.synthesizer.tts_model.get_conditioning_latents(
                    audio_path=[str(self.voice_ref_path)],
                    gpt_cond_len=30,
                    max_ref_length=60,
                )
            )
            log.info("Voice cloning conditioning loaded successfully")
        except Exception as exc:
            log.error("Voice conditioning failed: {}. Using default.", exc)
            self._gpt_cond = None
            self._speaker_emb = None

    def _generate_silence(self, duration_s: float) -> np.ndarray:
        """Generate silence of specified duration."""
        n_samples = int(duration_s * self.sample_rate)
        return np.zeros(n_samples, dtype=np.float32)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """
        Split text into sentences for streaming synthesis.

        Args:
            text: Input text.

        Returns:
            List of sentence strings.
        """
        # Split on sentence-ending punctuation while preserving the punctuation
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        # Filter very short fragments
        return [s.strip() for s in sentences if len(s.strip()) > 2]
