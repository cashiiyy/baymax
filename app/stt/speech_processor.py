"""
BAYMAX AI – Speech Processor
==============================
Orchestrates the full speech recognition pipeline:
Microphone → AudioCapture → WhisperEngine → TranscriptionResult

Supports both one-shot utterance capture and streaming operation.

Usage:
    from app.stt.speech_processor import SpeechProcessor
    processor = SpeechProcessor()
    result = processor.listen_and_transcribe()
    print(result.text)
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import AsyncGenerator, Callable, Optional

import numpy as np

from app.stt.audio_capture import AudioCapture
from app.stt.whisper_engine import TranscriptionResult, WhisperEngine
from app.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class SpeechEvent:
    """
    Emitted by the SpeechProcessor when a transcription is available.

    Attributes:
        result:     Full transcription result.
        session_id: Optional session identifier.
        elapsed_s:  Time from capture-start to transcript (latency).
    """
    result: TranscriptionResult
    session_id: Optional[str] = None
    elapsed_s: float = 0.0


class SpeechProcessor:
    """
    High-level speech processing pipeline combining audio capture and Whisper.

    Features:
        - Synchronous single-utterance transcription (listen_and_transcribe)
        - Continuous background listening with callbacks
        - Async streaming compatible interface

    Attributes:
        audio_capture:  AudioCapture instance.
        whisper:        WhisperEngine instance.
        on_transcribed: Optional callback invoked with each SpeechEvent.
    """

    def __init__(
        self,
        audio_capture: Optional[AudioCapture] = None,
        whisper: Optional[WhisperEngine] = None,
        on_transcribed: Optional[Callable[[SpeechEvent], None]] = None,
    ) -> None:
        self.audio_capture = audio_capture or AudioCapture()
        self.whisper = whisper or WhisperEngine()
        self.on_transcribed = on_transcribed
        self._running = False
        self._listener_thread: Optional[threading.Thread] = None
        log.info("SpeechProcessor initialized")

    # ── Public API ────────────────────────────────────────────────────────────

    def listen_and_transcribe(
        self,
        silence_duration_s: float = 1.5,
        max_duration_s: float = 30.0,
        session_id: Optional[str] = None,
    ) -> Optional[TranscriptionResult]:
        """
        Capture one complete utterance and return its transcription.

        This is a blocking call. It will wait until:
        - Speech is detected
        - A silence gap >= silence_duration_s is observed
        - OR max_duration_s is reached

        Args:
            silence_duration_s: Pause duration to end utterance.
            max_duration_s:     Hard timeout for recording.
            session_id:         Optional session ID for logging.

        Returns:
            TranscriptionResult, or None if no speech detected.
        """
        log.info(
            "Listening for utterance | session={} | silence={}s | max={}s",
            session_id,
            silence_duration_s,
            max_duration_s,
        )
        t_start = time.time()

        with self.audio_capture as ac:
            audio = ac.capture_utterance(
                silence_duration_s=silence_duration_s,
                max_duration_s=max_duration_s,
            )

        if audio is None or len(audio) == 0:
            log.info("No speech detected in capture window")
            return None

        result = self.whisper.transcribe(
            audio, sample_rate=self.audio_capture.sample_rate
        )

        elapsed = time.time() - t_start
        log.info(
            "Transcription ready | latency={:.2f}s | text='{}'",
            elapsed,
            result.text[:80],
        )

        if self.on_transcribed:
            event = SpeechEvent(
                result=result, session_id=session_id, elapsed_s=elapsed
            )
            self.on_transcribed(event)

        return result

    def start_continuous_listening(
        self,
        silence_duration_s: float = 1.0,
        max_duration_s: float = 30.0,
    ) -> None:
        """
        Start a background thread that continuously listens and transcribes.
        Each completed utterance triggers the `on_transcribed` callback.

        Args:
            silence_duration_s: Pause to end each utterance.
            max_duration_s:     Hard timeout per utterance.
        """
        if self._running:
            log.warning("Continuous listening already active")
            return

        self._running = True
        self._listener_thread = threading.Thread(
            target=self._continuous_loop,
            kwargs={
                "silence_duration_s": silence_duration_s,
                "max_duration_s": max_duration_s,
            },
            daemon=True,
            name="speech-listener",
        )
        self._listener_thread.start()
        log.info("Continuous listening started")

    def stop_continuous_listening(self) -> None:
        """Stop the continuous listening thread."""
        self._running = False
        self.audio_capture.stop()
        if self._listener_thread:
            self._listener_thread.join(timeout=3.0)
        log.info("Continuous listening stopped")

    async def transcribe_audio_bytes(
        self,
        audio_bytes: bytes,
        sample_rate: int = 16000,
    ) -> TranscriptionResult:
        """
        Async transcription of raw audio bytes (for API endpoint use).

        Args:
            audio_bytes: Raw PCM int16 bytes.
            sample_rate: Sample rate of the audio.

        Returns:
            TranscriptionResult.
        """
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        audio_array = audio_array / 32768.0

        # Run in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.whisper.transcribe(audio_array, sample_rate=sample_rate),
        )
        return result

    async def stream_transcriptions(
        self,
        audio_queue: asyncio.Queue,
    ) -> AsyncGenerator[TranscriptionResult, None]:
        """
        Async generator yielding transcriptions from an audio queue.

        Args:
            audio_queue: Queue that receives audio numpy arrays.

        Yields:
            TranscriptionResult for each received audio segment.
        """
        while True:
            audio = await audio_queue.get()
            if audio is None:
                break  # Sentinel value to stop
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda a=audio: self.whisper.transcribe(a),
            )
            if not result.is_empty:
                yield result

    # ── Private Methods ───────────────────────────────────────────────────────

    def _continuous_loop(
        self,
        silence_duration_s: float,
        max_duration_s: float,
    ) -> None:
        """Background loop for continuous transcription."""
        self.audio_capture.start()
        try:
            while self._running:
                audio = self.audio_capture.capture_utterance(
                    silence_duration_s=silence_duration_s,
                    max_duration_s=max_duration_s,
                )
                if audio is None:
                    continue

                result = self.whisper.transcribe(
                    audio,
                    sample_rate=self.audio_capture.sample_rate,
                )

                if not result.is_empty and self.on_transcribed:
                    event = SpeechEvent(result=result)
                    self.on_transcribed(event)

        except Exception as exc:
            log.error("Continuous listening error: {}", exc)
        finally:
            self.audio_capture.stop()
