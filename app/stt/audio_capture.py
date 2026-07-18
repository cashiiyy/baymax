"""
BAYMAX AI – Audio Capture
==========================
Captures audio from the microphone using sounddevice.
Implements Voice Activity Detection (VAD) to detect speech segments
and filter silence/background noise.

Usage:
    from app.stt.audio_capture import AudioCapture
    capture = AudioCapture()
    for chunk in capture.stream():
        process(chunk)
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Generator, Iterator, List, Optional

import numpy as np

from app.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class AudioChunk:
    """
    A captured audio frame.

    Attributes:
        data:         Raw audio samples as float32 numpy array.
        sample_rate:  Samples per second.
        timestamp:    Unix timestamp when chunk was captured.
        is_speech:    Whether VAD detected speech in this chunk.
    """
    data: np.ndarray
    sample_rate: int
    timestamp: float
    is_speech: bool = False


class AudioCapture:
    """
    Microphone audio capture with Voice Activity Detection.

    Uses webrtcvad for energy-based speech detection.
    Audio is buffered in a thread-safe queue, making it suitable
    for concurrent consumer threads (e.g., whisper inference).

    Attributes:
        sample_rate:    Recording sample rate (Hz). Must be 8k/16k/32k/48k for VAD.
        chunk_duration: Duration of each audio chunk in milliseconds.
        device_index:   Microphone device index (None = system default).
        vad_level:      VAD aggressiveness 0–3 (3 = most aggressive).
    """

    VALID_VAD_RATES = {8000, 16000, 32000, 48000}

    def __init__(
        self,
        sample_rate: Optional[int] = None,
        chunk_duration_ms: int = 30,  # 30ms frames required by webrtcvad
        device_index: Optional[int] = None,
        vad_level: Optional[int] = None,
    ) -> None:
        from config import settings

        self.sample_rate = sample_rate or settings.AUDIO_SAMPLE_RATE
        if self.sample_rate not in self.VALID_VAD_RATES:
            log.warning(
                "Sample rate {} not valid for VAD. Defaulting to 16000.",
                self.sample_rate,
            )
            self.sample_rate = 16000

        self.chunk_duration_ms = chunk_duration_ms
        self.chunk_size = int(self.sample_rate * chunk_duration_ms / 1000)
        self.device_index = device_index
        self.vad_level = vad_level if vad_level is not None else settings.VAD_AGGRESSIVENESS

        self._queue: queue.Queue[AudioChunk] = queue.Queue(maxsize=100)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._vad = self._init_vad()

        log.info(
            "AudioCapture initialized | rate={} chunk_ms={} vad_level={}",
            self.sample_rate,
            self.chunk_duration_ms,
            self.vad_level,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background audio capture thread."""
        if self._thread and self._thread.is_alive():
            log.warning("AudioCapture already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="audio-capture",
        )
        self._thread.start()
        log.info("AudioCapture started | device={}", self.device_index)

    def stop(self) -> None:
        """Stop the background capture thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        log.info("AudioCapture stopped")

    def stream(
        self,
        speech_only: bool = True,
        timeout: float = 0.1,
    ) -> Generator[AudioChunk, None, None]:
        """
        Generator that yields audio chunks from the capture queue.

        Args:
            speech_only: If True, only yield chunks where VAD detected speech.
            timeout:     Queue get timeout in seconds.

        Yields:
            AudioChunk instances.
        """
        if not self._thread or not self._thread.is_alive():
            self.start()

        while not self._stop_event.is_set():
            try:
                chunk = self._queue.get(timeout=timeout)
                if speech_only and not chunk.is_speech:
                    continue
                yield chunk
            except queue.Empty:
                continue

    def capture_utterance(
        self,
        silence_duration_s: float = 1.5,
        max_duration_s: float = 30.0,
    ) -> Optional[np.ndarray]:
        """
        Capture a complete speech utterance (start → silence end).

        Args:
            silence_duration_s: Seconds of silence to consider utterance complete.
            max_duration_s:     Maximum recording duration.

        Returns:
            Concatenated audio array or None if no speech detected.
        """
        if not self._thread or not self._thread.is_alive():
            self.start()

        frames: List[np.ndarray] = []
        silence_frames = 0
        silence_frame_limit = int(
            silence_duration_s * 1000 / self.chunk_duration_ms
        )
        max_frames = int(max_duration_s * 1000 / self.chunk_duration_ms)
        speech_started = False

        log.debug("Waiting for speech utterance...")

        for chunk in self.stream(speech_only=False):
            frames.append(chunk.data)

            if chunk.is_speech:
                speech_started = True
                silence_frames = 0
            elif speech_started:
                silence_frames += 1
                if silence_frames >= silence_frame_limit:
                    break

            if len(frames) >= max_frames:
                log.warning("Max utterance duration reached")
                break

        if not speech_started or not frames:
            return None

        audio = np.concatenate(frames, axis=0)
        duration_s = len(audio) / self.sample_rate
        log.info("Utterance captured | duration={:.2f}s", duration_s)
        return audio

    # ── Private Methods ───────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        """Background thread: continuously capture audio from microphone."""
        try:
            import sounddevice as sd

            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=self.chunk_size,
                device=self.device_index,
            ) as stream:
                log.info("Microphone stream opened")
                while not self._stop_event.is_set():
                    raw, overflowed = stream.read(self.chunk_size)
                    if overflowed:
                        log.debug("Audio buffer overflow – frames dropped")
                    audio_int16 = raw.flatten()
                    is_speech = self._detect_speech(audio_int16)

                    chunk = AudioChunk(
                        data=audio_int16.astype(np.float32) / 32768.0,
                        sample_rate=self.sample_rate,
                        timestamp=time.time(),
                        is_speech=is_speech,
                    )
                    if not self._queue.full():
                        self._queue.put_nowait(chunk)

        except Exception as exc:
            log.error("Audio capture error: {}", exc)

    def _detect_speech(self, audio_int16: np.ndarray) -> bool:
        """
        Run VAD on a 16-bit PCM frame.

        Args:
            audio_int16: int16 audio array.

        Returns:
            True if speech detected.
        """
        if self._vad is None:
            # Fallback: energy threshold
            rms = np.sqrt(np.mean(audio_int16.astype(np.float32) ** 2))
            return rms > 300

        try:
            frame_bytes = audio_int16.tobytes()
            return self._vad.is_speech(frame_bytes, self.sample_rate)
        except Exception:
            return False

    def _init_vad(self):
        """Initialize webrtcvad VAD. Returns None if not available."""
        try:
            import webrtcvad
            vad = webrtcvad.Vad(self.vad_level)
            log.debug("WebRTC VAD initialized | level={}", self.vad_level)
            return vad
        except ImportError:
            log.warning("webrtcvad not available – using energy-based VAD fallback")
            return None

    def __enter__(self) -> "AudioCapture":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()
