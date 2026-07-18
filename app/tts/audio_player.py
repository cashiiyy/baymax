"""
BAYMAX AI – Audio Player
=========================
Plays synthesized audio via sounddevice.
Supports blocking and non-blocking playback,
and streaming playback for low-latency response.

Usage:
    from app.tts.audio_player import AudioPlayer
    player = AudioPlayer()
    player.play(speech_audio)
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from app.tts.xtts_engine import SpeechAudio
from app.utils.logger import get_logger

log = get_logger(__name__)


class AudioPlayer:
    """
    Audio playback engine using sounddevice.

    Supports:
        - Blocking playback (play and wait)
        - Non-blocking background playback
        - Streaming playback of chunks
        - Stop/interrupt functionality

    Attributes:
        sample_rate: Default output sample rate.
    """

    def __init__(self, sample_rate: int = 24000) -> None:
        self.sample_rate = sample_rate
        self._stop_event = threading.Event()
        self._is_playing = False
        log.info("AudioPlayer initialized | rate={}", sample_rate)

    # ── Public API ────────────────────────────────────────────────────────────

    def play(
        self,
        speech: SpeechAudio,
        blocking: bool = True,
    ) -> None:
        """
        Play a SpeechAudio object.

        Args:
            speech:   SpeechAudio to play.
            blocking: If True, wait until playback completes.
        """
        if speech.audio_data is None or len(speech.audio_data) == 0:
            return

        self._stop_event.clear()

        if blocking:
            self._play_audio(speech.audio_data, speech.sample_rate)
        else:
            thread = threading.Thread(
                target=self._play_audio,
                args=(speech.audio_data, speech.sample_rate),
                daemon=True,
                name="audio-player",
            )
            thread.start()

    def play_raw(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None,
        blocking: bool = True,
    ) -> None:
        """
        Play a raw numpy audio array.

        Args:
            audio:       Float32 audio array.
            sample_rate: Sample rate (uses default if None).
            blocking:    If True, wait until playback completes.
        """
        sr = sample_rate or self.sample_rate
        self._stop_event.clear()

        if blocking:
            self._play_audio(audio, sr)
        else:
            thread = threading.Thread(
                target=self._play_audio,
                args=(audio, sr),
                daemon=True,
            )
            thread.start()

    def play_chunks(
        self,
        speech_chunks: list[SpeechAudio],
        blocking: bool = True,
    ) -> None:
        """
        Play a sequence of SpeechAudio chunks sequentially (for streaming TTS).

        Args:
            speech_chunks: List of SpeechAudio chunks.
            blocking:      If True, wait for all chunks to complete.
        """
        self._stop_event.clear()

        def _play_all() -> None:
            for chunk in speech_chunks:
                if self._stop_event.is_set():
                    break
                if chunk.audio_data is not None and len(chunk.audio_data) > 0:
                    self._play_audio(chunk.audio_data, chunk.sample_rate)

        if blocking:
            _play_all()
        else:
            thread = threading.Thread(
                target=_play_all,
                daemon=True,
                name="audio-stream-player",
            )
            thread.start()

    def stop(self) -> None:
        """Stop any currently playing audio."""
        self._stop_event.set()
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        self._is_playing = False
        log.info("Audio playback stopped")

    @property
    def is_playing(self) -> bool:
        """Return True if audio is currently playing."""
        return self._is_playing

    # ── Private Methods ───────────────────────────────────────────────────────

    def _play_audio(self, audio: np.ndarray, sample_rate: int) -> None:
        """Internal blocking playback via sounddevice."""
        try:
            import sounddevice as sd

            self._is_playing = True
            audio_float = audio.astype(np.float32)
            # Clamp to [-1, 1]
            audio_float = np.clip(audio_float, -1.0, 1.0)

            log.debug(
                "Playing audio | dur={:.2f}s | rate={}",
                len(audio_float) / sample_rate,
                sample_rate,
            )
            sd.play(audio_float, samplerate=sample_rate, blocking=True)
        except Exception as exc:
            log.error("Audio playback error: {}", exc)
        finally:
            self._is_playing = False
