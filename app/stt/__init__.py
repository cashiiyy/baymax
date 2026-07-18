"""BAYMAX AI – STT package."""
from app.stt.audio_capture import AudioCapture, AudioChunk
from app.stt.whisper_engine import WhisperEngine, TranscriptionResult, TranscriptionSegment
from app.stt.speech_processor import SpeechProcessor, SpeechEvent

__all__ = [
    "AudioCapture", "AudioChunk", "WhisperEngine",
    "TranscriptionResult", "TranscriptionSegment",
    "SpeechProcessor", "SpeechEvent",
]
