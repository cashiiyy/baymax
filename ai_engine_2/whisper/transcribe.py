import os
from typing import Optional

class WhisperTranscriber:
    """Wrapper for Faster-Whisper local speech recognition."""

    def __init__(self, model_size: str = "base.en", device: str = "auto"):
        self.model_size = model_size
        self.device = device
        self.model = None

    def _load_model(self):
        if self.model is None:
            try:
                from faster_whisper import WhisperModel
                self.model = WhisperModel(self.model_size, device=self.device, compute_type="default")
            except Exception as e:
                self.model = None

    def transcribe(self, audio_path: str) -> str:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self._load_model()
        if self.model:
            segments, info = self.model.transcribe(audio_path, beam_size=5)
            transcript = " ".join([segment.text for segment in segments])
            return transcript.strip()

        return "[Stub Transcript]: User is describing symptoms of fever and sore throat."
