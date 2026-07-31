import os
from typing import Optional

class XTTSSynthesizer:
    """Wrapper for Coqui XTTS v2 text-to-speech with custom cloned voice support."""

    def __init__(self, voice_reference_path: Optional[str] = None):
        self.voice_reference_path = voice_reference_path or os.getenv("BAYMAX_VOICE_SAMPLE", "cloned_voice.wav")
        self.tts = None

    def _load_tts(self):
        if self.tts is None:
            try:
                from TTS.api import TTS
                self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
            except Exception:
                self.tts = None

    def synthesize_to_file(self, text: str, output_path: str, speaker_wav: Optional[str] = None, language: str = "en") -> str:
        ref_audio = speaker_wav or self.voice_reference_path
        self._load_tts()
        
        if self.tts and os.path.exists(ref_audio):
            self.tts.tts_to_file(
                text=text,
                speaker_wav=ref_audio,
                language=language,
                file_path=output_path
            )
            return output_path
        
        # Write dummy placeholder sound/file if model reference not loaded
        with open(output_path, "wb") as f:
            f.write(b"RIFF....WAVEfmt ...data....[Stub Audio Generated]")
        return output_path
