"""
BAYMAX AI – Avatar Controller
===============================
Provides high-level methods to control the 3D Avatar via WebSocket.
Translates LLM text and synthesized audio into avatar animation commands.

Usage:
    from app.avatar.controller import AvatarController
    controller = AvatarController(ws_manager)
    await controller.speak("Hello there", audio_bytes)
"""

from __future__ import annotations

import base64
from typing import Optional

from app.avatar.phoneme_mapper import PhonemeMapper
from app.avatar.websocket_publisher import AvatarWebSocketManager
from app.utils.logger import get_logger

log = get_logger(__name__)


class AvatarController:
    """
    High-level controller for BAYMAX avatar animations and speech.

    Commands are broadcast via the injected AvatarWebSocketManager.

    Attributes:
        ws_manager: The WebSocket manager instance handling client connections.
    """

    def __init__(self, ws_manager: AvatarWebSocketManager) -> None:
        self.ws_manager = ws_manager
        log.info("AvatarController initialized")

    async def speak(
        self,
        text: str,
        audio_bytes: Optional[bytes] = None,
        duration_s: float = 0.0,
    ) -> None:
        """
        Command the avatar to speak text, optionally providing audio.

        Args:
            text:        Text to speak.
            audio_bytes: Optional raw WAV bytes to stream to the avatar.
            duration_s:  Estimated duration of speech.
        """
        if not text.strip():
            return

        visemes = PhonemeMapper.text_to_visemes(text)

        payload = {
            "command": "speak",
            "text": text,
            "visemes": visemes,
            "duration_s": duration_s,
        }

        if audio_bytes:
            # Send audio as base64 so frontend can play it synced
            payload["audio_b64"] = base64.b64encode(audio_bytes).decode("utf-8")

        log.debug("Sending speak command | text='{}'", text[:40])
        await self.ws_manager.broadcast(payload)

    async def set_emotion(
        self,
        emotion: str,
        intensity: float = 1.0,
    ) -> None:
        """
        Command the avatar to change facial expression.

        Args:
            emotion:   Emotion label (happy, sad, surprise, etc).
            intensity: Blendshape intensity (0–1).
        """
        payload = {
            "command": "set_emotion",
            "emotion": emotion,
            "intensity": intensity,
        }
        log.debug("Sending emotion command | emotion={}", emotion)
        await self.ws_manager.broadcast(payload)

    async def play_idle(self) -> None:
        """Command the avatar to return to the neutral idle animation."""
        payload = {
            "command": "idle",
        }
        await self.ws_manager.broadcast(payload)

    async def stop_speaking(self) -> None:
        """Interrupt any current speech and return to idle."""
        payload = {
            "command": "stop_speaking",
        }
        await self.ws_manager.broadcast(payload)
