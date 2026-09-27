"""
BAYMAX AI – Voicebox TTS Engine
=================================
Connects to the Voicebox desktop app server (default: http://127.0.0.1:17493)
to synthesize speech using the custom "BAYMAX" voice profile.

Usage:
    from app.tts.voicebox_engine import VoiceboxEngine
    engine = VoiceboxEngine()
    result = await engine.generate_speech("Hello, I am Baymax.")
    # result['audio_base64'], result['duration_seconds']
"""

from __future__ import annotations

import asyncio
import base64
import os
import time
from typing import Optional, Dict, Any, List

import httpx

from config import settings
from app.utils.logger import get_logger

log = get_logger(__name__)


class VoiceboxEngine:
    """
    Client for the Voicebox local TTS REST API.
    Synthesizes speech using voice profiles (specifically 'Baymax').
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        default_profile_name: Optional[str] = None,
        default_profile_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.base_url = (base_url or getattr(settings, "VOICEBOX_BASE_URL", "http://127.0.0.1:17493")).rstrip("/")
        self.default_profile_name = default_profile_name or getattr(settings, "VOICEBOX_VOICE_PROFILE", "Baymax")
        self._cached_profile_id = default_profile_id or getattr(settings, "VOICEBOX_PROFILE_ID", "2673a1ad-4529-4be8-ab9e-41998bc6ec56")
        self.timeout = timeout or float(getattr(settings, "VOICEBOX_TIMEOUT", 30.0))
        self._profile_cache: Dict[str, str] = {}
        if self._cached_profile_id:
            self._profile_cache[self.default_profile_name.lower()] = self._cached_profile_id

        log.info(
            "VoiceboxEngine initialized | base_url={} | profile={} | id={}",
            self.base_url,
            self.default_profile_name,
            self._cached_profile_id,
        )

    # ── Health & Status ───────────────────────────────────────────────────────

    async def is_healthy(self) -> bool:
        """Check if the Voicebox API server is reachable and responsive."""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(connect=2.0, read=3.0, write=3.0, pool=2.0)) as client:
                resp = await client.get(f"{self.base_url}/health")
                return resp.status_code == 200
        except Exception:
            return False

    async def check_health(self) -> Dict[str, Any]:
        """Comprehensive health check for Voice Box and Baymax profile."""
        start_t = time.time()
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(connect=2.0, read=4.0, write=4.0, pool=2.0)) as client:
                resp = await client.get(f"{self.base_url}/health")
                latency_ms = round((time.time() - start_t) * 1000, 2)
                if resp.status_code == 200:
                    profiles = await self.get_profiles()
                    baymax_found = any(
                        p.get("name", "").lower() == self.default_profile_name.lower()
                        for p in profiles
                    )
                    return {
                        "status": "online",
                        "base_url": self.base_url,
                        "baymax_profile_available": baymax_found,
                        "profile_count": len(profiles),
                        "latency_ms": latency_ms,
                        "error": None,
                    }
                return {
                    "status": "degraded",
                    "base_url": self.base_url,
                    "http_status": resp.status_code,
                    "error": resp.text[:200],
                }
        except Exception as exc:
            return {
                "status": "offline",
                "base_url": self.base_url,
                "baymax_profile_available": False,
                "error": str(exc),
            }

    async def get_profiles(self) -> List[Dict[str, Any]]:
        """Retrieve all available voice profiles from Voicebox."""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(connect=2.0, read=4.0, write=4.0, pool=2.0)) as client:
                resp = await client.get(f"{self.base_url}/profiles")
                if resp.status_code == 200:
                    profiles = resp.json()
                    for p in profiles:
                        p_name = p.get("name", "")
                        p_id = p.get("id", "")
                        if p_name and p_id:
                            self._profile_cache[p_name.lower()] = p_id
                    return profiles
        except Exception as exc:
            log.warning("Failed to fetch Voicebox profiles: {}", exc)
        return []

    async def resolve_profile_id(self, profile_name_or_id: Optional[str] = None) -> Optional[str]:
        """
        Resolve a profile name (e.g. 'Baymax') or ID to a valid profile_id.
        """
        target = (profile_name_or_id or self.default_profile_name).strip()
        target_lower = target.lower()

        # If already cached by name
        if target_lower in self._profile_cache:
            return self._profile_cache[target_lower]

        # Check if target is directly a profile UUID
        if len(target) == 36 and target.count("-") == 4:
            return target

        # Fetch profiles from Voicebox
        profiles = await self.get_profiles()
        for p in profiles:
            if p.get("name", "").lower() == target_lower:
                pid = p.get("id")
                self._profile_cache[target_lower] = pid
                return pid

        # Fallback to cached default
        return self._cached_profile_id

    # ── Speech Generation ─────────────────────────────────────────────────────

    async def generate_speech(
        self,
        text: str,
        profile_name_or_id: Optional[str] = None,
        language: str = "en",
        poll_interval: float = 0.4,
    ) -> Dict[str, Any]:
        """
        Synthesize speech for given text using Voicebox TTS.

        Args:
            text: Text to synthesize.
            profile_name_or_id: Voice profile name ('Baymax') or ID.
            language: Language code ('en').
            poll_interval: Polling frequency in seconds.

        Returns:
            Dict containing:
                - success: bool
                - audio_base64: base64-encoded WAV audio string
                - duration_seconds: float
                - voice_used: str
                - engine: 'voicebox'
                - format: 'wav'
        """
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Empty text provided for TTS synthesis")

        profile_id = await self.resolve_profile_id(profile_name_or_id)
        if not profile_id:
            raise RuntimeError(f"Could not resolve Voicebox profile for '{profile_name_or_id or self.default_profile_name}'")

        log.info("Requesting Voicebox TTS | profile_id={} | text_len={}", profile_id, len(clean_text))

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # 1. Initiate Generation Request
            payload = {
                "profile_id": profile_id,
                "text": clean_text,
                "language": language or "en",
            }
            try:
                gen_resp = await client.post(f"{self.base_url}/generate", json=payload)
            except httpx.ConnectError as conn_err:
                raise RuntimeError(f"Voicebox server is offline at {self.base_url}: {conn_err}") from conn_err

            if gen_resp.status_code not in (200, 201):
                raise RuntimeError(f"Voicebox generation request failed ({gen_resp.status_code}): {gen_resp.text}")

            gen_data = gen_resp.json()
            generation_id = gen_data.get("id")
            if not generation_id:
                raise RuntimeError(f"Voicebox response missing generation id: {gen_data}")

            # If generation completed synchronously
            if gen_data.get("status") == "completed":
                audio_bytes = await self._fetch_audio(client, generation_id)
                duration = float(gen_data.get("duration") or 0.0)
                return self._format_response(audio_bytes, duration, profile_name_or_id)

            # 2. Poll for Completion via /history/{generation_id}
            start_time = time.time()
            max_wait = self.timeout
            duration_s = 0.0

            while time.time() - start_time < max_wait:
                await asyncio.sleep(poll_interval)
                try:
                    hist_resp = await client.get(f"{self.base_url}/history/{generation_id}")
                    if hist_resp.status_code == 200:
                        hist_data = hist_resp.json()
                        status = hist_data.get("status")
                        if status == "completed":
                            duration_s = float(hist_data.get("duration") or 0.0)
                            break
                        elif status == "failed":
                            err_msg = hist_data.get("error") or "Unknown Voicebox synthesis error"
                            raise RuntimeError(f"Voicebox generation failed: {err_msg}")
                except httpx.RequestError as req_err:
                    log.warning("Voicebox poll warning: {}", req_err)

            else:
                raise TimeoutError(f"Voicebox TTS generation timed out after {max_wait}s (id={generation_id})")

            # 3. Retrieve Audio WAV Bytes
            audio_bytes = await self._fetch_audio(client, generation_id)
            return self._format_response(audio_bytes, duration_s, profile_name_or_id)

    async def _fetch_audio(self, client: httpx.AsyncClient, generation_id: str) -> bytes:
        """Fetch the synthesized audio WAV bytes for a generation."""
        audio_resp = await client.get(f"{self.base_url}/audio/{generation_id}")
        if audio_resp.status_code != 200:
            raise RuntimeError(
                f"Failed to fetch Voicebox audio for {generation_id} ({audio_resp.status_code}): {audio_resp.text}"
            )
        return audio_resp.content

    def _format_response(
        self,
        audio_bytes: bytes,
        duration: float,
        profile_name_or_id: Optional[str],
    ) -> Dict[str, Any]:
        """Format audio bytes into base64 payload response."""
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        if duration <= 0.0:
            # Estimate from WAV header/length if duration was 0
            duration = max(0.5, round(len(audio_bytes) / (24000 * 2), 2))

        profile_label = profile_name_or_id or self.default_profile_name
        return {
            "success": True,
            "audio_base64": audio_b64,
            "duration_seconds": duration,
            "voice_used": f"Voicebox-{profile_label}",
            "engine": "voicebox",
            "format": "wav",
        }


# Singleton instance
voicebox_engine = VoiceboxEngine()
