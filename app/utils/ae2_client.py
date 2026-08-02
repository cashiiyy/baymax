"""
BAYMAX AI — AI Engine 2 HTTP Client
=====================================
Async HTTP client that AI Engine 1 uses to talk to AI Engine 2's REST API.

All public methods:
  - Return ``None`` (or a sensible default) if AE2 is unreachable
  - Forward the ``X-Correlation-ID`` header for distributed tracing
  - Never raise exceptions that would crash AI Engine 1's pipeline

Usage:
    from app.utils.ae2_client import ae2_client

    result = await ae2_client.tts("Hello, I am BAYMAX.")
    transcript = await ae2_client.transcribe(audio_bytes, filename="voice.webm")
    ocr_data   = await ae2_client.ocr(file_bytes, filename="prescription.jpg")
"""

from __future__ import annotations

import uuid
import asyncio
from typing import Optional, Any

import httpx

from config import settings
from app.utils.logger import get_logger

log = get_logger(__name__)

# ── Timeouts ──────────────────────────────────────────────────────────────────
_CONNECT_TIMEOUT = 5.0   # give up connecting quickly so UI stays responsive
_READ_TIMEOUT    = settings.AE2_TIMEOUT


class AE2Client:
    """
    Async HTTP client for AI Engine 2 (Multimodal Perception Engine).

    Singleton — import ``ae2_client`` directly; do not instantiate this class.
    """

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _get_client(self) -> httpx.AsyncClient:
        """Return (and lazily create) the shared AsyncClient."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(connect=_CONNECT_TIMEOUT, read=_READ_TIMEOUT,
                                      write=_READ_TIMEOUT, pool=5.0),
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _correlation_id(self) -> str:
        return str(uuid.uuid4())

    def _headers(self, extra: dict | None = None) -> dict:
        h = {"X-Correlation-ID": self._correlation_id()}
        if extra:
            h.update(extra)
        return h

    async def _post_json(self, path: str, **kwargs) -> Optional[dict]:
        """POST with JSON body — returns parsed response or None."""
        try:
            r = await self._get_client().post(path, headers=self._headers(), **kwargs)
            r.raise_for_status()
            return r.json()
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            log.warning("AE2 unreachable [{}]: {}", path, exc)
            return None
        except httpx.HTTPStatusError as exc:
            log.error("AE2 HTTP error [{}] {}: {}", path, exc.response.status_code, exc.response.text)
            return None
        except Exception as exc:
            log.exception("AE2 unexpected error [{}]: {}", path, exc)
            return None

    async def _post_multipart(self, path: str, files: dict,
                              data: dict | None = None) -> Optional[dict]:
        """POST multipart/form-data — returns parsed response or None."""
        try:
            r = await self._get_client().post(
                path,
                headers=self._headers(),
                files=files,
                data=data or {},
            )
            r.raise_for_status()
            return r.json()
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            log.warning("AE2 unreachable [{}]: {}", path, exc)
            return None
        except httpx.HTTPStatusError as exc:
            log.error("AE2 HTTP error [{}] {}: {}", path, exc.response.status_code, exc.response.text)
            return None
        except Exception as exc:
            log.exception("AE2 unexpected error [{}]: {}", path, exc)
            return None

    async def _get(self, path: str) -> Optional[dict]:
        """GET request — returns parsed response or None."""
        try:
            r = await self._get_client().get(path, headers=self._headers())
            r.raise_for_status()
            return r.json()
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            log.warning("AE2 unreachable [{}]: {}", path, exc)
            return None
        except Exception as exc:
            log.exception("AE2 error [{}]: {}", path, exc)
            return None

    # ── Public API ────────────────────────────────────────────────────────────

    async def health(self) -> bool:
        """
        Ping AI Engine 2's /health endpoint.
        Returns True if AE2 is online and healthy.
        """
        data = await self._get("/health")
        if data is None:
            return False
        status = data.get("status", "")
        return status in ("ok", "healthy")

    async def status(self) -> Optional[dict]:
        """Return the full /status response from AE2 (component readiness)."""
        return await self._get("/status")

    async def tts(
        self,
        text: str,
        voice: str = "default",
        language: str = "en",
        stream: bool = False,
    ) -> Optional[dict]:
        """
        Text-to-Speech synthesis via AE2.

        Returns a dict with at least:
          - ``audio_base64`` (str) — base64-encoded WAV
          - ``duration_seconds`` (float)
          - ``voice_used`` (str)

        Returns None if AE2 is unavailable.
        """
        return await self._post_json(
            "/tts",
            json={"text": text, "voice": voice, "language": language,
                  "stream": stream, "format": "wav"},
        )

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.webm",
        language: str = "",
        vad_filter: bool = True,
    ) -> Optional[dict]:
        """
        Speech-to-Text transcription via AE2 (Faster-Whisper).

        Returns a dict with at least:
          - ``text`` (str)
          - ``language`` (str)
          - ``word_count`` (int)
          - ``has_speech`` (bool)

        Returns None if AE2 is unavailable.
        """
        return await self._post_multipart(
            "/transcribe",
            files={"file": (filename, audio_bytes, "audio/webm")},
            data={"language": language, "vad_filter": str(vad_filter).lower(),
                  "word_timestamps": "true"},
        )

    async def ocr(
        self,
        file_bytes: bytes,
        filename: str = "document.jpg",
        lang: str = "eng",
    ) -> Optional[dict]:
        """
        OCR extraction via AE2 (Tesseract + PyMuPDF).

        Returns a dict with at least:
          - ``raw_text`` (str)
          - ``document_type`` (str)
          - ``extracted_fields`` (list)
          - ``overall_confidence`` (float)

        Returns None if AE2 is unavailable.
        """
        content_type = "application/pdf" if filename.lower().endswith(".pdf") else "image/jpeg"
        return await self._post_multipart(
            "/ocr",
            files={"file": (filename, file_bytes, content_type)},
            data={"lang": lang, "psm": "3", "oem": "3"},
        )

    async def vision(self, image_bytes: bytes, filename: str = "frame.jpg") -> Optional[dict]:
        """
        Vision analysis via AE2 (face detection, emotion context, person detection).

        Returns a dict with at least:
          - ``faces`` (list)
          - ``emotions`` (list[{label, confidence}])
          - ``person_detected`` (bool)
          - ``observations`` (list[str])

        Returns None if AE2 is unavailable.
        """
        return await self._post_multipart(
            "/vision",
            files={"file": (filename, image_bytes, "image/jpeg")},
            data={
                "face_detection": "true",
                "emotion_context": "true",
                "roi_detection": "true",
                "mediapipe": "true",
                "person_detection": "true",
            },
        )

    async def document(self, file_bytes: bytes, filename: str = "document.pdf") -> Optional[dict]:
        """
        Structured document parsing via AE2 (PDF/image/text).

        Returns a dict with parsed sections, tables, metadata, and fields.
        Returns None if AE2 is unavailable.
        """
        content_type = "application/pdf" if filename.lower().endswith(".pdf") else "image/jpeg"
        return await self._post_multipart(
            "/document",
            files={"file": (filename, file_bytes, content_type)},
        )

    async def analyse(
        self,
        file_bytes: bytes,
        filename: str,
        tasks: list[str] | None = None,
    ) -> Optional[dict]:
        """
        Combined parallel analysis via AE2's /analyse endpoint.

        ``tasks`` is a list such as ``["ocr", "vision"]`` or ``["ocr", "transcribe"]``.
        Defaults to ``["ocr"]``.

        Returns combined results from all tasks, or None if AE2 is unavailable.
        """
        tasks = tasks or ["ocr"]
        content_type = "application/pdf" if filename.lower().endswith(".pdf") else "image/jpeg"
        return await self._post_multipart(
            "/analyse",
            files={"file": (filename, file_bytes, content_type)},
            data={"tasks": ",".join(tasks)},
        )

    async def get_voices(self) -> list[dict]:
        """
        List available TTS voice profiles on AE2.
        Returns an empty list if AE2 is unavailable.
        """
        data = await self._get("/tts/voices")
        if data is None:
            return []
        return data.get("voices", [])


# ── Singleton ─────────────────────────────────────────────────────────────────
ae2_client = AE2Client(base_url=settings.AE2_BASE_URL)
