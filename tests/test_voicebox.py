import asyncio
import pytest
from unittest.mock import patch, MagicMock
from app.tts.voicebox_engine import VoiceboxEngine


def test_voicebox_health():
    async def _test():
        engine = VoiceboxEngine(base_url="http://127.0.0.1:17493")
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"message": "voicebox API", "version": "0.5.0"}
            mock_get.return_value = mock_resp

            healthy = await engine.is_healthy()
            assert healthy is True
    asyncio.run(_test())


def test_voicebox_get_profiles():
    async def _test():
        engine = VoiceboxEngine(base_url="http://127.0.0.1:17493")
        sample_profiles = [
            {"id": "2673a1ad-4529-4be8-ab9e-41998bc6ec56", "name": "Baymax", "language": "en"}
        ]
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = sample_profiles
            mock_get.return_value = mock_resp

            profiles = await engine.get_profiles()
            assert len(profiles) == 1
            assert profiles[0]["name"] == "Baymax"

            # Check resolution
            resolved_id = await engine.resolve_profile_id("Baymax")
            assert resolved_id == "2673a1ad-4529-4be8-ab9e-41998bc6ec56"
    asyncio.run(_test())


def test_voicebox_generate_speech_completed():
    async def _test():
        engine = VoiceboxEngine(base_url="http://127.0.0.1:17493", default_profile_id="2673a1ad-4529-4be8-ab9e-41998bc6ec56")
        with patch("httpx.AsyncClient.post") as mock_post, patch("httpx.AsyncClient.get") as mock_get:
            # Mock /generate
            gen_resp = MagicMock()
            gen_resp.status_code = 200
            gen_resp.json.return_value = {
                "id": "gen-123",
                "status": "completed",
                "duration": 2.5
            }
            mock_post.return_value = gen_resp

            # Mock /audio/gen-123
            audio_resp = MagicMock()
            audio_resp.status_code = 200
            audio_resp.content = b"RIFF" + b"\x00" * 100
            mock_get.return_value = audio_resp

            result = await engine.generate_speech("Hello, I am Baymax.")
            assert result["success"] is True
            assert result["engine"] == "voicebox"
            assert result["format"] == "wav"
            assert result["duration_seconds"] == 2.5
            assert len(result["audio_base64"]) > 0
    asyncio.run(_test())


def test_voicebox_generate_speech_polling():
    async def _test():
        engine = VoiceboxEngine(base_url="http://127.0.0.1:17493", default_profile_id="2673a1ad-4529-4be8-ab9e-41998bc6ec56")
        with patch("httpx.AsyncClient.post") as mock_post, patch("httpx.AsyncClient.get") as mock_get:
            # Mock /generate starting with status="generating"
            gen_resp = MagicMock()
            gen_resp.status_code = 200
            gen_resp.json.return_value = {
                "id": "gen-456",
                "status": "generating"
            }
            mock_post.return_value = gen_resp

            # Mock /history/gen-456 polling then /audio/gen-456
            hist_resp = MagicMock()
            hist_resp.status_code = 200
            hist_resp.json.return_value = {
                "id": "gen-456",
                "status": "completed",
                "duration": 3.2
            }

            audio_resp = MagicMock()
            audio_resp.status_code = 200
            audio_resp.content = b"RIFF_SAMPLE_WAV_BYTES"

            mock_get.side_effect = [hist_resp, audio_resp]

            result = await engine.generate_speech("Testing polling", poll_interval=0.01)
            assert result["success"] is True
            assert result["duration_seconds"] == 3.2
    asyncio.run(_test())
