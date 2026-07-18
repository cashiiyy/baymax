"""
BAYMAX AI – Avatar Routes
===========================
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from app.avatar.websocket_publisher import AvatarWebSocketManager
from app.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/avatar", tags=["Avatar Integration"])

# Global WebSocket Manager for the avatar frontend
avatar_ws_manager = AvatarWebSocketManager()


class AvatarCommand(BaseModel):
    command: str
    text: str = ""
    emotion: str = "neutral"
    intensity: float = 1.0


@router.websocket("/stream")
async def avatar_websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for the 3D Avatar Frontend (Blender/Unity/Web).
    The client connects here to receive lip-sync and emotion commands.
    """
    await avatar_ws_manager.connect(websocket)
    try:
        while True:
            # Wait for any client messages (ping/pong)
            data = await websocket.receive_text()
            log.debug("Received from avatar client: {}", data)
    except WebSocketDisconnect:
        avatar_ws_manager.disconnect(websocket)
    except Exception as exc:
        log.error("Avatar websocket error: {}", exc)
        avatar_ws_manager.disconnect(websocket)


@router.post("/command")
async def send_test_command(cmd: AvatarCommand):
    """Test endpoint to manually trigger avatar commands."""
    payload = cmd.model_dump()
    await avatar_ws_manager.broadcast(payload)
    return {"status": "sent", "payload": payload}
