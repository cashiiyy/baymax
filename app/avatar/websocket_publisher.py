"""
BAYMAX AI – WebSocket Publisher
=================================
Manages active WebSocket connections to Avatar clients (e.g. Blender/Unity).
Broadcasts commands (speak, emotion, idle) to all connected clients.

Usage:
    from app.avatar.websocket_publisher import AvatarWebSocketManager
    manager = AvatarWebSocketManager()
    await manager.broadcast({"command": "speak", "text": "Hello"})
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from fastapi import WebSocket

from app.utils.logger import get_logger

log = get_logger(__name__)


class AvatarWebSocketManager:
    """
    Maintains a list of active WebSocket connections to Avatar frontends.
    Allows broadcasting commands for lip-sync and expressions.
    """

    def __init__(self) -> None:
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        log.info(
            "Avatar client connected | active_clients={}",
            len(self.active_connections),
        )

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a disconnected WebSocket."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            log.info(
                "Avatar client disconnected | active_clients={}",
                len(self.active_connections),
            )

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """
        Broadcast a JSON message to all connected Avatar clients.

        Args:
            message: Dictionary to serialize to JSON.
        """
        if not self.active_connections:
            log.debug("No avatar clients connected to broadcast message")
            return

        json_msg = json.dumps(message)
        dead_connections: List[WebSocket] = []

        for connection in self.active_connections:
            try:
                await connection.send_text(json_msg)
            except Exception as exc:
                log.warning("Failed to send to avatar client: {}", exc)
                dead_connections.append(connection)

        # Cleanup dead connections
        for dead in dead_connections:
            self.disconnect(dead)

    async def ping(self) -> None:
        """Send a ping frame to check client health."""
        await self.broadcast({"command": "ping"})
