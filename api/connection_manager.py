import asyncio
import logging
from typing import Dict, Optional, Tuple, Any
import uuid

from fastapi import WebSocket
from sqlmodel import Session
from pydantic import BaseModel
from deepgram.clients.live.v1.client import LiveClient  # Assuming this is the type

# Assuming models are in the same directory
from .models import User, CallSession

logger = logging.getLogger(__name__)

# --- Client Info Model ---


class ClientInfo(BaseModel):
    websocket: WebSocket
    client_id: str
    user_id: Optional[uuid.UUID] = None
    session_id: Optional[uuid.UUID] = None
    dg_connection: Optional[LiveClient] = None

    class Config:
        arbitrary_types_allowed = True

# --- Connection Manager Class ---


class ConnectionManager:
    def __init__(self):
        # Store active client info: client_id -> ClientInfo
        self.active_connections: Dict[str, ClientInfo] = {}
        # Lock for thread-safe operations on shared dictionaries
        self._lock = asyncio.Lock()

    async def connect(self, client_info: ClientInfo):
        """Registers a new client connection."""
        async with self._lock:
            self.active_connections[client_info.client_id] = client_info

        logger.info(
            f"Client {client_info.client_id} connected. Total clients: {len(self.active_connections)}")

    async def disconnect(self, client_id: str):
        """Removes a client connection and cleans up associated data."""
        async with self._lock:
            client_info = self.active_connections.pop(client_id, None)

        if client_info:
            # Close WebSocket if still present
            try:
                await client_info.websocket.close()
                logger.info(f"WebSocket closed for client {client_id}")
            except Exception as e:
                logger.warning(
                    f"Error closing WebSocket for {client_id}: {e}")

        logger.info(
            f"Client {client_id} disconnected. Remaining clients: {len(self.active_connections)}")
        return client_info

    async def get_client_info(self, client_id: str) -> Optional[ClientInfo]:
        """Gets the ClientInfo object for a given client ID."""
        async with self._lock:
            return self.active_connections.get(client_id)

    async def send_personal_message(self, message: str, client_id: str):
        async with self._lock:
            client_info = self.active_connections.get(client_id)
        if client_info:
            try:
                await client_info.websocket.send_text(message)
            except Exception as e:
                logger.error(f"Failed to send message to {client_id}: {e}")

    async def send_error(self, client_info: ClientInfo, error_message: str):
        """Sends an error message to a specific client."""
        if client_info and client_info.websocket:
            try:
                error_data = {"type": "error", "message": error_message}
                await client_info.websocket.send_json(error_data)
            except Exception as e:
                logger.error(
                    f"Failed to send error to {client_info.client_id}: {e}")
