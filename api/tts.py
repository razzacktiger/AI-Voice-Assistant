import os
import asyncio
import logging
from typing import AsyncGenerator

from fastapi import WebSocket
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    SpeakOptions,
)
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global Deepgram client (initialized once)
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    logger.warning(
        "DEEPGRAM_API_KEY not set. TTS functionality will be disabled.")
    # Optionally raise an error if TTS is critical
    # raise ValueError("DEEPGRAM_API_KEY is required for TTS")

deepgram_client = None
try:
    if DEEPGRAM_API_KEY:
        deepgram_client = DeepgramClient(DEEPGRAM_API_KEY)
except Exception as e:
    logger.error(f"Failed to initialize Deepgram client: {e}")


async def generate_and_stream_tts(text: str, websocket: WebSocket):
    """
    Generates TTS audio using Deepgram and streams it over the WebSocket.

    Args:
        text (str): The text to synthesize.
        websocket (WebSocket): The WebSocket connection to stream audio to.
    """
    if not deepgram_client:
        logger.warning("Deepgram client not available. Skipping TTS.")
        return

    logger.info(f"--- TTS Generation --- Text: {text}")
    options = SpeakOptions(
        model="aura-asteria-en",  # Choose a Deepgram Aura model
        encoding="linear16",
        container="none"  # Raw audio stream
    )

    try:
        # Use the correct method chain: speak.rest.v("1").stream
        # .stream() returns the response object synchronously
        speak_result = deepgram_client.speak.rest.v("1").stream(
            {"text": text},
            options
        )

        # The response object (`SpeakStreamResponse`) from `stream()` has a `stream` attribute
        # which is an async iterator for the audio bytes.
        async for chunk in speak_result.stream.aiter_bytes():
            if chunk:
                await websocket.send_bytes(chunk)
                # Optional: Small sleep to prevent overwhelming the connection
                # await asyncio.sleep(0.01)

        logger.info("TTS stream finished.")

    except Exception as e:
        logger.error(f"Error during Deepgram TTS generation or streaming: {e}")
        # Decide if you want to send an error message back to the client
        # await websocket.send_text(json.dumps({"type": "error", "message": "TTS failed"}))
