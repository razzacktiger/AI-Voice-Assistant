# api/dependencies.py

import os
import logging
from typing import Optional
import threading  # Use threading Lock for lazy init

from deepgram import DeepgramClient
from dotenv import load_dotenv

# Remove direct import from websocket
# from .websocket import deepgram_client as global_dg_client

load_dotenv()  # Ensure env vars are loaded here too

logger = logging.getLogger(__name__)

# --- Global state for Deepgram Client (initialized lazily) ---
_dg_client: Optional[DeepgramClient] = None
_dg_client_initialized = False
_dg_init_lock = threading.Lock()
# -------------------------------------------------------------


def initialize_deepgram_client():
    """Initializes the Deepgram client if not already done. Thread-safe."""
    global _dg_client, _dg_client_initialized
    # Use double-checked locking for efficiency
    if not _dg_client_initialized:
        with _dg_init_lock:
            if not _dg_client_initialized:
                logger.info("Attempting to initialize Deepgram client...")
                api_key = os.getenv("DEEPGRAM_API_KEY")
                if not api_key:
                    logger.warning(
                        "DEEPGRAM_API_KEY not set in environment. Deepgram client cannot be initialized.")
                    # Mark as initialized (to prevent retries) even if key is missing
                    _dg_client_initialized = True
                    return  # _dg_client remains None

                try:
                    _dg_client = DeepgramClient(api_key)
                    logger.info("Deepgram client initialized successfully.")
                except Exception as e:
                    logger.error(
                        f"Failed to initialize Deepgram client: {e}", exc_info=True)
                    _dg_client = None  # Ensure client is None on error
                finally:
                    _dg_client_initialized = True  # Mark as initialized regardless of outcome


def get_deepgram_live_client() -> Optional[DeepgramClient]:
    """FastAPI dependency to provide the initialized Deepgram client.
    Initializes the client on the first call.

    Returns:
        Optional[DeepgramClient]: The initialized client instance, or None if init failed.
    """
    if not _dg_client_initialized:
        initialize_deepgram_client()  # Initialize on first access

    if not _dg_client:
        logger.warning(
            "Accessing Deepgram client dependency, but client is not initialized (API key missing or init failed).")
        # Depending on requirements, you might raise an HTTPException here
        # raise HTTPException(status_code=503, detail="Speech service unavailable")

    return _dg_client

# Add other dependencies here as the application grows
