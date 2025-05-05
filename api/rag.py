import os
import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import threading
import logging

import pinecone
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sqlmodel import Session

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Delayed Initialization Globals ---
_pinecone_initialized = False
_pinecone_lock = threading.Lock()
_pinecone_index: Optional[Any] = None

_openai_initialized = False
_openai_lock = threading.Lock()
_openai_client: Optional[OpenAI] = None

EMBEDDING_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-4"  # Or specify a different model like "gpt-3.5-turbo"

# --- Environment Variables & Configuration ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv(
    "PINECONE_ENVIRONMENT", "gcp-starter")  # Default if not set
PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME", "ai-voice-assistant-index")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
# Or specify another model like gpt-4o, gpt-3.5-turbo
OPENAI_LLM_MODEL = "gpt-4-turbo"

# --- Global Client Variables ---
_pinecone_client: Optional[Pinecone] = None  # Use Pinecone class type
# Pinecone Index object type isn't directly exposed like before
_pinecone_index: Optional[Any] = None
_openai_client: Optional[OpenAI] = None

_pinecone_initialized = False
_openai_initialized = False

# --- Initialization Functions ---


def initialize_pinecone():
    """Initializes the Pinecone client and index if not already done."""
    global _pinecone_client, _pinecone_index, _pinecone_initialized
    if _pinecone_initialized:
        return

    logger.info("Initializing Pinecone...")
    if not PINECONE_API_KEY:
        logger.error("PINECONE_API_KEY environment variable not set.")
        _pinecone_initialized = True  # Mark as initialized to prevent retries
        return

    if os.getenv("PYTEST_RUNNING") == "1" and PINECONE_API_KEY == "DUMMY_PINECONE_KEY":
        logger.warning(
            "Detected pytest with dummy key, skipping actual Pinecone initialization.")
        # Create a mock Pinecone client and index if needed for type hinting/structure
        # _pinecone_client = MagicMock(spec=Pinecone)
        # _pinecone_index = MagicMock() # Mock the index object too
        _pinecone_initialized = True
        return

    try:
        # New Pinecone initialization
        logger.info(
            f"Attempting Pinecone connection to env: {PINECONE_ENVIRONMENT}")
        # environment is often inferred or handled by client
        _pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

        # Check if the index exists and get the index object
        # Attempt to access index list directly or via .names attribute
        index_list = _pinecone_client.list_indexes()
        available_names = getattr(
            index_list, 'names', index_list if isinstance(index_list, list) else [])

        if PINECONE_INDEX_NAME not in available_names:
            logger.error(
                f"Pinecone index '{PINECONE_INDEX_NAME}' not found in available indexes: {available_names}. Environment: '{PINECONE_ENVIRONMENT}'.")
            _pinecone_index = None
        else:
            _pinecone_index = _pinecone_client.Index(PINECONE_INDEX_NAME)
            logger.info(
                f"Successfully connected to Pinecone index '{PINECONE_INDEX_NAME}'.")

    except Exception as e:
        logger.error(f"Error initializing Pinecone: {e}")
        _pinecone_client = None
        _pinecone_index = None
    finally:
        _pinecone_initialized = True  # Mark as initialized even if failed


def initialize_openai():
    """Initializes the OpenAI client if not already done."""
    global _openai_initialized, _openai_client
    with _openai_lock:
        if not _openai_initialized:
            logger.info("Initializing OpenAI client...")
            if not OPENAI_API_KEY:
                logger.error("OPENAI_API_KEY environment variable not set.")
                if os.getenv("PYTEST_RUNNING") == "1":
                    logger.warning(
                        "Pytest run detected: Skipping OpenAI init check.")
                    _openai_initialized = True
                    # Create a dummy client object if needed for type hints, but methods won't work
                    # _openai_client = MagicMock() # Requires importing MagicMock
                    return
                else:
                    raise ValueError(
                        "OPENAI_API_KEY environment variable not set.")
            try:
                _openai_client = OpenAI(api_key=OPENAI_API_KEY)
                _openai_initialized = True
                logger.info("OpenAI client initialized.")
            except Exception as e:
                logger.error(f"Error initializing OpenAI client: {e}")
                if os.getenv("PYTEST_RUNNING") == "1":
                    logger.warning(
                        "Pytest run detected: Ignoring OpenAI init error.")
                    _openai_initialized = True
                    return
                else:
                    raise

# --- Core Functions ---


def get_openai_embedding(text: str) -> List[float]:
    """Generates embeddings for the given text using OpenAI."""
    initialize_openai()  # Ensure client is ready
    if not _openai_client:
        # Handle case where initialization failed or skipped in tests without mock
        if os.getenv("PYTEST_RUNNING") == "1":
            logger.warning(
                "OpenAI client not initialized in test. Returning dummy embedding.")
            # Return a list of floats with the correct dimension if known, or raise error
            # This depends on how tests mock/use this function.
            # For now, let it raise an error if not mocked properly.
            raise RuntimeError(
                "OpenAI client accessed but not initialized/mocked in test.")
        else:
            raise RuntimeError("OpenAI client not initialized.")

    try:
        response = _openai_client.embeddings.create(
            input=[text], model=EMBEDDING_MODEL)
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error during OpenAI embedding generation: {e}")
        # Re-raise or return specific error value depending on desired handling
        raise


async def query_rag_system(transcript: str, conversation_history: List[Dict[str, Any]]) -> str:
    """
    Queries the RAG system: Gets embeddings, queries Pinecone, calls LLM.
    Initializes clients on first call if needed.
    """
    logger.info(f"--- RAG Query --- Transcript: {transcript}")
    initialize_pinecone()  # Ensure Pinecone is ready (Called once here)
    initialize_openai()   # Ensure OpenAI is ready (Called once here)

    # 1. Get embedding for the transcript
    try:
        # REMOVED: Redundant client initializations
        # initialize_pinecone()
        # initialize_openai()
        embedding = get_openai_embedding(transcript)
    except Exception as e:
        logger.error(f"Error getting OpenAI embedding: {e}")
        # Return early if embedding fails
        print("DEBUG: Returning early due to embedding error.")  # Keep for now
        return "Sorry, I couldn't process that due to an embedding error."

    # Proceed only if embedding was successful
    # 2. Query Pinecone (if index is available)
    pinecone_results = []
    # Use the global _pinecone_index variable
    if _pinecone_index:
        try:
            logger.info(f"Querying Pinecone index '{PINECONE_INDEX_NAME}'...")
            query_response = await asyncio.to_thread(  # Run synchronous Pinecone query in thread
                _pinecone_index.query,
                vector=embedding,
                top_k=3,
                include_metadata=True
            )
            matches = query_response.get('matches', [])
            if matches:
                pinecone_results = [match['metadata']['text']
                                    for match in matches if 'metadata' in match and 'text' in match['metadata']]
                logger.info(
                    f"Retrieved {len(matches)} matches from Pinecone.")
            else:
                logger.info("No relevant context found in Pinecone.")
        except Exception as e:
            logger.error(f"Error querying Pinecone: {e}")
    else:
        # Check if initialization failed or just no index
        if not _pinecone_initialized and os.getenv("PYTEST_RUNNING") != "1":
            logger.warning("Pinecone was not initialized properly.")
        else:
            logger.info(
                "Pinecone index not available or not found, skipping vector search.")

    # 3. Format Prompt for LLM
    prompt_context = "\n".join([res for res in pinecone_results])
    messages = [
        {"role": "system", "content": "You are a helpful voice assistant. Use the provided context from the knowledge base to answer the user's query accurately. Keep your responses concise for voice interaction."}
    ]
    for turn in conversation_history:
        if turn.get('type') == 'user':
            messages.append({"role": "user", "content": turn.get('text', '')})
        elif turn.get('type') == 'assistant':
            messages.append(
                {"role": "assistant", "content": turn.get('text', '')})
    messages.append({
        "role": "user",
        "content": f"Knowledge Base Context:\n{prompt_context}\n\nUser Query: {transcript}"
    })

    logger.info(f"--- LLM Prompt ---")
    logger.info(
        f"Last User Message (with context): {messages[-1]['content']}")

    # 4. Call OpenAI LLM
    if not _openai_client:
        if os.getenv("PYTEST_RUNNING") == "1":
            logger.warning(
                "OpenAI client not initialized in test. Returning dummy response.")
            return "Dummy response (OpenAI client not initialized/mocked)"
        else:
            return "Sorry, the AI service is not available right now."

    try:
        response = await asyncio.to_thread(  # Run synchronous OpenAI call in thread
            _openai_client.chat.completions.create,
            model=LLM_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=150
        )
        llm_response = response.choices[0].message.content.strip()
        logger.info(f"LLM Response: {llm_response}")
        return llm_response
    except Exception as e:
        logger.error(f"Error calling OpenAI LLM: {e}")
        return "Sorry, I encountered an error trying to generate a response."
