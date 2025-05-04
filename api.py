from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, status, Query
import uvicorn
import os
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, auth
from pydantic import BaseModel
from sqlmodel import SQLModel, create_engine, Session, Field, select
from typing import Generator, AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
# Make sure models.py is in the same directory or adjust path
from models import User, CallSession
# Deepgram SDK imports
from deepgram import (DeepgramClient, DeepgramClientOptions,
                      LiveTranscriptionEvents, LiveOptions,)
import logging
import functools  # Import functools if needed later
from pinecone import Pinecone, ServerlessSpec, PodSpec  # Import Pinecone
from openai import OpenAI  # Import OpenAI

# Load environment variables
load_dotenv()

# --- Firebase Admin SDK Initialization ---
SERVICE_ACCOUNT_KEY_PATH = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY_PATH")
if not SERVICE_ACCOUNT_KEY_PATH:
    print("Warning: FIREBASE_SERVICE_ACCOUNT_KEY_PATH environment variable not set. Firebase auth disabled.")
    # exit(1) # Or handle more gracefully
else:
    try:
        cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
        firebase_admin.initialize_app(cred)
        print("Firebase Admin SDK initialized successfully.")
    except Exception as e:
        print(f"Error initializing Firebase Admin SDK: {e}")
        # Potentially exit or disable auth features

# --- Database Connection Setup --- (PostgreSQL or Supabase)
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("Error: DATABASE_URL environment variable not set.")
    # For development, you might use a default SQLite DB
    # DATABASE_URL = "sqlite:///./database.db"
    # print("Using default SQLite database.")
    engine = None  # Or handle error
else:
    # echo=True is useful for debugging, shows SQL statements
    # connect_args = {"check_same_thread": False} # Only needed for SQLite
    engine = create_engine(DATABASE_URL, echo=True)


def create_db_and_tables():
    if engine:
        print("Creating database tables (if they don't exist)...")
        # --- Ensure import is HERE ---
        # Import models specifically for table creation
        from models import User, CallSession
        # -----------------------------
        SQLModel.metadata.create_all(engine)
        print("Database tables checked/created.")
    else:
        print("Database engine not initialized. Skipping table creation.")

# Dependency to get DB session


def get_session() -> Generator[Session, None, None]:
    """
    FastAPI dependency to get a database session.

    Yields a SQLAlchemy Session object for database operations within a request.
    Ensures the session is properly closed after the request.

    Raises:
        HTTPException: 503 Service Unavailable if the database engine is not configured.

    Yields:
        Generator[Session, None, None]: A SQLAlchemy Session object.
    """
    if not engine:
        # Reason: If the database connection wasn't established during startup,
        # we cannot proceed with database operations. Raising 503 indicates
        # a server-side configuration issue.
        raise HTTPException(status_code=503, detail="Database not configured")
    with Session(engine) as session:
        yield session


# --- Utility Functions ---

async def query_pinecone_index(transcript: str, top_k: int = 3) -> list[str]:
    """Queries the Pinecone index with the given transcript.

    Args:
        transcript (str): The text transcript from STT.
        top_k (int): The number of top results to retrieve from Pinecone.

    Returns:
        list[str]: A list of relevant context strings retrieved from Pinecone,
                   or an empty list if an error occurs or Pinecone is disabled.
    """
    if not pinecone_client:
        logging.warning("Pinecone client not available. Skipping query.")
        return []

    if not PINECONE_INDEX_NAME:
        logging.error("PINECONE_INDEX_NAME not configured. Skipping query.")
        return []

    logging.info(
        f"Querying Pinecone index '{PINECONE_INDEX_NAME}' with transcript: '{transcript[:50]}...'")

    try:
        # 1. Generate embeddings for the transcript
        embedding = await get_openai_embedding(transcript)
        if not embedding:
            logging.error(
                "Failed to generate embedding for transcript. Cannot query Pinecone.")
            return []

        # Placeholder: For now, let's assume we have a dummy embedding or use the transcript itself
        # dummy_vector = [0.1] * 1536 # Replace with actual embedding dimension

        # Get a handle to the index
        index = pinecone_client.Index(PINECONE_INDEX_NAME)

        # 2. Perform the query using the generated embedding
        query_response = index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True  # Include metadata to get original text
        )

        # 3. Process the results
        logging.info(
            f"Pinecone query successful. Found {len(query_response['matches'])} matches.")
        # Extract the original text from metadata (assuming it's stored under 'text')
        results = []
        if query_response and query_response['matches']:
            for match in query_response['matches']:
                if 'metadata' in match and 'text' in match['metadata']:
                    results.append(match['metadata']['text'])
                else:
                    logging.warning(
                        f"Pinecone match missing metadata or text: {match.get('id')}")
        else:
            logging.info("Pinecone query returned no matches.")

        # Placeholder response (REMOVE)
        # logging.info(f"Pinecone query successful (Placeholder). Would search for top {top_k} results.")
        # results = [f"Placeholder context related to '{transcript[:20]}' {i+1}" for i in range(top_k)]

        return results

    except Exception as e:
        logging.error(
            f"Error querying Pinecone index '{PINECONE_INDEX_NAME}': {e}")
        return []


async def get_openai_embedding(text: str, model: str = "text-embedding-3-small") -> list[float] | None:
    """Generates embeddings for the given text using the OpenAI API.

    Args:
        text (str): The text to embed.
        model (str): The OpenAI embedding model to use.

    Returns:
        list[float] | None: The embedding vector, or None if an error occurs
                            or the OpenAI client is disabled.
    """
    if not openai_client:
        logging.warning(
            "OpenAI client not available. Cannot generate embeddings.")
        return None

    try:
        # OpenAI recommends replacing newlines for better performance
        text_to_embed = text.replace("\n", " ")
        response = await openai_client.embeddings.create(
            input=[text_to_embed],
            model=model
        )
        embedding = response.data[0].embedding
        logging.info(f"Generated embedding for text: '{text[:50]}...'")
        return embedding
    except Exception as e:
        logging.error(f"Error generating OpenAI embedding: {e}")
        return None


# --- LLM Helper ---

# Modified function signature and prompt formatting
async def get_llm_response(transcript: str, pinecone_context: list[str]) -> str | None:
    """Generates a response from the LLM based on transcript and context."""
    if not openai_client:
        logging.warning(
            "OpenAI client not initialized. Cannot get LLM response.")
        return None

    # Format the context from Pinecone results
    context_str = "\n".join(f"- {item}" for item in pinecone_context)
    if not context_str:
        context_str = "No relevant context found."

    # Construct the prompt using a system and user message
    messages = [
        {
            "role": "system",
            "content": """You are a helpful AI voice assistant. Your goal is to answer the user's query based on the provided transcript and relevant context. Be concise and helpful."""
        },
        {
            "role": "user",
            "content": f"""Transcript of user query:
{transcript}

Relevant context:
{context_str}

Based on the transcript and context, please provide a helpful response."""
        }
    ]

    try:
        logging.info(f"Sending prompt to OpenAI LLM...")  # Log before API call
        # print(f"DEBUG: Prompt Messages: {messages}") # Optional debug print

        response = await openai_client.chat.completions.create(
            model="gpt-4",  # Or specify another model like gpt-3.5-turbo
            messages=messages,
            temperature=0.7,  # Adjust creativity/factuality
            max_tokens=150  # Limit response length
        )
        result = response.choices[0].message.content
        logging.info(f"Received LLM response.")  # Log success
        return result

    except Exception as e:
        logging.error(f"Error getting response from OpenAI: {e}")
        return None

# --- Lifespan Management ---


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    print("Application startup: Creating DB tables...")
    create_db_and_tables()
    yield
    print("Application shutdown.")
    # Add any cleanup logic here if needed

# --- FastAPI App Initialization with Lifespan ---
app = FastAPI(lifespan=lifespan)


# --- Globals / Config (Replace with proper config management later) ---
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME", "ai-voice-assistant")  # Default name

# Deepgram Client Configuration
# Note: Consider moving sensitive config loading logic elsewhere later
if not DEEPGRAM_API_KEY:
    logging.warning("DEEPGRAM_API_KEY not found. Deepgram features disabled.")
    deepgram = None
else:
    try:
        dg_config: DeepgramClientOptions = DeepgramClientOptions(
            verbose=logging.DEBUG)
        deepgram: DeepgramClient | None = DeepgramClient(
            DEEPGRAM_API_KEY, dg_config)
        logging.info("Deepgram client initialized.")
    except Exception as e:
        logging.error(f"Error initializing Deepgram client: {e}")
        deepgram = None

# Pinecone Client
pinecone_client: Pinecone | None = None
if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
    logging.warning(
        "PINECONE_API_KEY or PINECONE_ENVIRONMENT not found. Pinecone features disabled.")
else:
    try:
        pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
        # Optional: Check if index exists and create if not?
        # This might be better done in a separate setup script or lifespan event
        # if PINECONE_INDEX_NAME not in pinecone_client.list_indexes().names:
        #     pinecone_client.create_index(
        #         name=PINECONE_INDEX_NAME,
        #         dimension=1536, # Example dimension for OpenAI ada-002
        #         metric='cosine',
        #         spec=PodSpec(environment=PINECONE_ENVIRONMENT) # Or ServerlessSpec
        #     )
        #     logging.info(f"Created Pinecone index '{PINECONE_INDEX_NAME}'")
        logging.info("Pinecone client initialized.")
    except Exception as e:
        logging.error(f"Error initializing Pinecone client: {e}")
        pinecone_client = None

# OpenAI Client
openai_client: OpenAI | None = None
if not OPENAI_API_KEY:
    logging.warning("OPENAI_API_KEY not found. OpenAI features disabled.")
else:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        # Add organization ID if needed/available:
        # openai_client = OpenAI(api_key=OPENAI_API_KEY, organization=os.getenv("OPENAI_ORG_ID"))
        logging.info("OpenAI client initialized.")
    except Exception as e:
        logging.error(f"Error initializing OpenAI client: {e}")
        openai_client = None

# --- Pydantic Models ---


class TokenData(BaseModel):
    id_token: str


# UserInfo might not be needed if get_current_user returns the DB model
# class UserInfo(BaseModel):
#     uid: str
#     email: str | None = None

# --- Uncomment the UserInfo definition ---
class UserInfo(BaseModel):
    uid: str
    email: str | None = None
    # Add other relevant user fields
# -----------------------------------------

# --- Authentication (Firebase) ---


async def verify_firebase_token(token: str | None = None) -> dict | None:
    """Verifies Firebase ID token and returns user payload or None."""
    if not token:
        return None
    if not firebase_admin._apps:  # Check if Firebase was initialized
        print("Firebase not initialized, cannot verify token.")
        return None
    try:
        # Verify the ID token while checking if the token is revoked.
        decoded_token = auth.verify_id_token(token)
        return decoded_token  # Contains user info like uid, email etc.
    except auth.InvalidIdTokenError:
        print("Invalid Firebase ID token")
        return None
    except Exception as e:
        print(f"Error verifying Firebase token: {e}")
        return None


async def get_current_user(token: str | None = Depends(lambda x: x.headers.get("Authorization")), session: Session = Depends(get_session)) -> User:
    """FastAPI dependency to get user from DB based on Firebase token.
    Verifies token, then fetches user from DB or creates if not found.
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing",
        )

    parts = token.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format",
        )
    id_token = parts[1]

    user_payload = await verify_firebase_token(id_token)
    if user_payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired Firebase token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    firebase_uid = user_payload["uid"]
    email = user_payload.get("email")

    # Check if user exists in our database
    statement = select(User).where(User.firebase_uid == firebase_uid)
    db_user = session.exec(statement).first()

    if db_user is None:
        # User does not exist, create them
        print(f"Creating new user entry for firebase_uid: {firebase_uid}")
        db_user = User(firebase_uid=firebase_uid, email=email)
        session.add(db_user)
        session.commit()
        # Refresh to get DB-assigned ID and created_at
        session.refresh(db_user)
    else:
        # Optional: Update email if it changed in Firebase?
        # if email and db_user.email != email:
        #    db_user.email = email
        #    session.add(db_user)
        #    session.commit()
        #    session.refresh(db_user)
        pass  # User exists

    # Return the database User object (SQLModel instance)
    return db_user

# Example protected endpoint - now uses User model


# Restore response_model
@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    """Example endpoint protected by Firebase authentication, returns DB user."""
    return current_user

# TODO: Add actual signup/login endpoints.
# These usually involve frontend interaction with Firebase client SDK
# and then potentially sending the ID token to backend endpoints like /users/me
# or custom endpoints to create/update user profile in your own DB.

# --- API Endpoints (Stubs) - Update signatures to use User model ---


@app.post("/api/call/start")
async def start_call(current_user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    """Initiates a new call session (protected)."""
    print(
        f"Starting call for user ID: {current_user.id}, Firebase UID: {current_user.firebase_uid}")
    # Create a new CallSession entry in the DB, link to current_user.id
    # Use the DB user's primary key
    new_session = CallSession(user_id=current_user.id)
    session.add(new_session)
    session.commit()
    session.refresh(new_session)
    session_id = new_session.id
    print(f"Created new call session with ID: {session_id}")
    return {"session_id": session_id, "message": "Call session started"}


@app.post("/api/rag/query")
# Define request body model later
async def rag_query(query: dict, current_user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    """Handles RAG query processing (protected)."""
    # TODO: Implement RAG logic (Pinecone -> LLM)
    transcript = query.get("transcript")
    session_id = query.get("session_id")
    return {"response": f"Processed query for user {current_user.id} in session {session_id}: {transcript}"}


@app.get("/api/analytics/report")
# Assuming admin access check needed
async def get_analytics_report(current_user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    """Returns analytics summary (protected)."""
    # TODO: Add role/permission check for admin access
    # TODO: Implement analytics retrieval using DB session
    return {"calls_today": 10, "avg_duration_sec": 60}


@app.get("/api/admin/export")
# Assuming admin access check needed
async def export_analytics(current_user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    """Returns analytics as CSV (protected)."""
    # TODO: Add role/permission check for admin access
    # TODO: Implement CSV export using DB session
    return {"message": "CSV export not implemented yet"}

# --- WebSocket Endpoint ---


async def get_current_user_ws(token: str | None = Query(None)) -> UserInfo | None:
    """Authenticates user for WebSocket connection using a token passed via query param."""
    # Option 1: Query parameter (e.g., ws://.../?token=<token>) - Implemented
    # Option 2: Subprotocol (e.g., Sec-WebSocket-Protocol: bearer, <token>) - More complex

    if not token:
        print("WS Auth: Token missing from query parameter")
        return None

    user_payload = await verify_firebase_token(token)
    if user_payload is None:
        print("WS Auth: Token invalid")
        return None

    print(f"WS Auth: User {user_payload.get('uid')} token verified")
    # Return Pydantic model with basic info, DB check happens in endpoint
    return UserInfo(uid=user_payload["uid"], email=user_payload.get("email"))


# --- Standalone Deepgram Event Handlers ---

async def handle_deepgram_message(result, **kwargs):
    """Handles incoming transcript messages, queries Pinecone, gets LLM response."""
    if result and hasattr(result, 'channel') and hasattr(result.channel, 'alternatives') and \
       len(result.channel.alternatives) > 0 and hasattr(result.channel.alternatives[0], 'transcript'):
        transcript = result.channel.alternatives[0].transcript
        if len(transcript.strip()) == 0:
            return
        logging.info(f"Deepgram -> Transcript: {transcript}")

        # 1. Query Pinecone
        logging.info(
            f"Querying Pinecone for transcript: '{transcript[:50]}...'")
        pinecone_context = await query_pinecone_index(transcript)
        if pinecone_context:
            logging.info(
                f"Pinecone -> Found {len(pinecone_context)} context items.")
            # Log first item for brevity
            logging.info(
                f"Pinecone -> Example context: {pinecone_context[0][:100]}...")
        else:
            logging.info("Pinecone -> No relevant context found.")

        # 2. Get LLM Response
        logging.info("Getting LLM response based on transcript and context...")
        llm_response = await get_llm_response(transcript, pinecone_context)

        # 3. Process LLM Response (Currently Logging)
        if llm_response:
            logging.info(f"LLM -> Response: {llm_response}")
            # TODO: Send response back to the client via WebSocket
        else:
            logging.warning("LLM -> Failed to get response.")

    else:
        logging.warning(
            f"Deepgram -> Received unexpected message structure: {result}")


async def handle_deepgram_metadata(metadata, **kwargs):
    """Handles metadata messages from Deepgram."""
    logging.info(f"Deepgram -> Metadata: {metadata}")


async def handle_deepgram_speech_started(speech_started, **kwargs):
    """Handles speech started events from Deepgram."""
    logging.info("Deepgram -> Speech Started")


async def handle_deepgram_utterance_end(utterance_end, **kwargs):
    """Handles utterance end events from Deepgram."""
    logging.info("Deepgram -> Utterance Ended")
    # utterance_transcript = transcript_collector.get() # Need state management if collecting
    # logging.info(f"Utterance Transcript: {utterance_transcript}")


async def handle_deepgram_error(error, **kwargs):
    """Handles error messages from Deepgram."""
    logging.error(f"Deepgram -> Error: {error}")
    # TODO: Decide how to propagate errors (e.g., close client connection?)


async def handle_deepgram_open(open_event, **kwargs):
    """Handles the connection open event from Deepgram."""
    logging.info("Deepgram -> Connection Opened.")


async def handle_deepgram_close(close_event, **kwargs):
    """Handles the connection close event from Deepgram."""
    logging.info("Deepgram -> Connection Closed")


async def handle_deepgram_unhandled(unhandled, **kwargs):
    """Handles unhandled messages from Deepgram."""
    logging.warning(f"Deepgram -> Unhandled message: {unhandled}")


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    token: str | None = Query(None),  # Explicitly get token from query
    session: Session = Depends(get_session)  # Inject DB session
):
    """Handles WebSocket connection, auth, validation, and Deepgram STT integration."""

    # --- WebSocket Authentication & Validation ---
    user_info = await get_current_user_ws(token)
    if not user_info:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Authentication failed")
        return

    # --- Fetch User from DB ---
    # Reason: We need the DB User object's ID to validate against the CallSession.
    statement = select(User).where(User.firebase_uid == user_info.uid)
    db_user = session.exec(statement).first()

    if not db_user:
        # This case should theoretically not happen if verify_firebase_token succeeded
        # and get_current_user's implicit creation works, but check defensively.
        print(
            f"WS Error: Authenticated user UID {user_info.uid} not found in DB.")
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR, reason="User DB record not found")
        return

    # --- Validate session_id ---
    try:
        session_id_int = int(session_id)
        statement_session = select(CallSession).where(
            CallSession.id == session_id_int)
        call_session = session.exec(statement_session).first()

        if not call_session:
            print(f"WS Warn: Call session ID {session_id_int} not found.")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Call session not found")
            return

        if call_session.user_id != db_user.id:
            print(
                f"WS Warn: User {db_user.id} attempted to access session {session_id_int} owned by user {call_session.user_id}.")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Session access forbidden")
            return

    except ValueError:
        print(f"WS Error: Invalid session ID format: {session_id}")
        await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA, reason="Invalid session ID format")
        return
    except Exception as e:
        # Catch potential DB errors during fetch
        print(f"WS Error: Database error during session validation: {e}")
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR, reason="Session validation database error")
        return

    # --- Accept Connection ---
    await websocket.accept()
    print(
        f"WebSocket connection accepted for session: {session_id}, user: {db_user.id}")

    # --- Deepgram Connection Setup ---
    dg_connection = None
    # transcript_collector = [] # State needed if collecting across utterances

    try:
        print("Attempting to connect to Deepgram...")
        # Configure Deepgram options (adjust as needed)
        # Ref: https://developers.deepgram.com/docs/streaming-audio
        dg_options = LiveOptions(
            model="nova-2",  # Or other model like "base"
            language="en-US",
            smart_format=True,
            encoding="linear16",  # Adjust if client sends different encoding
            sample_rate=16000,   # Adjust to match client audio
            # Add other options like interim_results, endpointing, etc.
        )

        # Create Deepgram connection
        dg_connection = await deepgram.listen.asyncwebsocket.v("1").stream(dg_options)

        # --- Assign Standalone Event Handlers ---
        # Use the functions defined outside the endpoint
        dg_connection.on(LiveTranscriptionEvents.Transcript,
                         handle_deepgram_message)
        dg_connection.on(LiveTranscriptionEvents.Metadata,
                         handle_deepgram_metadata)
        dg_connection.on(LiveTranscriptionEvents.SpeechStarted,
                         handle_deepgram_speech_started)
        dg_connection.on(LiveTranscriptionEvents.UtteranceEnd,
                         handle_deepgram_utterance_end)
        dg_connection.on(LiveTranscriptionEvents.Error, handle_deepgram_error)
        # dg_connection.on(LiveTranscriptionEvents.Open, handle_deepgram_open) # Usually not needed
        dg_connection.on(LiveTranscriptionEvents.Close, handle_deepgram_close)
        dg_connection.on(LiveTranscriptionEvents.Unhandled,
                         handle_deepgram_unhandled)

        logging.info("Deepgram connection and handlers configured.")

        # --- Main WebSocket Loop (Expect Bytes for Audio) ---
        while True:
            # Primarily listen for bytes (audio)
            # Use receive() which can handle text or bytes based on mode
            # Or be more explicit if client only sends one type mostly
            message = await websocket.receive()
            message_type = message.get("type")

            if message_type == "websocket.receive":
                if "text" in message:
                    text_data = message["text"]
                    print(
                        f"Received text message from User {db_user.id}: {text_data}")
                    # Handle potential control messages
                    await websocket.send_text(f"Server received text: {text_data}")
                elif "bytes" in message:
                    audio_chunk = message["bytes"]
                    # Ensure it's bytes before sending to Deepgram
                    if isinstance(audio_chunk, bytes):
                        # print(f"Received audio chunk from User {db_user.id}: {len(audio_chunk)} bytes") # Less verbose
                        if dg_connection:
                            await dg_connection.send(audio_chunk)
                        else:
                            print(
                                "Warning: Deepgram connection not active, cannot send audio.")
                    else:
                        print(
                            f"Received 'bytes' key but data is not bytes type: {type(audio_chunk)}")
            elif message_type == "websocket.disconnect":
                print(f"Received disconnect message type.")
                # The main exception handler below will catch WebSocketDisconnect
                break  # Exit loop on disconnect message
            else:
                print(f"Received unexpected message type: {message_type}")

    except WebSocketDisconnect:
        print(
            f"WebSocket disconnected for session: {session_id}, user ID: {db_user.id}")
        if dg_connection:
            await dg_connection.finish()
            print("Deepgram connection finished due to WebSocket disconnect.")
    except Exception as e:
        print(
            f"WebSocket/Deepgram error for session {session_id}, user ID: {db_user.id}: {e}")
        if dg_connection:
            await dg_connection.finish()
            print("Deepgram connection finished due to error.")
        # Close WebSocket connection if not already closed
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except RuntimeError:
            pass
    finally:
        # Ensure Deepgram connection is cleaned up if it was opened
        if dg_connection:
            await dg_connection.finish()
            print("Deepgram connection cleanup finalized.")

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting FastAPI server...")
    # Ensure reload=True is used if running directly
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
