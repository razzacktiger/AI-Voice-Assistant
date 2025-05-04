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
if not DEEPGRAM_API_KEY:
    print("Warning: DEEPGRAM_API_KEY environment variable not set.")
# Add other keys/configs

# Deepgram Client Configuration
# Note: Consider moving sensitive config loading logic elsewhere later
dg_config: DeepgramClientOptions = DeepgramClientOptions(
    verbose=False  # Set to True for detailed SDK logging
)
deepgram: DeepgramClient = DeepgramClient(DEEPGRAM_API_KEY, dg_config)

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
    transcript_collector = []  # Simple list to collect transcripts for now

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

        # Create Deepgram connection - USE asyncwebsocket instead of asynclive
        dg_connection = deepgram.listen.asyncwebsocket.v(
            "1").stream(dg_options)

        # --- Deepgram Event Handlers ---
        async def on_message(self, result, **kwargs):
            # Ensure result has the expected structure before accessing deeply
            if result and hasattr(result, 'channel') and hasattr(result.channel, 'alternatives') and \
               len(result.channel.alternatives) > 0 and hasattr(result.channel.alternatives[0], 'transcript'):
                sentence = result.channel.alternatives[0].transcript
                if len(sentence.strip()) == 0:  # Check if transcript is empty or just whitespace
                    return
                logging.info(f"Deepgram ->Transcript: {sentence}")
                # TODO: Process the transcript (e.g., RAG query, send back to client?)
                # For now, just log it.
                # Example: await websocket.send_text(f"Transcript: {sentence}")
            else:
                logging.warning(
                    f"Deepgram -> Received unexpected message structure: {result}")

        async def on_metadata(self, metadata, **kwargs):
            print(f"Deepgram Metadata: {metadata}")

        async def on_speech_started(self, speech_started, **kwargs):
            print("Deepgram Speech Started")

        async def on_utterance_end(self, utterance_end, **kwargs):
            print("Deepgram Utterance Ended")
            # If we have collected transcript parts, maybe process them here?
            if transcript_collector:
                full_transcript = " ".join(transcript_collector)
                print(f"Utterance Transcript: {full_transcript}")
                # TODO: Consider if RAG/LLM triggers here or only on final
                transcript_collector.clear()  # Clear for next utterance

        async def on_error(self, error, **kwargs):
            print(f"Deepgram Error: {error}")
            # TODO: Handle potential closure of WebSocket connection to client

        async def on_open(self, open, **kwargs):
            print("Deepgram Connection Opened.")

        async def on_close(self, close, **kwargs):
            print("Deepgram Connection Closed.")

        # Assign event handlers
        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        dg_connection.on(LiveTranscriptionEvents.Metadata, on_metadata)
        dg_connection.on(
            LiveTranscriptionEvents.SpeechStarted, on_speech_started)
        dg_connection.on(LiveTranscriptionEvents.UtteranceEnd,
                         on_utterance_end)
        dg_connection.on(LiveTranscriptionEvents.Error, on_error)
        dg_connection.on(LiveTranscriptionEvents.Open, on_open)
        dg_connection.on(LiveTranscriptionEvents.Close, on_close)

        print("Deepgram connection and handlers configured.")
        # TODO: Start forwarding audio from client WebSocket to dg_connection

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
