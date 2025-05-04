from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
import uvicorn
import os
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, auth
from pydantic import BaseModel
from sqlmodel import SQLModel, create_engine, Session, Field, select
from typing import Generator, AsyncGenerator
from contextlib import asynccontextmanager
# Make sure models.py is in the same directory or adjust path
from models import User, CallSession

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
# Add other keys/configs

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


async def get_current_user_ws(token: str | None = None) -> UserInfo | None:
    """Authenticates user for WebSocket connection using a token passed via query param or subprotocol."""
    # This needs a mechanism to pass the token securely during WebSocket connection
    # Option 1: Query parameter (e.g., ws://.../?token=<token>)
    # Option 2: Subprotocol (e.g., Sec-WebSocket-Protocol: bearer, <token>)
    # Query parameter is simpler but less standard for auth.

    if not token:
        print("WS Auth: Token missing")
        return None

    user_payload = await verify_firebase_token(token)
    if user_payload is None:
        print("WS Auth: Token invalid")
        return None

    print(f"WS Auth: User {user_payload.get('uid')} authenticated")
    return UserInfo(uid=user_payload["uid"], email=user_payload.get("email"))


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str, token: str | None = None):
    """Handles WebSocket connection for real-time voice communication (with auth attempt)."""

    # --- WebSocket Authentication ---
    current_user = await get_current_user_ws(token)
    if not current_user:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Authentication failed")
        return

    await websocket.accept()
    print(
        f"WebSocket connection established for session: {session_id} by user {current_user.uid}")

    # TODO: Validate session_id belongs to current_user
    # TODO: Initiate Deepgram STT connection here

    # --- Main WebSocket Loop ---
    try:
        while True:
            data = await websocket.receive()
            # Handle different message types (text, bytes)
            if isinstance(data, dict) and 'text' in data:
                print(
                    f"Received text message from {current_user.uid}: {data['text']}")
                # Potentially handle control messages from client
                await websocket.send_text(f"Server received text: {data['text']}")
            elif isinstance(data, dict) and 'bytes' in data:
                audio_chunk = data['bytes']
                print(
                    f"Received audio chunk from {current_user.uid}: {len(audio_chunk)} bytes")
                # TODO: Forward audio_chunk to Deepgram STT
                # TODO: Receive transcript from Deepgram STT
                # TODO: Trigger RAG query -> LLM -> TTS
                # TODO: Send TTS audio back to client
            else:
                # Handle unexpected data format
                print(f"Received unexpected data format: {type(data)}")

    except WebSocketDisconnect:
        print(
            f"WebSocket disconnected for session: {session_id}, user: {current_user.uid}")
        # TODO: Clean up resources (e.g., close Deepgram connection)
    except Exception as e:
        print(
            f"WebSocket error for session {session_id}, user: {current_user.uid}: {e}")
        # Internal Error
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting FastAPI server...")
    # Ensure reload=True is used if running directly
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
