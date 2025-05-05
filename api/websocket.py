import os
import asyncio
import json
import base64
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any

from fastapi import WebSocket, WebSocketDisconnect, Depends, Query, HTTPException, status, APIRouter
from sqlmodel import Session, select
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
)
from dotenv import load_dotenv
import logging
import uuid

from .database import get_session
from .models import User, CallSession, CallStatus, LogLevel
from .auth import verify_token_and_get_user_info
from .connection_manager import ConnectionManager, ClientInfo
from .rag import query_rag_system
from .tts import generate_and_stream_tts
from .dependencies import get_deepgram_live_client
from .logging_utils import add_log_entry

load_dotenv()

logger = logging.getLogger(__name__)
manager = ConnectionManager()

# Create an APIRouter instance
router = APIRouter()

# Helper function to get or create user in DB (used only by WebSocket)


async def _get_or_create_user_ws(firebase_uid: str, email: str, db: Session) -> User:
    """Gets or creates a user in the database, used internally by WebSocket."""
    try:
        statement = select(User).where(User.firebase_uid == firebase_uid)
        existing_user = db.exec(statement).one_or_none()
        if existing_user:
            return existing_user
        else:
            logger.info(
                f"WS: Creating new user for firebase_uid: {firebase_uid}")
            # Log user creation attempt
            add_log_entry(db, LogLevel.INFO, "user_create_attempt", component="websocket_auth",
                          message=f"Attempting creation for firebase_uid {firebase_uid}", details={"email": email})
            new_user = User(firebase_uid=firebase_uid, email=email)
            db.add(new_user)
            db.commit()
            db.refresh(new_user)
            # Log user creation success
            add_log_entry(db, LogLevel.INFO, "user_create_success",
                          component="websocket_auth", user_id=new_user.id, details={"email": email})
            return new_user
    except Exception as e:
        logger.error(f"WS: Database error for user {firebase_uid}: {e}")
        # Log user creation failure
        add_log_entry(db, LogLevel.ERROR, "user_create_error", component="websocket_auth", message=str(
            e), details={"firebase_uid": firebase_uid, "email": email})
        db.rollback()  # Rollback on error
        raise  # Re-raise to be handled by the main endpoint


async def create_call_session(db: Session, user_id: uuid.UUID) -> uuid.UUID:
    """Creates a new CallSession record in the database."""
    session = CallSession(user_id=user_id)
    db.add(session)
    try:
        db.commit()
        db.refresh(session)
        logger.info(f"CallSession {session.id} created for user {user_id}.")
        # Log is typically added in the caller after success
        # add_log_entry(db, LogLevel.INFO, "call_session_start", user_id=user_id, session_id=session.id, component="websocket")
        return session.id
    except Exception as e:
        logger.error(
            f"Database error creating CallSession for user {user_id}: {e}")
        # Log session creation failure
        add_log_entry(db, LogLevel.ERROR, "call_session_create_error",
                      user_id=user_id, component="websocket", message=str(e))
        db.rollback()
        raise Exception("Failed to create call session record.")  # Re-raise


async def update_conversation_history(db: Session, session_id: uuid.UUID, turn: Dict[str, Any]):
    """Appends a turn to the conversation history of a CallSession."""
    session = db.get(CallSession, session_id)
    if session:
        try:
            # Create a mutable copy
            current_history = list(session.conversation_history)
            current_history.append(turn)
            session.conversation_history = current_history
            db.add(session)
            db.commit()
            logger.debug(
                f"Updated conversation history for session {session_id}")
            # Log history update
            add_log_entry(db, LogLevel.DEBUG, "history_update", session_id=session_id,
                          component="websocket", details={"turn_type": turn.get("type")})
        except Exception as e:
            logger.error(
                f"DB error updating history for session {session_id}: {e}")
            add_log_entry(db, LogLevel.ERROR, "db_history_update_error",
                          session_id=session_id, component="websocket", message=str(e))
            db.rollback()
    else:
        logger.warning(
            f"Attempted to update history for non-existent session {session_id}")
        add_log_entry(db, LogLevel.WARNING, "db_history_update_fail",
                      session_id=session_id, component="websocket", message="Session not found")


async def get_conversation_history(db: Session, session_id: uuid.UUID) -> List[Dict[str, Any]]:
    """Retrieves the conversation history for a CallSession."""
    session = db.get(CallSession, session_id)
    if session:
        return session.conversation_history
    else:
        logger.warning(
            f"Attempted to get history for non-existent session {session_id}")
        add_log_entry(db, LogLevel.WARNING, "db_history_get_fail",
                      session_id=session_id, component="websocket", message="Session not found")
        return []


async def handle_deepgram_transcript(result, client_info: ClientInfo, db: Session):
    """Handles incoming transcripts from Deepgram."""
    # Get needed info from client_info
    websocket = client_info.websocket
    client_id = client_info.client_id
    call_session_id = client_info.session_id
    user_id = client_info.user_id

    try:
        transcript = result.channel.alternatives[0].transcript
        if not transcript:
            return

        logger.info(f"Transcript received (Client {client_id}): {transcript}")
        add_log_entry(db, LogLevel.INFO, "stt_result",
                      user_id=user_id,
                      session_id=call_session_id,
                      component="stt_handler",
                      message=transcript,
                      details={"is_final": result.is_final, "speech_final": result.speech_final})

        if result.is_final and result.speech_final:
            logger.info(
                f"Final transcript received for client {client_id}. Querying RAG...")
            user_turn = {"type": "user", "text": transcript,
                         "timestamp": datetime.now(timezone.utc).isoformat()}
            # Update history via DB function
            await update_conversation_history(db, call_session_id, user_turn)

            # Get history via DB function
            history = await get_conversation_history(db, call_session_id)

            add_log_entry(db, LogLevel.INFO, "rag_query_start",
                          user_id=user_id,
                          session_id=call_session_id,
                          component="stt_handler",
                          details={"transcript": transcript})
            try:
                # Pass history excluding current user turn (last item)
                llm_response = await query_rag_system(transcript, history[:-1])
                add_log_entry(db, LogLevel.INFO, "rag_query_success",
                              user_id=user_id,
                              session_id=call_session_id,
                              component="stt_handler",
                              message=llm_response)
            except Exception as rag_err:
                logger.error(f"RAG Error for client {client_id}: {rag_err}")
                add_log_entry(db, LogLevel.ERROR, "rag_query_error",
                              user_id=user_id,
                              session_id=call_session_id,
                              component="stt_handler",
                              message=str(rag_err),
                              details={"traceback": str(rag_err.__traceback__)})
                await manager.send_error(client_info, "Error processing your request.")
                return

            if llm_response:
                logger.info(
                    f"LLM response for client {client_id}: {llm_response}")
                assistant_turn = {"type": "assistant", "text": llm_response,
                                  "timestamp": datetime.now(timezone.utc).isoformat()}
                # Update history via DB function
                await update_conversation_history(db, call_session_id, assistant_turn)

                add_log_entry(db, LogLevel.INFO, "tts_request",
                              user_id=user_id,
                              session_id=call_session_id,
                              component="stt_handler",
                              message=llm_response)
                try:
                    await generate_and_stream_tts(llm_response, websocket)
                    logger.info(f"TTS stream completed for client {client_id}")
                    add_log_entry(db, LogLevel.INFO, "tts_stream_success",
                                  user_id=user_id,
                                  session_id=call_session_id,
                                  component="stt_handler")
                except Exception as tts_err:
                    logger.error(
                        f"TTS Error for client {client_id}: {tts_err}")
                    add_log_entry(db, LogLevel.ERROR, "tts_error",
                                  user_id=user_id,
                                  session_id=call_session_id,
                                  component="stt_handler",
                                  message=str(tts_err),
                                  details={"traceback": str(tts_err.__traceback__)})
                    await manager.send_error(client_info, "Error generating audio response.")
            else:
                logger.warning(
                    f"Received empty LLM response for client {client_id}")
                add_log_entry(db, LogLevel.WARNING, "llm_empty_response",
                              user_id=user_id,
                              session_id=call_session_id,
                              component="stt_handler")

    except Exception as e:
        logger.error(
            f"Error processing transcript for client {client_id}: {e}", exc_info=True)
        add_log_entry(db, LogLevel.ERROR, "transcript_processing_error",
                      user_id=user_id,
                      session_id=call_session_id,
                      component="stt_handler",
                      message=str(e),
                      details={"traceback": str(e.__traceback__)})


async def setup_deepgram_connection(client_info: ClientInfo, db: Session):
    """Sets up the Deepgram live transcription connection and event handlers."""
    deepgram_client = get_deepgram_live_client()
    if not deepgram_client:
        raise Exception("Deepgram client not initialized")

    client_id = client_info.client_id
    websocket = client_info.websocket
    user_id = client_info.user_id
    call_session_id = client_info.session_id

    try:
        # 1. Get the connection interface using the recommended method
        dg_connection = deepgram_client.listen.asyncwebsocket.v("1")
        # --- ADD DEBUG LOG ---
        logger.debug(
            f"DEBUG: dg_connection object type: {type(dg_connection)}, has start attr: {hasattr(dg_connection, 'start')}")
        if hasattr(dg_connection, 'start'):
            logger.debug(
                f"DEBUG: dg_connection.start object type: {type(dg_connection.start)}")
        # ----------------------

        # 2. Register event handlers BEFORE connecting
        async def on_metadata(inner_dg_connection, metadata, **kwargs):
            logger.info(f"DG Metadata (Client {client_id}): {metadata}")

        async def on_speech_started(inner_dg_connection, speech_started, **kwargs):
            logger.info(f"DG Speech Started (Client {client_id})")

        async def on_utterance_end(inner_dg_connection, utterance_end, **kwargs):
            logger.info(f"DG Utterance End (Client {client_id})")

        async def on_error(inner_dg_connection, error, **kwargs):
            logger.error(f"DG Error (Client {client_id}): {error}")
            add_log_entry(db, LogLevel.ERROR, "stt_error",
                          user_id=user_id,
                          session_id=call_session_id,
                          component="deepgram_handler",
                          message="Deepgram STT Error",
                          details={"error": str(error)})
            # Ensure websocket is closed gracefully
            try:
                await websocket.close(code=status.WS_1011_INTERNAL_ERROR, reason="STT Error")
            except Exception:
                pass  # Ignore if already closed

        dg_connection.on(LiveTranscriptionEvents.Transcript, lambda dg_conn, result, **kwargs: asyncio.create_task(
            handle_deepgram_transcript(result, client_info, db)))
        dg_connection.on(LiveTranscriptionEvents.Metadata, on_metadata)
        dg_connection.on(
            LiveTranscriptionEvents.SpeechStarted, on_speech_started)
        dg_connection.on(LiveTranscriptionEvents.UtteranceEnd,
                         on_utterance_end)
        dg_connection.on(LiveTranscriptionEvents.Error, on_error)

        # 3. Define connection options
        options = LiveOptions(
            model="nova-2", language="en-US", smart_format=True,
            encoding="linear16", sample_rate=16000,
            endpointing=300,
        )

        # 4. Start the connection
        await dg_connection.start(options)

        logger.info(f"Deepgram connection established for client {client_id}")
        add_log_entry(db, LogLevel.INFO, "stt_connection_start",
                      user_id=user_id,
                      session_id=call_session_id,
                      component="websocket")
        return dg_connection

    except Exception as e:
        logger.error(
            f"Failed to connect to Deepgram for client {client_id}: {e}", exc_info=True)
        add_log_entry(db, LogLevel.ERROR, "stt_connection_error",
                      user_id=user_id,
                      session_id=call_session_id,
                      component="websocket",
                      message=str(e),
                      details={"traceback": str(e.__traceback__)})
        # Ensure websocket is closed gracefully
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR, reason="Failed to connect to STT service")
        except Exception:
            pass  # Ignore if already closed
        raise


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
    db: Session = Depends(get_session),
    deepgram_client: DeepgramClient = Depends(get_deepgram_live_client)
):
    """Handles WebSocket connections, authentication, and message processing."""
    client_id = str(uuid.uuid4())
    # Create ClientInfo object early
    client_info = ClientInfo(websocket=websocket, client_id=client_id)

    logger.info(f"New WebSocket connection attempt from client {client_id}")
    await websocket.accept()

    user_record: Optional[User] = None
    call_session_id: Optional[uuid.UUID] = None
    dg_connection = None
    error_occurred = False  # Flag to track if error happened before disconnect

    try:
        if not token:
            # Use client_info with send_error
            await manager.send_error(client_info, "Authentication token missing.")
            add_log_entry(db, LogLevel.WARNING, "auth_failed", component="websocket",
                          message="Token missing", details={"client_id": client_id})
            error_occurred = True
            raise WebSocketDisconnect(code=4001, reason="Token missing")

        try:
            user_info = await verify_token_and_get_user_info(token=token)
            firebase_uid = user_info.get("uid")
            email = user_info.get("email")
            logger.info(
                f"Token verified for client {client_id}, uid: {firebase_uid}")

            user_record = await _get_or_create_user_ws(firebase_uid, email, db)
            if not user_record:
                error_occurred = True
                raise Exception("Failed to retrieve or create user record.")
            # Update client_info with user_id
            client_info.user_id = user_record.id

        except Exception as auth_exc:
            logger.error(
                f"Authentication failed for client {client_id}: {auth_exc}")
            add_log_entry(db, LogLevel.ERROR, "auth_failed", component="websocket", message=str(
                auth_exc), details={"client_id": client_id})
            # Use client_info with send_error
            await manager.send_error(client_info, f"Authentication failed: {auth_exc}")
            error_occurred = True
            raise WebSocketDisconnect(
                code=4003, reason="Authentication Failed")

        try:
            call_session_id = await create_call_session(db, user_record.id)
            # Update client_info with session_id
            client_info.session_id = call_session_id
            logger.info(
                f"Created CallSession {call_session_id} for user {user_record.id}")
            add_log_entry(db, LogLevel.INFO, "call_session_start", user_id=user_record.id,
                          session_id=call_session_id, component="websocket")
        except Exception as cs_exc:
            logger.error(
                f"Failed to create CallSession for user {user_record.id}: {cs_exc}")
            # Use client_info with send_error
            await manager.send_error(client_info, "Failed to initialize call session.")
            error_occurred = True
            raise WebSocketDisconnect(
                code=1011, reason="Call Session Creation Failed")

        # ---> RE-ADD LOG 1 <---
        logger.info(f"DEBUG WS: Checking if deepgram_client exists...")
        if not deepgram_client:
            logger.error(
                "Cannot establish STT connection: Deepgram client not available (via dependency).")
            await manager.send_error(client_info, "STT service unavailable.")
            error_occurred = True
            raise WebSocketDisconnect(
                code=status.WS_1011_INTERNAL_ERROR, reason="STT service unavailable")

        # ---> RE-ADD LOG 2 <---
        logger.info(f"DEBUG WS: About to call setup_deepgram_connection...")
        # Pass client_info to setup function
        dg_connection = await setup_deepgram_connection(client_info, db)
        # Update client_info with dg_connection
        client_info.dg_connection = dg_connection

        # Pass the completed client_info to the manager
        await manager.connect(client_info)
        add_log_entry(db, LogLevel.INFO, "websocket_connect", user_id=user_record.id,
                      session_id=call_session_id, component="websocket", message=f"Client {client_id} connected.")

        # --- Add a small sleep to allow async tasks to potentially progress further in tests ---
        await asyncio.sleep(0.01)
        # --------------------------------------------------------------------------------------

        while True:
            data = await websocket.receive_bytes()
            if dg_connection:
                await dg_connection.send(data)

    except WebSocketDisconnect as e:
        logger.warning(
            f"WebSocket disconnected for client {client_id}: {e.reason} (code: {e.code})")
        # Status determined in finally block
    except HTTPException as e:
        logger.error(
            f"HTTP Exception during WebSocket handling for {client_id}: {e.detail}")
        add_log_entry(db, LogLevel.ERROR, "websocket_http_exception",
                      user_id=client_info.user_id,
                      session_id=client_info.session_id,
                      component="websocket",
                      message=e.detail, details={"status_code": e.status_code})
        error_occurred = True  # Mark error before close
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason=f"Error: {e.detail}")
    except Exception as e:
        logger.error(
            f"Unexpected error for client {client_id}: {e}", exc_info=True)
        add_log_entry(db, LogLevel.ERROR, "websocket_runtime_error",
                      user_id=client_info.user_id,
                      session_id=client_info.session_id,
                      component="websocket",
                      message=str(e), details={"traceback": str(e.__traceback__)})
        error_occurred = True  # Mark error before close
        # Attempt to close gracefully if possible
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR, reason=f"Internal server error")
        except RuntimeError:
            logger.warning(
                f"WebSocket already closed for client {client_id} during exception handling.")

    finally:
        logger.info(f"Cleaning up connection for client {client_id}")
        # Determine final status based on whether an error was explicitly caught
        final_status = CallStatus.ERROR if error_occurred else CallStatus.ENDED
        await cleanup_connection(db, client_info, final_status)


async def cleanup_connection(db: Session, client_info: ClientInfo, status: CallStatus):
    """Cleans up resources for a disconnected client."""
    client_id = client_info.client_id
    session_id = client_info.session_id
    user_id = client_info.user_id
    # Get dg_connection from client_info if needed for closing
    dg_connection = client_info.dg_connection

    logger.info(
        f"Cleaning up connection for client {client_id} (Session: {session_id}) with status {status.value}")
    add_log_entry(db, LogLevel.INFO, "websocket_disconnect_cleanup",
                  user_id=user_id,
                  session_id=session_id,
                  component="websocket",
                  message=f"Client {client_id} disconnected. Final status: {status.value}",
                  details={"final_status": status.value})

    # Remove from manager using client_id
    # disconnect() now handles closing the websocket itself
    await manager.disconnect(client_id)
    # logger info about removal is now inside disconnect

    # Close Deepgram connection
    if dg_connection:
        try:
            await dg_connection.finish()
            logger.info(f"Deepgram connection finished for client {client_id}")
            add_log_entry(db, LogLevel.INFO, "stt_connection_stop",
                          user_id=user_id, session_id=session_id, component="websocket")
        except Exception as e:
            logger.error(
                f"Error closing Deepgram connection for client {client_id}: {e}")
            add_log_entry(db, LogLevel.ERROR, "stt_connection_stop_error", user_id=user_id,
                          session_id=session_id, component="websocket", message=str(e))

    # Update CallSession in DB
    if session_id:
        session = db.get(CallSession, session_id)
        if session:
            session.end_time = datetime.now(timezone.utc)
            session.status = status
            db.add(session)
            try:
                db.commit()
                logger.info(
                    f"Updated CallSession {session_id} status to {status.value}")
                add_log_entry(db, LogLevel.INFO, "call_session_end", user_id=user_id,
                              session_id=session_id, component="websocket", details={"final_status": status.value})
            except Exception as e:
                logger.error(
                    f"DB error updating CallSession {session_id}: {e}")
                add_log_entry(db, LogLevel.ERROR, "db_session_update_error", user_id=user_id,
                              session_id=session_id, component="websocket", message=str(e))
                db.rollback()
        else:
            logger.warning(
                f"CallSession {session_id} not found during cleanup for client {client_id}")
            add_log_entry(db, LogLevel.WARNING, "db_session_update_fail", user_id=user_id,
                          session_id=session_id, component="websocket", message="Session not found during cleanup")

    logger.info(f"Cleanup complete for client {client_id}")
