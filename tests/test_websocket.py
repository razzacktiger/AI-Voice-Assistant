# Tests for WebSocket connection, auth, and core flow, including STT/RAG/TTS

import pytest
from unittest.mock import patch, MagicMock, AsyncMock, ANY
import asyncio
import uuid
from fastapi import status, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.testclient import TestClient
from datetime import datetime, timezone, timedelta
from sqlmodel import Session

# Import app and dependencies/models from the new structure
from api.main import app, AnalyticsReport
from api.models import User, CallSession, LogEntry, LogLevel, CallStatus
# Import the specific auth functions needed
from api.auth import verify_token_and_get_user_info  # Replaces get_current_user
from api.database import get_session
from api.websocket import manager as websocket_manager
from api.rag import query_rag_system
from api.tts import generate_and_stream_tts
from deepgram import (
    DeepgramClient,
    LiveOptions,
    LiveTranscriptionEvents,
    LiveResultResponse,
)
from api.connection_manager import ClientInfo
from api.dependencies import get_deepgram_live_client

# TestClient using the app from api.main
client = TestClient(app)

# --- Fixtures ---


@pytest.fixture
def mock_db_session():
    # ... (keep existing mock_db_session fixture) ...
    mock_session = MagicMock(spec=Session)
    mock_exec = MagicMock()
    mock_scalar = MagicMock()
    # Configure mock behaviors as needed for tests
    mock_session.exec.return_value = mock_exec
    mock_exec.scalar.return_value = mock_scalar
    mock_exec.one_or_none.return_value = None  # Default
    mock_session.get.return_value = None  # Default
    yield mock_session, mock_exec


@pytest.fixture
def mock_deepgram_stt_fixture():
    """Factory for creating Deepgram STT **connection** mocks."""
    def _factory(send_raises=None, finish_raises=None):
        # Mock the connection object returned by client.listen...v('1')
        mock_dg_connection = AsyncMock()  # Use AsyncMock for the connection obj itself
        mock_dg_connection.send = AsyncMock(side_effect=send_raises)
        mock_dg_connection.finish = AsyncMock(side_effect=finish_raises)
        mock_dg_connection.start = AsyncMock()  # <<< Keep mock for START method
        mock_dg_connection._callbacks = {}

        def on_side_effect(event, callback):
            # Store the callback associated with the event
            mock_dg_connection._callbacks[event] = callback
            print(f"DEBUG: Registered callback for {event}")  # Debug print
        mock_dg_connection.on = MagicMock(side_effect=on_side_effect)

        return mock_dg_connection
    yield _factory


@pytest.fixture
def auto_mock_external_services(mock_db_session, mock_deepgram_stt_fixture):
    """Fixture to mock external services AND dependencies for websocket tests.
    Must be explicitly requested by tests.
    """
    # Mock DB Session setup & FastAPI override
    mock_session, mock_exec_result = mock_db_session
    original_get_session = app.dependency_overrides.get(get_session)
    app.dependency_overrides[get_session] = lambda: mock_session

    # Mock User Data & DB Responses
    mock_user_instance = User(
        id=uuid.uuid4(),
        firebase_uid="mock_ws_firebase_uid",
        email="mock_ws_user@example.com",
        created_at=datetime.now(timezone.utc)
    )
    mock_exec_result.one_or_none.return_value = mock_user_instance

    def refresh_side_effect(obj, *args, **kwargs):
        obj.id = obj.id or uuid.uuid4()
        return obj
    mock_session.refresh = MagicMock(side_effect=refresh_side_effect)

    # --- Start Patches ---
    patches = {
        "firebase": patch('api.auth.auth.verify_id_token'),
        "rag": patch('api.websocket.query_rag_system', new_callable=AsyncMock),
        "tts": patch('api.websocket.generate_and_stream_tts', new_callable=AsyncMock),
        "setup_deepgram": patch('api.websocket.setup_deepgram_connection', new_callable=AsyncMock),
        "update_hist": patch('api.websocket.update_conversation_history', new_callable=AsyncMock),
        "get_hist": patch('api.websocket.get_conversation_history', new_callable=AsyncMock),
    }
    mocks = {name: p.start() for name, p in patches.items()}

    # --- Configure Dependency Overrides & Mocks ---
    mocks["firebase"].return_value = {
        'uid': mock_user_instance.firebase_uid,
        'email': mock_user_instance.email
    }

    # Create mocks needed for the overridden dependency
    stt_connection_mock = mock_deepgram_stt_fixture()
    mock_dg_client_instance = MagicMock(spec=DeepgramClient)
    # Configure the client instance mock as before
    mock_v1_interface = MagicMock(return_value=stt_connection_mock)
    mock_asyncwebsocket = MagicMock()
    mock_asyncwebsocket.v = mock_v1_interface
    mock_listen = MagicMock()
    mock_listen.asyncwebsocket = mock_asyncwebsocket
    mock_dg_client_instance.listen = mock_listen

    # ---> OVERRIDE the dependency directly <---
    original_get_dg_client = app.dependency_overrides.get(
        get_deepgram_live_client)
    # Create a specific mock JUST for the override
    mock_get_dg_client_override = MagicMock(
        return_value=mock_dg_client_instance)
    app.dependency_overrides[get_deepgram_live_client] = lambda: mock_get_dg_client_override(
    )
    # ----------------------------------------------

    mocks["rag"].return_value = "Mock LLM Response"
    # setup still returns the connection mock
    mocks["setup_deepgram"].return_value = stt_connection_mock
    mocks["get_hist"].return_value = []  # Default history is empty

    # Yield necessary mocks (including the override mock if needed for assertions)
    yield {
        "mock_session": mock_session,
        "mock_user": mock_user_instance,
        "mock_stt_connection": stt_connection_mock,
        "mock_get_dg_client_override": mock_get_dg_client_override,  # Yield the override mock
        **mocks
    }

    # --- Teardown ---
    for p in patches.values():
        p.stop()
    # Restore original dependencies
    if original_get_session:
        app.dependency_overrides[get_session] = original_get_session
    elif get_session in app.dependency_overrides:
        del app.dependency_overrides[get_session]
    if original_get_dg_client:
        app.dependency_overrides[get_deepgram_live_client] = original_get_dg_client
    elif get_deepgram_live_client in app.dependency_overrides:
        del app.dependency_overrides[get_deepgram_live_client]

    websocket_manager.active_connections = {}


# --- Helper to Simulate Deepgram Message ---
async def simulate_deepgram_transcript(mock_dg_connection, transcript: str):
    """Finds and calls the Transcript event handler."""
    handler = mock_dg_connection._callbacks.get(
        LiveTranscriptionEvents.Transcript)
    if handler:
        mock_result = MagicMock()
        mock_result.channel.alternatives = [MagicMock()]
        mock_result.channel.alternatives[0].transcript = transcript
        # Call the lambda/asyncio.create_task wrapper
        await handler(mock_result)
        await asyncio.sleep(0.01)  # Allow tasks to run
    else:
        raise Exception(
            "Transcript handler not registered on mock Deepgram connection")


# --- Test Cases ---

@pytest.mark.asyncio
async def test_websocket_auth_success(auto_mock_external_services):
    """Tests successful WebSocket connection and authentication, including DG start."""
    token = "valid-ws-token"
    MockSetupDeepgram = auto_mock_external_services["setup_deepgram"]
    MockGetDeepgramClient = auto_mock_external_services["mock_get_dg_client_override"]

    with client.websocket_connect(f"/ws?token={token}") as websocket:
        await asyncio.sleep(0.2)  # Keep increased sleep

        # Assert that the dependency for the client was called
        MockGetDeepgramClient.assert_called()
        # Assert that the setup function was awaited
        MockSetupDeepgram.assert_awaited_once()
        # Assert connection was added to manager INSIDE the block
        assert len(websocket_manager.active_connections) == 1

    # Assert connection is removed AFTER the block (cleanup)
    await asyncio.sleep(0.1)  # Allow time for cleanup
    assert len(websocket_manager.active_connections) == 0


@pytest.mark.asyncio
async def test_websocket_auth_fail_no_token():
    """Tests WebSocket connection failure when token is missing."""
    # Don't expect WebSocketDisconnect exception here, check received data
    # with pytest.raises(WebSocketDisconnect) as exc_info:
    try:
        with client.websocket_connect("/ws") as websocket:
            # Connection will likely accept then immediately close with error
            # Try receiving data, expect an error or closure
            received = websocket.receive_json()
            # If it doesn't raise, check the error message
            assert received.get("type") == "error"
            assert "token missing" in received.get("message", "").lower()
    except WebSocketDisconnect as e:
        # If disconnect IS raised by receive_json, check code/reason
        assert e.code == 4001
        assert "Token missing" in e.reason
    # else:
    #     # If no exception and no error JSON, fail
    #     pytest.fail("Expected WebSocketDisconnect or error message, got neither.")


@pytest.mark.asyncio
async def test_websocket_auth_fail_invalid_token():
    """Tests WebSocket connection failure with invalid token (auth dependency raises exception)."""
    async def raise_auth_error(*args, **kwargs):
        # Simulate the auth failure
        raise HTTPException(status_code=401, detail="Test Invalid Token")

    # Override the actual verification function for this test
    original_verify = app.dependency_overrides.get(
        verify_token_and_get_user_info)
    app.dependency_overrides[verify_token_and_get_user_info] = lambda: raise_auth_error

    try:
        with client.websocket_connect("/ws?token=invalid-token") as websocket:
            # Expect immediate closure or error message
            received = websocket.receive_json()
            assert received.get("type") == "error"
            assert "authentication failed" in received.get(
                "message", "").lower()
            # Optional: check detail if needed
            # assert "test invalid token" in received.get("message", "").lower()
    except WebSocketDisconnect as e:
        assert e.code == 4003
        assert "Authentication Failed" in e.reason
    # else:
    #     pytest.fail("Expected WebSocketDisconnect or error message, got neither.")
    finally:
        # Restore original dependency override
        if original_verify:
            app.dependency_overrides[verify_token_and_get_user_info] = original_verify
        elif verify_token_and_get_user_info in app.dependency_overrides:
            del app.dependency_overrides[verify_token_and_get_user_info]


@pytest.mark.asyncio
async def test_websocket_receive_audio_forward_to_deepgram(auto_mock_external_services):
    """Tests that connection establishes and STT connection is ready and audio is forwarded."""
    MockSetupDeepgram = auto_mock_external_services["setup_deepgram"]
    mock_stt_conn = auto_mock_external_services["mock_stt_connection"]
    token = "valid-ws-token"

    with client.websocket_connect(f"/ws?token={token}") as websocket:
        await asyncio.sleep(0.1)  # Allow connection setup
        MockSetupDeepgram.assert_awaited_once()
        assert len(websocket_manager.active_connections) == 1

        # ---> Send bytes IMMEDIATELY after setup (NO await) <---
        websocket.send_bytes(b'audio_chunk')
        await asyncio.sleep(0.05)  # Allow send to process on server side

        # Assert that the send method on the *mocked connection* was called
        dg_conn_mock = MockSetupDeepgram.return_value
        dg_conn_mock.send.assert_awaited_with(b'audio_chunk')

    # Assert connection is removed AFTER the block (cleanup)
    await asyncio.sleep(0.1)  # Allow time for cleanup
    assert len(websocket_manager.active_connections) == 0


@pytest.mark.asyncio
async def test_websocket_deepgram_transcript_flow(auto_mock_external_services):
    """Tests the flow: Deepgram transcript -> RAG -> TTS."""
    MockSetupDeepgram = auto_mock_external_services["setup_deepgram"]
    MockQueryRag = auto_mock_external_services["rag"]
    MockGenerateTts = auto_mock_external_services["tts"]
    mock_update_hist = auto_mock_external_services["update_hist"]
    mock_get_hist = auto_mock_external_services["get_hist"]
    mock_session = auto_mock_external_services["mock_session"]
    mock_user = auto_mock_external_services["mock_user"]
    token = "valid-ws-token"

    with client.websocket_connect(f"/ws?token={token}") as websocket:
        await asyncio.sleep(0.1)  # Allow connection setup
        MockSetupDeepgram.assert_awaited_once()
        assert len(websocket_manager.active_connections) == 1

        # Get the client_info that was passed to the mocked setup_deepgram
        client_info_arg = MockSetupDeepgram.call_args[0][0]
        # Ensure it has the websocket object
        client_info_arg.websocket = websocket
        # Ensure it has user and session IDs (might be set in mock setup or here)
        client_info_arg.user_id = client_info_arg.user_id or mock_user.id
        client_info_arg.session_id = client_info_arg.session_id or uuid.uuid4()

        # --- Simulate the transcript directly calling the handler ---
        # Import the handler function
        from api.websocket import handle_deepgram_transcript

        # Create the mock transcript result
        mock_transcript_result = MagicMock()
        mock_transcript_result.is_final = True
        mock_transcript_result.speech_final = True
        mock_transcript_result.channel.alternatives = [MagicMock()]
        mock_transcript_result.channel.alternatives[0].transcript = "User said something"

        # Call the handler function directly
        await handle_deepgram_transcript(mock_transcript_result, client_info_arg, mock_session)
        await asyncio.sleep(0.1)  # Allow handler tasks to run
        # -----------------------------------------------------------

        # Assertions remain mostly the same, but use the session_id from client_info_arg
        mock_update_hist.assert_any_call(
            mock_session, client_info_arg.session_id, {
                "type": "user", "text": "User said something", "timestamp": ANY}
        )
        mock_get_hist.assert_awaited_with(
            mock_session, client_info_arg.session_id)
        MockQueryRag.assert_awaited_with("User said something", [])
        mock_update_hist.assert_any_call(
            mock_session, client_info_arg.session_id, {"type": "assistant",
                                                       "text": "Mock LLM Response", "timestamp": ANY}
        )
        MockGenerateTts.assert_awaited_with("Mock LLM Response", websocket)

    await asyncio.sleep(0.1)
    assert len(websocket_manager.active_connections) == 0


@pytest.mark.asyncio
async def test_websocket_deepgram_empty_transcript(auto_mock_external_services):
    """Tests that an empty transcript from Deepgram is ignored."""
    MockSetupDeepgram = auto_mock_external_services["setup_deepgram"]
    MockQueryRag = auto_mock_external_services["rag"]
    MockGenerateTts = auto_mock_external_services["tts"]
    mock_update_hist = auto_mock_external_services["update_hist"]
    mock_get_hist = auto_mock_external_services["get_hist"]
    mock_session = auto_mock_external_services["mock_session"]
    mock_user = auto_mock_external_services["mock_user"]

    with client.websocket_connect("/ws?token=valid-token") as websocket:
        await asyncio.sleep(0.1)  # Allow connection setup
        MockSetupDeepgram.assert_awaited_once()
        assert len(websocket_manager.active_connections) == 1

        # Get client_info as in previous test
        client_info_arg = MockSetupDeepgram.call_args[0][0]
        client_info_arg.websocket = websocket
        client_info_arg.user_id = client_info_arg.user_id or mock_user.id
        client_info_arg.session_id = client_info_arg.session_id or uuid.uuid4()

        # --- Simulate the empty transcript ---
        from api.websocket import handle_deepgram_transcript
        mock_empty_result = MagicMock()
        mock_empty_result.is_final = True
        mock_empty_result.speech_final = True
        mock_empty_result.channel.alternatives = [MagicMock()]
        mock_empty_result.channel.alternatives[0].transcript = ""

        # Call the handler directly
        await handle_deepgram_transcript(mock_empty_result, client_info_arg, mock_session)
        await asyncio.sleep(0.05)
        # ------------------------------------

        MockQueryRag.assert_not_awaited()
        MockGenerateTts.assert_not_awaited()
        mock_update_hist.assert_not_called()
        mock_get_hist.assert_not_called()

    await asyncio.sleep(0.1)
    assert len(websocket_manager.active_connections) == 0


@pytest.mark.asyncio
async def test_websocket_disconnect_cleanup(auto_mock_external_services):
    """Tests that the REAL cleanup_connection updates DB and finishes STT."""
    MockSetupDeepgram = auto_mock_external_services["setup_deepgram"]
    mock_session = auto_mock_external_services["mock_session"]
    mock_user = auto_mock_external_services["mock_user"]
    mock_stt_conn = auto_mock_external_services["mock_stt_connection"]

    test_session_id = uuid.uuid4()  # Define the ID we expect
    mock_call_session_obj = CallSession(
        id=test_session_id,
        user_id=mock_user.id,
        status=CallStatus.ACTIVE,
        conversation_history=[],
        start_time=datetime.now(timezone.utc) - timedelta(minutes=1)
    )
    mock_session.get.return_value = mock_call_session_obj

    # ---> Patch create_call_session just for this test <---
    with patch('api.websocket.create_call_session', new_callable=AsyncMock, return_value=test_session_id):
        with client.websocket_connect("/ws?token=valid-token") as websocket:
            await asyncio.sleep(0.1)
            MockSetupDeepgram.assert_awaited_once()
            assert len(websocket_manager.active_connections) == 1
            MockSetupDeepgram.return_value = mock_stt_conn
    # ------------------------------------------------------

    # --- Assertions AFTER disconnect (where real cleanup runs) ---
    await asyncio.sleep(0.1)
    assert len(websocket_manager.active_connections) == 0
    mock_stt_conn.finish.assert_awaited_once()
    mock_session.get.assert_called_once_with(
        CallSession, test_session_id)  # Should now match
    assert mock_call_session_obj.status == CallStatus.ENDED
    assert mock_call_session_obj.end_time is not None
    # Check that add was called with the session object AT SOME POINT
    mock_session.add.assert_any_call(mock_call_session_obj)
