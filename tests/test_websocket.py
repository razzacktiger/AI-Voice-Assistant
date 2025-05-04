# Tests for WebSocket connection, auth, and core flow

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
from fastapi import status, WebSocketDisconnect
from fastapi.testclient import TestClient
from datetime import datetime, timezone
from deepgram import LiveOptions

# Import app and models/dependencies needed
from api import app, User, CallSession

# Note: Fixtures mock_db_session and auto_override_dependencies
# are automatically available from tests/conftest.py
client = TestClient(app)


# --- Test Cases for WebSocket Endpoint (/ws/{session_id}) ---

@pytest.mark.asyncio
async def test_websocket_connect_success(
    mock_db_session, auto_override_dependencies
):
    """Tests successful WebSocket connection and validation."""
    mock_session, mock_exec = mock_db_session
    mock_verify_token = auto_override_dependencies
    firebase_uid = "test_uid_123"
    user_email = "test@example.com"
    db_user_id = 1
    session_id = 42
    mock_verify_token.return_value = {"uid": firebase_uid, "email": user_email}
    now = datetime.now(timezone.utc)
    mock_user = User(id=db_user_id, firebase_uid=firebase_uid,
                     email=user_email, created_at=now)
    mock_call_session = CallSession(id=session_id, user_id=db_user_id)

    def exec_side_effect(statement):
        query_str = str(statement).lower()
        if "user" in query_str and "firebase_uid" in query_str:
            mock_exec.first.return_value = mock_user
        elif "callsession" in query_str and "id" in query_str:
            mock_exec.first.return_value = mock_call_session
        else:
            mock_exec.first.return_value = None
        return mock_exec
    mock_session.exec.side_effect = exec_side_effect

    try:
        with client.websocket_connect(f"/ws/{session_id}?token=valid_token") as websocket:
            assert websocket
    except Exception as e:
        print(f"Exception during websocket_connect: {e}")
        raise


@pytest.mark.asyncio
async def test_websocket_connect_fail_no_token(mock_db_session, auto_override_dependencies):
    """Tests WebSocket connection failure when token is missing."""
    session_id = 42
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect(f"/ws/{session_id}") as websocket:
            pass
    assert exc_info.value.code == status.WS_1008_POLICY_VIOLATION
    assert exc_info.value.reason == "Authentication failed"


@pytest.mark.asyncio
async def test_websocket_connect_fail_invalid_token(
    mock_db_session, auto_override_dependencies
):
    """Tests WebSocket connection failure with an invalid token."""
    mock_verify_token = auto_override_dependencies
    mock_verify_token.return_value = None
    session_id = 42
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect(f"/ws/{session_id}?token=invalid_token") as websocket:
            pass
    assert exc_info.value.code == status.WS_1008_POLICY_VIOLATION
    assert exc_info.value.reason == "Authentication failed"


@pytest.mark.asyncio
async def test_websocket_connect_fail_user_not_in_db(
    mock_db_session, auto_override_dependencies
):
    """Tests WebSocket connection failure if user (from valid token) is not in DB."""
    mock_session, mock_exec = mock_db_session
    mock_verify_token = auto_override_dependencies
    mock_verify_token.return_value = {
        "uid": "test_uid_no_db", "email": "no@db.com"}

    def exec_side_effect(statement):
        query_str = str(statement).lower()
        if "user" in query_str and "firebase_uid" in query_str:
            mock_exec.first.return_value = None
        mock_other = MagicMock()
        mock_other.first.return_value = None
        return mock_other
    mock_session.exec.side_effect = exec_side_effect

    session_id = 42
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect(f"/ws/{session_id}?token=valid_token_no_db") as websocket:
            pass
    assert exc_info.value.code == status.WS_1011_INTERNAL_ERROR
    assert exc_info.value.reason == "User DB record not found"


@pytest.mark.asyncio
async def test_websocket_connect_fail_session_not_found(
    mock_db_session, auto_override_dependencies
):
    """Tests WebSocket connection failure if session ID is not found."""
    mock_session, mock_exec = mock_db_session
    mock_verify_token = auto_override_dependencies
    firebase_uid = "test_uid_123"
    db_user_id = 1
    now = datetime.now(timezone.utc)
    mock_user = User(id=db_user_id, firebase_uid=firebase_uid, created_at=now)
    mock_verify_token.return_value = {"uid": firebase_uid}

    def exec_side_effect(statement):
        query_str = str(statement).lower()
        if "user" in query_str and "firebase_uid" in query_str:
            mock_exec.first.return_value = mock_user
        elif "callsession" in query_str and "id" in query_str:
            mock_exec.first.return_value = None
        else:
            mock_exec.first.return_value = None
        return mock_exec
    mock_session.exec.side_effect = exec_side_effect

    session_id = 999
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect(f"/ws/{session_id}?token=valid_token") as websocket:
            pass
    assert exc_info.value.code == status.WS_1008_POLICY_VIOLATION
    assert exc_info.value.reason == "Call session not found"
    mock_verify_token.assert_awaited_once()
    assert mock_session.exec.call_count >= 1


@pytest.mark.asyncio
async def test_websocket_connect_fail_session_wrong_user(
    mock_db_session, auto_override_dependencies
):
    """Tests WebSocket connection failure if session belongs to another user."""
    mock_session, mock_exec = mock_db_session
    mock_verify_token = auto_override_dependencies
    firebase_uid = "test_uid_123"
    db_user_id = 1
    other_user_id = 2
    now = datetime.now(timezone.utc)
    mock_user = User(id=db_user_id, firebase_uid=firebase_uid, created_at=now)
    mock_call_session = CallSession(id=42, user_id=other_user_id)
    mock_verify_token.return_value = {"uid": firebase_uid}

    def exec_side_effect(statement):
        query_str = str(statement).lower()
        if "user" in query_str and "firebase_uid" in query_str:
            mock_exec.first.return_value = mock_user
        elif "callsession" in query_str and "id" in query_str:
            mock_exec.first.return_value = mock_call_session
        else:
            mock_exec.first.return_value = None
        return mock_exec
    mock_session.exec.side_effect = exec_side_effect

    session_id = 42
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect(f"/ws/{session_id}?token=valid_token") as websocket:
            pass
    assert exc_info.value.code == status.WS_1008_POLICY_VIOLATION
    assert exc_info.value.reason == "Session access forbidden"
    mock_verify_token.assert_awaited_once()
    assert mock_session.exec.call_count >= 1


@pytest.mark.asyncio
async def test_websocket_connect_fail_invalid_session_id_format(
    mock_db_session, auto_override_dependencies
):
    """Tests WebSocket connection failure with non-integer session ID."""
    mock_verify_token = auto_override_dependencies
    mock_verify_token.return_value = {"uid": "test_uid_123"}
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect(f"/ws/invalid-session-id?token=valid_token") as websocket:
            pass
    assert exc_info.value.code == status.WS_1003_UNSUPPORTED_DATA
    assert exc_info.value.reason == "Invalid session ID format"

# --- Test Cases for Deepgram Integration in WebSocket ---


@patch('api.deepgram', create=True)
@pytest.mark.asyncio
async def test_websocket_deepgram_flow(
    mock_dg_client, mock_db_session, auto_override_dependencies
):
    """Tests Deepgram connection setup, simulated forwarding, and cleanup."""
    mock_dg_connection = AsyncMock()
    mock_dg_connection.on = MagicMock()
    mock_dg_connection.send = AsyncMock()
    mock_dg_connection.finish = AsyncMock()
    mock_stream_method = AsyncMock(return_value=mock_dg_connection)
    mock_v1_object = MagicMock()
    mock_v1_object.stream = mock_stream_method
    mock_dg_client.listen = MagicMock()
    mock_dg_client.listen.asyncwebsocket = MagicMock()
    mock_dg_client.listen.asyncwebsocket.v = MagicMock(
        return_value=mock_v1_object)

    firebase_uid = "test_dg_user"
    db_user_id = 10
    session_id = 50
    now = datetime.now(timezone.utc)
    mock_user = User(id=db_user_id, firebase_uid=firebase_uid, created_at=now)
    mock_call_session = CallSession(id=session_id, user_id=db_user_id)
    mock_session = mock_db_session[0]
    mock_exec = mock_db_session[1]

    def exec_side_effect(statement):
        query_str = str(statement).lower()
        if "user" in query_str and "firebase_uid" in query_str:
            mock_exec.first.return_value = mock_user
        elif "callsession" in query_str and "id" in query_str:
            mock_exec.first.return_value = mock_call_session
        else:
            mock_exec.first.return_value = None
        return mock_exec
    mock_session.exec.side_effect = exec_side_effect
    mock_verify_token = auto_override_dependencies
    mock_verify_token.return_value = {"uid": firebase_uid}

    audio_data = b'\x01\x02\x03\x04'
    try:
        with client.websocket_connect(f"/ws/{session_id}?token=valid_dg_token") as websocket:
            await asyncio.sleep(0.2)
            mock_dg_client.listen.asyncwebsocket.v.assert_called_once_with("1")
            mock_v1_object.stream.assert_called_once()
            call_args, call_kwargs = mock_v1_object.stream.call_args
            assert isinstance(call_args[0], LiveOptions)
            mock_dg_connection.on.assert_called()
            assert mock_dg_connection.on.call_count >= 7
            await mock_dg_connection.send(audio_data)

    except WebSocketDisconnect as e:
        pytest.fail(
            f"WebSocket disconnected unexpectedly: {e.code} / {e.reason}")
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")

    await asyncio.sleep(0.1)
    mock_dg_connection.send.assert_awaited_once_with(audio_data)
    mock_dg_connection.finish.assert_awaited_once()
