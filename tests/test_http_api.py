# Tests for general HTTP API endpoints

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import status, HTTPException
from fastapi.testclient import TestClient
from datetime import datetime, timezone

# Import app and models/dependencies needed
from api import app, User, CallSession, get_current_user

# Note: Fixtures mock_db_session and auto_override_dependencies
# are automatically available from tests/conftest.py
client = TestClient(app)

# --- Test Cases for /api/call/start ---


@pytest.mark.asyncio
async def test_start_call_success(mock_db_session, auto_override_dependencies):
    """Tests successful call session start via /api/call/start."""
    mock_session = mock_db_session[0]
    firebase_uid = "test_uid_call_start"
    db_user_id = 6
    token = "valid_call_start_token"
    now = datetime.now(timezone.utc)
    mock_user = User(id=db_user_id, firebase_uid=firebase_uid,
                     created_at=now, email=None)

    app.dependency_overrides[get_current_user] = lambda: mock_user

    def refresh_side_effect(obj):
        if isinstance(obj, CallSession):
            obj.id = 99  # Assign a dummy session ID
    mock_session.refresh.side_effect = refresh_side_effect

    response = client.post(
        "/api/call/start", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == 99
    assert data["message"] == "Call session started"

    mock_session.add.assert_called_once()
    added_object = mock_session.add.call_args[0][0]
    assert isinstance(added_object, CallSession)
    assert added_object.user_id == db_user_id
    mock_session.commit.assert_called_once()
    mock_session.refresh.assert_called_once_with(added_object)

    del app.dependency_overrides[get_current_user]


@pytest.mark.asyncio
async def test_start_call_auth_fail(auto_override_dependencies):
    """Tests /api/call/start failure when auth fails."""
    token = "invalid_call_start_token"

    def raise_invalid_token():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired Firebase token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    app.dependency_overrides[get_current_user] = raise_invalid_token

    response = client.post(
        "/api/call/start", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 401

    del app.dependency_overrides[get_current_user]

# TODO: Add tests for /api/rag/query (once implemented)
# TODO: Add tests for /api/analytics/report (once implemented)
# TODO: Add tests for /api/admin/export (once implemented)
