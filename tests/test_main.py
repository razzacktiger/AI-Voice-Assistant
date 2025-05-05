# Tests for general HTTP API endpoints

import pytest
import os  # Add os import for file cleanup
import uuid  # Add uuid import
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import status, HTTPException
from fastapi.testclient import TestClient
from datetime import datetime, timezone, timedelta

# --- Adjust imports for new structure ---
# We need the app from api.main and dependencies/models
from api.main import app  # Get app from the new main module
from api.models import User, CallSession  # Import models
# Import AnalyticsReport from main where it's defined
from api.main import AnalyticsReport
# Use the new auth functions
from api.auth import get_current_active_user, verify_token_and_get_user_info
from api.database import get_session  # Import session dependency

# Use TestClient with the app from api.main
# Fixtures (e.g., mock_db_session) should still be available from conftest.py
client = TestClient(app)

# --- Test Cases for / (root) ---


def test_read_root():
    """Tests the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the AI Voice Assistant API"}

# --- Test Cases for /users/me ---


@pytest.mark.asyncio
async def test_read_users_me_success():
    """Tests successful retrieval of current user details."""
    # The endpoint now depends on get_current_active_user
    # We need to mock that dependency
    firebase_uid = "test_uid_users_me"
    db_user_id = uuid.uuid4()
    email = "test.user@example.com"
    now = datetime.now(timezone.utc)
    mock_user = User(id=db_user_id, firebase_uid=firebase_uid,
                     email=email, created_at=now)

    # Override the final dependency to return the mock user
    # This bypasses both token verification and DB lookup for this test
    original_override = app.dependency_overrides.get(get_current_active_user)
    app.dependency_overrides[get_current_active_user] = lambda: mock_user

    # No token needed if dependency overridden
    response = client.get("/users/me")

    assert response.status_code == 200
    data = response.json()
    assert data["firebase_uid"] == firebase_uid
    assert data["email"] == email
    assert data["id"] == str(db_user_id)  # Compare as string

    # Clean up the override
    if original_override:
        app.dependency_overrides[get_current_active_user] = original_override
    else:
        del app.dependency_overrides[get_current_active_user]


@pytest.mark.asyncio
async def test_read_users_me_auth_fail():
    """Tests /users/me failure when auth fails."""
    # Simulate auth failure by mocking the *first* dependency in the chain
    original_verify = app.dependency_overrides.get(
        verify_token_and_get_user_info)

    def raise_auth_exception():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Simulated token verification failed",
            headers={"WWW-Authenticate": "Bearer"},
        )
    app.dependency_overrides[verify_token_and_get_user_info] = raise_auth_exception

    # The client still needs to send *something* that looks like a token
    response = client.get(
        "/users/me", headers={"Authorization": "Bearer invalid-token"})

    assert response.status_code == 401
    assert "Simulated token verification failed" in response.text

    # Clean up the override
    if original_verify:
        app.dependency_overrides[verify_token_and_get_user_info] = original_verify
    else:
        del app.dependency_overrides[verify_token_and_get_user_info]

# --- Test Cases for /api/analytics/report (Stub Implementation) ---


@pytest.mark.asyncio
async def test_get_analytics_report_success(mock_db_session):
    """Tests successful retrieval of the stub analytics report."""
    mock_session, _ = mock_db_session  # Get the mock session
    # Setup mock user data
    firebase_uid = "test_uid_analytics"
    db_user_id = uuid.uuid4()
    email = "analytics.user@example.com"
    now = datetime.now(timezone.utc)
    mock_user = User(id=db_user_id, firebase_uid=firebase_uid,
                     email=email, created_at=now)

    # Override the user dependency & session dependency
    original_get_user = app.dependency_overrides.get(get_current_active_user)
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    original_get_session = app.dependency_overrides.get(get_session)
    app.dependency_overrides[get_session] = lambda: mock_session

    # --- Configure Mocks: Chain exec -> scalar_one / all ---
    # Mock the session data
    session1_start = now - timedelta(minutes=10)
    session1_end = now - timedelta(minutes=5)
    session2_start = now - timedelta(minutes=4)
    session2_end = now - timedelta(minutes=2)
    mock_sessions_data = [
        CallSession(id=uuid.uuid4(), user_id=db_user_id,
                    start_time=session1_start, end_time=session1_end),
        CallSession(id=uuid.uuid4(), user_id=db_user_id,
                    start_time=session2_start, end_time=session2_end),
        CallSession(id=uuid.uuid4(), user_id=db_user_id,
                    start_time=now - timedelta(minutes=1))  # Active session
    ]

    # Mock the chain for the count query
    mock_count_query_result = MagicMock()
    mock_count_query_result.scalar_one.return_value = 5  # The final count

    # Mock the chain for the session list query
    mock_list_query_result = MagicMock()
    mock_list_query_result.all.return_value = mock_sessions_data  # The final list

    # Make session.exec return the correct mock result based on the call
    # Note: This assumes the count query happens first!
    mock_session.exec.side_effect = [
        mock_count_query_result, mock_list_query_result]
    # Clear any default return value
    mock_session.exec.return_value = None

    # No token needed if dependency overridden
    response = client.get("/api/analytics/report")

    assert response.status_code == 200
    data = response.json()
    assert data["total_calls"] == 5  # Check against the scalar_one value
    expected_avg_duration = ((session1_end - session1_start).total_seconds() +
                             (session2_end - session2_start).total_seconds()) / 2
    assert data["average_duration_seconds"] == pytest.approx(
        expected_avg_duration)

    # Verify exec calls were made
    assert mock_session.exec.call_count == 2
    # Verify the intermediate methods were called
    mock_count_query_result.scalar_one.assert_called_once()
    mock_list_query_result.all.assert_called_once()

    # Clean up overrides
    if original_get_user:
        app.dependency_overrides[get_current_active_user] = original_get_user
    else:
        del app.dependency_overrides[get_current_active_user]
    if original_get_session:  # Restore original get_session override if existed
        app.dependency_overrides[get_session] = original_get_session
    elif get_session in app.dependency_overrides:  # Otherwise remove override
        del app.dependency_overrides[get_session]

# --- Test Cases for /api/admin/export (Stub Implementation) ---


@pytest.mark.asyncio
async def test_export_analytics_csv_success(mock_db_session):
    """Tests successful retrieval of the CSV export using StreamingResponse."""
    mock_session, _ = mock_db_session  # Get the mock session
    # Setup mock user data
    firebase_uid = "test_uid_export"
    db_user_id = uuid.uuid4()
    email = "export.user@example.com"
    now = datetime.now(timezone.utc)
    mock_user = User(id=db_user_id, firebase_uid=firebase_uid,
                     email=email, created_at=now)

    # Override dependencies
    original_get_user = app.dependency_overrides.get(get_current_active_user)
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    original_get_session = app.dependency_overrides.get(get_session)
    app.dependency_overrides[get_session] = lambda: mock_session

    # --- Configure Mock: Chain exec -> all ---
    # Mock the session data
    session1_start = now - timedelta(minutes=5)
    session1_end = now - timedelta(minutes=1)
    session2_start = now - timedelta(minutes=15)
    session2_end = now - timedelta(minutes=10)
    mock_csv_sessions = [
        CallSession(id=uuid.uuid4(), user_id=db_user_id,
                    start_time=session1_start, end_time=session1_end),
        CallSession(id=uuid.uuid4(), user_id=db_user_id,
                    start_time=session2_start, end_time=session2_end)
    ]

    # Mock the chain for the list query
    mock_list_query_result_csv = MagicMock()
    mock_list_query_result_csv.all.return_value = mock_csv_sessions  # The final list

    # Configure session.exec to return the mock result
    mock_session.exec.return_value = mock_list_query_result_csv
    mock_session.exec.side_effect = None  # Clear side_effect

    response = client.get("/api/admin/export")  # No token needed

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/csv; charset=utf-8"
    # Check Content-Disposition exactly
    assert response.headers["content-disposition"] == 'attachment; filename=analytics_export.csv'

    # Check content directly from the StreamingResponse text
    csv_content = response.text
    assert "session_id,user_email,start_time,end_time,duration_seconds" in csv_content
    # Check if data rows corresponding to mock_csv_sessions are present
    assert str(mock_csv_sessions[0].id) in csv_content
    assert email in csv_content
    assert str(mock_csv_sessions[0].start_time) in csv_content
    assert str(mock_csv_sessions[0].end_time) in csv_content
    assert str((session1_end - session1_start).total_seconds()) in csv_content
    assert str(mock_csv_sessions[1].id) in csv_content
    assert str((session2_end - session2_start).total_seconds()) in csv_content

    # Verify exec call and chained call
    mock_session.exec.assert_called_once()
    mock_list_query_result_csv.all.assert_called_once()

    # Clean up overrides
    if original_get_user:
        app.dependency_overrides[get_current_active_user] = original_get_user
    else:
        del app.dependency_overrides[get_current_active_user]
    if original_get_session:
        app.dependency_overrides[get_session] = original_get_session
    elif get_session in app.dependency_overrides:
        del app.dependency_overrides[get_session]

    # No need to clean up dummy file anymore
    # if os.path.exists("temp_export.csv"):
    #     os.remove("temp_export.csv")

# Note: The old tests for /api/call/start are removed as that endpoint
# is no longer part of the HTTP API (it's handled by WebSocket connection now).
