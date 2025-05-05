# tests/test_admin.py

import pytest
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch, ANY

from fastapi.testclient import TestClient
from sqlmodel import Session

from api.main import app
from api.models import User, LogEntry, LogLevel  # Import LogLevel
from api.database import get_session
from api.auth import get_current_active_user

# Mock admin user for dependency override
mock_admin_user = User(
    id=uuid.uuid4(),
    firebase_uid="mock_admin_user_uid",
    email="admin@test.com",
    # TODO: Add an is_admin flag to User model later for proper checks
    created_at=datetime.now(timezone.utc)
)

# Fixture to override auth and db dependencies for admin tests


@pytest.fixture(scope="function", autouse=True)
def override_admin_dependencies():
    original_get_session = app.dependency_overrides.get(get_session)
    original_get_user = app.dependency_overrides.get(get_current_active_user)

    # Create a NEW mock session for each test function
    mock_session = MagicMock(spec=Session)
    app.dependency_overrides[get_session] = lambda: mock_session
    app.dependency_overrides[get_current_active_user] = lambda: mock_admin_user

    yield mock_session

    # Teardown remains the same
    if original_get_session:
        app.dependency_overrides[get_session] = original_get_session
    else:
        del app.dependency_overrides[get_session]
    if original_get_user:
        app.dependency_overrides[get_current_active_user] = original_get_user
    else:
        del app.dependency_overrides[get_current_active_user]

# --- Test Data ---


def create_mock_log_entries():
    user1_id = uuid.uuid4()
    session1_id = uuid.uuid4()
    now = datetime.now(timezone.utc)
    return [
        LogEntry(id=1, timestamp=now - timedelta(minutes=10), level=LogLevel.INFO,
                 component="websocket", event="connect", user_id=user1_id, session_id=session1_id),
        LogEntry(id=2, timestamp=now - timedelta(minutes=9), level=LogLevel.DEBUG,
                 component="rag", event="query_start", user_id=user1_id, session_id=session1_id),
        LogEntry(id=3, timestamp=now - timedelta(minutes=8), level=LogLevel.ERROR, component="stt_handler",
                 event="stt_error", user_id=user1_id, session_id=session1_id, message="Deepgram failed"),
        LogEntry(id=4, timestamp=now - timedelta(minutes=5), level=LogLevel.WARNING,
                 component="auth", event="auth_failed", details={"reason": "Bad token"}),  # No user/session
    ]

# --- Tests ---


def test_get_logs_no_filters(override_admin_dependencies):
    """Tests the /admin/logs endpoint with no filters."""
    mock_session = override_admin_dependencies
    mock_session.reset_mock()
    mock_logs = create_mock_log_entries()
    # Mock exec() to return an object where calling .all() returns the list of LogEntry instances
    mock_session.exec.return_value.all.return_value = mock_logs

    client = TestClient(app)
    response = client.get("/admin/logs")

    # Assertions remain the same for now
    assert response.status_code == 200
    data = response.json()
    assert len(data) == len(mock_logs)
    assert data[0]["id"] == mock_logs[0].id
    assert data[0]["component"] == "websocket"
    # Check that exec was called
    mock_session.exec.assert_called_once()


def test_get_logs_with_filters(override_admin_dependencies):
    """Tests the /admin/logs endpoint with various filters."""
    mock_session = override_admin_dependencies
    mock_session.reset_mock()
    filtered_logs = [create_mock_log_entries()[2]]
    # Mock exec() to return an object where calling .all() returns the filtered list
    mock_session.exec.return_value.all.return_value = filtered_logs

    client = TestClient(app)
    response = client.get(
        "/admin/logs?level=ERROR&component=stt_handler&limit=10")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["level"] == "ERROR"
    assert data[0]["component"] == "stt_handler"

    # Check that the executed statement included WHERE clauses (by string inspection)
    call_args, call_kwargs = mock_session.exec.call_args
    statement = call_args[0]
    assert "WHERE logentry.level = :level_1" in str(statement)
    assert "AND logentry.component = :component_1" in str(statement)
    assert "LIMIT :param_1" in str(statement)
    mock_session.exec.assert_called_once()


def test_get_logs_pagination(override_admin_dependencies):
    """Tests pagination parameters (skip, limit) for /admin/logs."""
    mock_session = override_admin_dependencies
    mock_session.reset_mock()
    paginated_logs = [create_mock_log_entries()[1]]
    # Mock exec() to return an object where calling .all() returns the paginated list
    mock_session.exec.return_value.all.return_value = paginated_logs

    client = TestClient(app)
    response = client.get("/admin/logs?skip=1&limit=1")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["id"] == 2  # Should be the second log entry

    # Check limit and offset were likely applied (via string inspection)
    call_args, call_kwargs = mock_session.exec.call_args
    statement = call_args[0]
    assert "LIMIT :param_1" in str(statement)
    assert "OFFSET :param_2" in str(statement)
    mock_session.exec.assert_called_once()
