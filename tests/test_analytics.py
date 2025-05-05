# tests/test_analytics.py

import pytest
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
import csv
import io

from fastapi.testclient import TestClient
from sqlmodel import Session

from api.main import app
from api.models import User, CallSession, CallStatus, AnalyticsReport
from api.database import get_session
from api.auth import get_current_active_user

# Mock user for dependency override
mock_analytics_user = User(
    id=uuid.uuid4(),
    firebase_uid="mock_analytics_user_uid",
    email="analytics@test.com",
    created_at=datetime.now(timezone.utc)
)

# Fixture to override auth and db dependencies


@pytest.fixture(scope="function", autouse=True)
def override_dependencies():
    original_get_session = app.dependency_overrides.get(get_session)
    original_get_user = app.dependency_overrides.get(get_current_active_user)

    # Create a NEW mock session for each test function
    mock_session = MagicMock(spec=Session)
    app.dependency_overrides[get_session] = lambda: mock_session
    app.dependency_overrides[get_current_active_user] = lambda: mock_analytics_user

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


def create_mock_call_sessions():
    user_id = uuid.uuid4()
    now = datetime.now(timezone.utc)
    return [
        CallSession(id=uuid.uuid4(), user_id=user_id, start_time=now - timedelta(minutes=10),
                    end_time=now - timedelta(minutes=8), status=CallStatus.ENDED),  # 120s
        CallSession(id=uuid.uuid4(), user_id=user_id, start_time=now - timedelta(minutes=5),
                    end_time=now - timedelta(minutes=4, seconds=30), status=CallStatus.ENDED),  # 30s
        CallSession(id=uuid.uuid4(), user_id=user_id, start_time=now -
                    timedelta(minutes=2), end_time=None, status=CallStatus.ERROR),
        CallSession(id=uuid.uuid4(), user_id=user_id, start_time=now - timedelta(minutes=1),
                    end_time=None, status=CallStatus.ACTIVE),  # Ignored in duration calc
    ]

# --- Tests ---


def test_get_analytics_summary_success(override_dependencies):
    """Tests the /analytics/summary endpoint successfully."""
    # Mock the internal report generation function directly
    mock_report = AnalyticsReport(
        total_calls=4,
        average_call_duration_seconds=75.0,
        total_error_calls=1,
        error_rate=25.0
    )
    with patch('api.analytics.get_analytics_report', return_value=mock_report) as mock_get_report:
        client = TestClient(app)
        response = client.get("/analytics/summary")

        assert response.status_code == 200
        data = response.json()
        assert data["total_calls"] == 4
        assert data["total_error_calls"] == 1
        assert data["average_call_duration_seconds"] == 75.0
        assert data["error_rate"] == 25.0
        mock_get_report.assert_called_once()  # Check the patched function was called


def test_get_analytics_summary_no_data(override_dependencies):
    """Tests the /analytics/summary endpoint when there are no calls."""
    # Mock the internal report generation function directly
    mock_report = AnalyticsReport(total_calls=0, total_error_calls=0)
    with patch('api.analytics.get_analytics_report', return_value=mock_report) as mock_get_report:
        client = TestClient(app)
        response = client.get("/analytics/summary")

        assert response.status_code == 200
        data = response.json()
        assert data["total_calls"] == 0
        assert data["total_error_calls"] == 0
        assert data["average_call_duration_seconds"] is None
        assert data["error_rate"] is None
        mock_get_report.assert_called_once()


def test_export_analytics_success(override_dependencies):
    """Tests the /analytics/export endpoint successfully generates CSV."""
    mock_session = override_dependencies
    mock_sessions = create_mock_call_sessions()

    # Mock exec().all() to return the list of CallSession instances
    mock_session.exec.return_value.all.return_value = mock_sessions

    client = TestClient(app)
    response = client.get("/analytics/export")

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/csv; charset=utf-8"
    assert "attachment; filename=analytics_export.csv" in response.headers[
        "content-disposition"]

    # Check CSV content
    content = response.content.decode("utf-8")
    reader = csv.reader(io.StringIO(content))
    rows = list(reader)

    assert len(rows) == len(mock_sessions) + 1  # Header + data rows
    assert rows[0] == ["session_id", "user_id", "start_time",
                       "end_time", "status", "duration_seconds"]
    # Check first data row (adjust indices/values based on create_mock_call_sessions)
    assert rows[1][0] == str(mock_sessions[0].id)
    assert rows[1][4] == CallStatus.ENDED.value
    assert rows[1][5] == "120.0"  # Check calculated duration
    # Check error row (duration should be empty)
    assert rows[3][0] == str(mock_sessions[2].id)
    assert rows[3][4] == CallStatus.ERROR.value
    assert rows[3][5] == ""
    # Check active row (duration should be empty)
    assert rows[4][0] == str(mock_sessions[3].id)
    assert rows[4][4] == CallStatus.ACTIVE.value
    assert rows[4][5] == ""

    # Assert exec was called once
    mock_session.exec.assert_called_once()
    # We can't easily assert the statement content without more complex mocking
