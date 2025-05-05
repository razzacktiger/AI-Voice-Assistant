# tests/test_logging_utils.py

import pytest
from unittest.mock import MagicMock, call
from datetime import datetime, timedelta, timezone
from uuid import uuid4

# Import the function to test and supporting models/enums
from api.logging_utils import add_log_entry, delete_old_logs
from api.models import LogEntry, LogLevel

# Note: We assume the mock_db_session fixture from conftest.py is available
# It provides (mock_session, mock_exec)


# Use freeze_time if installed, otherwise calculate manually
@pytest.mark.freeze_time("2024-05-05 12:00:00")
def test_delete_old_logs(mock_db_session):
    """Tests that delete_old_logs correctly removes logs older than the cutoff."""
    mock_session, mock_exec = mock_db_session

    # --- Arrange ---
    user_id = uuid4()
    session_id = uuid4()
    days_to_keep = 30
    now = datetime.now(timezone.utc)  # Reference point (2024-05-05 12:00:00)

    # Create mock LogEntry objects
    log_old = LogEntry(
        id=1,
        timestamp=now - timedelta(days=days_to_keep + 1),  # Older than cutoff
        level=LogLevel.INFO,
        event="old_event",
        user_id=user_id,
        session_id=session_id
    )
    log_recent = LogEntry(
        id=2,
        timestamp=now - timedelta(days=days_to_keep - 1),  # Newer than cutoff
        level=LogLevel.WARNING,
        event="recent_event",
        user_id=user_id,
        session_id=session_id
    )
    log_boundary = LogEntry(
        id=3,
        # Exactly on cutoff boundary (should NOT be deleted)
        timestamp=now - timedelta(days=days_to_keep),
        level=LogLevel.ERROR,
        event="boundary_event",
        user_id=user_id,
        session_id=session_id
    )

    # Configure the mock session's execute chain
    # When select(LogEntry).where(...) is executed, return ONLY the old log
    # Note: We are testing the `query-then-delete` path inside delete_old_logs
    mock_exec.all.return_value = [log_old]

    # --- Act ---
    deleted_count = delete_old_logs(db=mock_session, days_to_keep=days_to_keep)

    # --- Assert ---
    assert deleted_count == 1  # Should report 1 deleted entry

    # Verify the select statement was executed (implicitly tested by mock_exec setup)
    # We expect LogEntry.timestamp < cutoff_date
    # The check happens within the mock_db_session fixture or implicitly
    assert mock_session.exec.call_count == 1  # Verify select was run

    # Verify delete was called once with the old log entry
    mock_session.delete.assert_called_once_with(log_old)

    # Verify commit was called
    mock_session.commit.assert_called_once()

# TODO: Add tests for add_log_entry if needed (e.g., check commit/rollback behavior)
# TODO: Add test for delete_old_logs with DB error during query or delete

# tests/test_logging_utils.py
