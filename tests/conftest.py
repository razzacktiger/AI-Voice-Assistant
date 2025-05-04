# Shared fixtures for Pytest

import pytest
import pytest_asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from sqlmodel import Session

# Assuming tests are run from project root
# Need to import app and dependencies for overriding
from api import app, get_session, User  # Import necessary items

# Fixture moved from test_api.py


@pytest.fixture
def mock_db_session():
    """Provides a mock database session."""
    mock_session = MagicMock(spec=Session)
    mock_exec = MagicMock()
    mock_session.exec.return_value = mock_exec
    # Add mock methods needed
    mock_session.add = MagicMock()
    mock_session.commit = MagicMock()
    mock_session.refresh = MagicMock()
    return mock_session, mock_exec

# Fixture moved from test_api.py


@pytest_asyncio.fixture(autouse=True)
def auto_override_dependencies(mock_db_session):
    """Overrides dependencies globally for tests. Specific tests can add more."""
    mock_session, _ = mock_db_session
    original_get_session = app.dependency_overrides.get(get_session)

    # Mock verify_firebase_token globally first
    with patch('api.verify_firebase_token', new_callable=AsyncMock) as mock_verify:
        # Override get_session while verify_firebase_token is patched
        app.dependency_overrides[get_session] = lambda: mock_session

        yield mock_verify  # Yield the verify mock for tests that need it

    # --- Cleanup ---
    # Restore original get_session override or clear
    if original_get_session:
        app.dependency_overrides[get_session] = original_get_session
    elif get_session in app.dependency_overrides:
        del app.dependency_overrides[get_session]

    # Clear other overrides that might have been set by individual tests
    # Need to know which specific dependencies might be overridden
    # Example: Clear get_current_user if used
    # from api import get_current_user # Import if needed
    # if get_current_user in app.dependency_overrides:
    #    del app.dependency_overrides[get_current_user]
