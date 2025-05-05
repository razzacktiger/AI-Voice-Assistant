# Shared fixtures for Pytest

from api.database import get_session  # DB session dependency
from api.main import app  # App instance
import pytest
import pytest_asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from sqlmodel import Session
from dotenv import load_dotenv
import os

# Load test environment variables FIRST
# Assumes .env.test is in the project root where pytest is run
print("Loading .env.test...")
# Use find_dotenv to locate the file relative to conftest.py if needed
# from dotenv import find_dotenv
# load_dotenv(find_dotenv(".env.test"), override=True)
load_dotenv(".env.test", override=True)

# Now import app modules AFTER env vars are potentially set
# Remove top-level import of get_current_user
# from api.auth import get_current_user

# --- Fixtures ---


@pytest.fixture
def mock_db_session():
    """Provides a mock database session with mocked methods."""
    mock_session = MagicMock(spec=Session)
    mock_exec = MagicMock()
    # Configure common methods used
    mock_exec.first = MagicMock(return_value=None)
    mock_session.exec.return_value = mock_exec
    mock_session.add = MagicMock()
    mock_session.commit = MagicMock()
    mock_session.refresh = MagicMock()
    mock_session.get = MagicMock(return_value=None)  # Add get mock
    return mock_session, mock_exec  # Return session and the mock exec object


@pytest_asyncio.fixture(autouse=True)
def auto_override_db_session(mock_db_session):
    """Overrides the database session dependency globally for all tests,
       including patching where it might be used directly in WebSocket code."""
    # Remove the import of get_current_user here
    # from api.auth import get_current_user

    mock_session, _ = mock_db_session
    original_get_session = app.dependency_overrides.get(get_session)

    # Override get_session globally via FastAPI dependency overrides
    app.dependency_overrides[get_session] = lambda: mock_session
    print(f"Overriding FastAPI get_session: {id(mock_session)}")

    # --- Also patch get_session directly within the websocket module ---
    # This ensures that `next(get_session())` within the websocket code
    # also receives the mock session during testing.
    patcher = patch('api.websocket.get_session',
                    return_value=iter([mock_session]))
    mock_ws_get_session = patcher.start()
    print(f"Patching api.websocket.get_session: {id(mock_session)}")
    # ---------------------------------------------------------------

    yield mock_session  # Yield the session for tests that might need it directly

    # --- Cleanup ---
    print(f"Stopping api.websocket.get_session patch: {id(mock_session)}")
    patcher.stop()  # Stop the direct patch

    print(f"Cleaning up FastAPI get_session override: {id(mock_session)}")
    # Restore original get_session override or clear
    if original_get_session:
        app.dependency_overrides[get_session] = original_get_session
    elif get_session in app.dependency_overrides:
        del app.dependency_overrides[get_session]

    # Remove cleanup logic for get_current_user override
    # if get_current_user in app.dependency_overrides:
    #     print("Cleaning up get_current_user override")
    #     del app.dependency_overrides[get_current_user]

# Note: Removed the global patch for firebase verification.
# Individual test files/fixtures (like in test_websocket.py)
# should handle mocking external services like Firebase Auth, Deepgram, etc.,
# specific to their needs using `patch` or dedicated fixtures.
