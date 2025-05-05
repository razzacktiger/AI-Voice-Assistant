# Tests for the authentication dependency function

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import status, HTTPException, Depends
from sqlmodel import Session, select
from datetime import datetime, timezone
import uuid

# Import the function to test and necessary models/helpers
from api.auth import verify_token_and_get_user_info, get_current_active_user
from api.models import User
# No longer need to import firebase_admin.auth directly for patching here
# from firebase_admin import auth

# Remove unused TestClient instantiation
# client = TestClient(app)

# --- Fixtures ---

# Mock firebase_admin dependencies


@pytest.fixture(autouse=True)
def mock_firebase_dependencies():
    """Automatically mock firebase_admin for all tests in this module."""
    # Patch the top-level firebase_admin import in the module where it's used (api.auth)
    with patch('api.auth.firebase_admin') as mock_admin, \
            patch('api.auth.auth') as mock_auth_module:

        # Configure the mock auth module (used for verify_id_token)
        mock_auth_module.verify_id_token = MagicMock()

        # Configure the mock admin module
        # Mock initialize_app to prevent actual SDK init attempts
        mock_admin.initialize_app = MagicMock()
        # Ensure _apps behaves as if initialized (or not) based on test needs
        # Default to looking like it's not initialized, relying on verify_id_token patch
        mock_admin._apps = {}
        # Route the auth attribute to our mock auth module
        mock_admin.auth = mock_auth_module

        yield mock_admin, mock_auth_module


@pytest.fixture
def mock_db():
    """Provides a mock SQLModel Session.
    Returns a tuple: (mock_session, mock_exec_result)
    """
    mock_session = MagicMock(spec=Session)
    # Mock the chain: session.exec(...).one_or_none() / .first() / .all() etc.
    mock_exec_result = MagicMock()
    # Configure default behaviors for different chain endings
    mock_exec_result.one_or_none.return_value = None
    mock_exec_result.first.return_value = None  # Alias for one_or_none often
    mock_exec_result.all.return_value = []
    mock_exec_result.scalar_one.return_value = 0  # Default count

    mock_session.exec.return_value = mock_exec_result
    # Make commit/refresh/add do nothing by default
    mock_session.commit = MagicMock()
    mock_session.refresh = MagicMock()
    mock_session.add = MagicMock()
    return mock_session, mock_exec_result

# --- Test Cases for get_current_user (assuming PYTEST_RUNNING=1) ---
# These tests focus on the DB interaction part, assuming the
# initial token verification is simulated successfully by the function itself.

# @pytest.mark.asyncio
# async def test_get_current_user_existing_user_pytest(mock_db, mock_firebase_dependencies):
#     ...

# @pytest.mark.asyncio
# async def test_get_current_user_new_user_pytest(mock_db, mock_firebase_dependencies):
#     ...

# @pytest.mark.asyncio
# async def test_get_current_user_invalid_token_pytest(mock_db, mock_firebase_dependencies):
#     ...

# @pytest.mark.asyncio
# async def test_get_current_user_db_error_on_lookup_pytest(mock_db, mock_firebase_dependencies):
#     ...

# @pytest.mark.asyncio
# async def test_get_current_user_db_error_on_create_pytest(mock_db, mock_firebase_dependencies):
#     ...

# --- Test Cases ---

# Test verify_token_and_get_user_info


@pytest.mark.asyncio
async def test_verify_token_success(mock_firebase_dependencies):
    """Tests successful token verification."""
    mock_admin, mock_auth_module = mock_firebase_dependencies
    mock_admin._apps = {'[DEFAULT]': MagicMock()}  # Simulate initialized

    token = "valid_token"
    expected_uid = "test-uid"
    expected_email = "test@example.com"
    mock_auth_module.verify_id_token.return_value = {
        'uid': expected_uid, 'email': expected_email}

    user_info = await verify_token_and_get_user_info(token=token)

    assert user_info == {"uid": expected_uid, "email": expected_email}
    mock_auth_module.verify_id_token.assert_called_once_with(token)


@pytest.mark.asyncio
async def test_verify_token_failure(mock_firebase_dependencies):
    """Tests failed token verification."""
    mock_admin, mock_auth_module = mock_firebase_dependencies
    mock_admin._apps = {'[DEFAULT]': MagicMock()}

    token = "invalid_token"
    mock_auth_module.verify_id_token.side_effect = Exception(
        "Verification failed")

    with pytest.raises(HTTPException) as exc_info:
        await verify_token_and_get_user_info(token=token)

    assert exc_info.value.status_code == 401
    assert "Could not validate credentials" in exc_info.value.detail

# Test get_current_active_user


@pytest.mark.asyncio
async def test_get_current_active_user_existing(mock_db):
    """Tests retrieving an existing user from DB."""
    mock_session, mock_exec_result = mock_db
    firebase_uid = "existing-uid"
    email = "existing@example.com"
    user_info = {"uid": firebase_uid, "email": email}
    existing_user_obj = User(
        id=uuid.uuid4(), firebase_uid=firebase_uid, email=email)

    mock_session.exec.return_value = mock_exec_result
    mock_exec_result.one_or_none.return_value = existing_user_obj

    # Note: We pass user_info directly, mocking the Depends(verify...)
    user = await get_current_active_user(user_info=user_info, db=mock_session)

    assert user == existing_user_obj
    mock_session.exec.assert_called_once()
    mock_session.add.assert_not_called()


@pytest.mark.asyncio
async def test_get_current_active_user_new(mock_db):
    """Tests creating a new user in DB."""
    mock_session, mock_exec_result = mock_db
    firebase_uid = "new-uid"
    email = "new@example.com"
    user_info = {"uid": firebase_uid, "email": email}

    mock_session.exec.return_value = mock_exec_result
    mock_exec_result.one_or_none.return_value = None  # User not found

    # Simulate refresh assigning ID
    def refresh_side_effect(user_obj):
        user_obj.id = uuid.uuid4()
    mock_session.refresh.side_effect = refresh_side_effect

    user = await get_current_active_user(user_info=user_info, db=mock_session)

    mock_session.exec.assert_called_once()
    mock_session.add.assert_called_once()
    added_user = mock_session.add.call_args[0][0]
    assert added_user.firebase_uid == firebase_uid
    assert added_user.email == email
    mock_session.commit.assert_called_once()
    mock_session.refresh.assert_called_once_with(added_user)
    assert user == added_user


@pytest.mark.asyncio
async def test_get_current_active_user_db_error_lookup(mock_db):
    """Tests DB error during user lookup."""
    mock_session, _ = mock_db
    firebase_uid = "lookup-fail-uid"
    email = "lookup-fail@example.com"
    user_info = {"uid": firebase_uid, "email": email}

    mock_session.exec.side_effect = Exception("DB Lookup Error")

    with pytest.raises(HTTPException) as exc_info:
        await get_current_active_user(user_info=user_info, db=mock_session)

    assert exc_info.value.status_code == 500
    assert "Internal server error while accessing user data" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_current_active_user_db_error_create(mock_db):
    """Tests DB error during user creation."""
    mock_session, mock_exec_result = mock_db
    firebase_uid = "create-fail-uid"
    email = "create-fail@example.com"
    user_info = {"uid": firebase_uid, "email": email}

    mock_session.exec.return_value = mock_exec_result
    mock_exec_result.one_or_none.return_value = None  # User not found
    mock_session.commit.side_effect = Exception("DB Create Error")

    with pytest.raises(HTTPException) as exc_info:
        await get_current_active_user(user_info=user_info, db=mock_session)

    assert exc_info.value.status_code == 500
    assert "Internal server error while accessing user data" in exc_info.value.detail
    mock_session.add.assert_called_once()  # Add should still be called
