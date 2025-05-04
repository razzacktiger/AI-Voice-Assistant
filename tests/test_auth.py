# Tests for authentication related functions and endpoints

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import status, HTTPException
from fastapi.testclient import TestClient
from datetime import datetime, timezone

# Import app and models/dependencies needed for auth tests
from api import app, User, get_current_user

# Note: Fixtures mock_db_session and auto_override_dependencies
# are automatically available from tests/conftest.py
client = TestClient(app)

# --- Test Cases for /users/me ---


@pytest.mark.asyncio
async def test_read_users_me_success(mock_db_session, auto_override_dependencies):
    """Tests successful retrieval of user info via /users/me."""
    firebase_uid = "test_uid_http"
    user_email = "http@example.com"
    db_user_id = 5
    token = "valid_http_token"
    now = datetime.now(timezone.utc)
    mock_user = User(id=db_user_id, firebase_uid=firebase_uid,
                     email=user_email, created_at=now)

    app.dependency_overrides[get_current_user] = lambda: mock_user

    response = client.get(
        "/users/me", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    user_data = response.json()
    assert user_data["id"] == db_user_id
    assert user_data["firebase_uid"] == firebase_uid
    assert user_data["email"] == user_email
    assert "created_at" in user_data

    del app.dependency_overrides[get_current_user]


@pytest.mark.asyncio
async def test_read_users_me_create_user(mock_db_session, auto_override_dependencies):
    """Tests /users/me with a user returned by overridden dependency."""
    firebase_uid = "new_user_uid"
    user_email = "new@example.com"
    token = "valid_new_user_token"
    now = datetime.now(timezone.utc)
    db_user_id = 999
    mock_user = User(id=db_user_id, firebase_uid=firebase_uid,
                     email=user_email, created_at=now)

    app.dependency_overrides[get_current_user] = lambda: mock_user

    response = client.get(
        "/users/me", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    user_data = response.json()
    assert user_data["id"] == db_user_id
    assert user_data["firebase_uid"] == firebase_uid
    assert user_data["email"] == user_email
    assert "created_at" in user_data

    del app.dependency_overrides[get_current_user]


@pytest.mark.asyncio
async def test_read_users_me_auth_header_missing(auto_override_dependencies):
    """Tests /users/me failure when Authorization header is missing."""
    def raise_missing_header():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing",
        )
    app.dependency_overrides[get_current_user] = raise_missing_header

    response = client.get("/users/me")
    assert response.status_code == 401
    assert "Authorization header missing" in response.text

    del app.dependency_overrides[get_current_user]


@pytest.mark.asyncio
async def test_read_users_me_invalid_token(auto_override_dependencies):
    """Tests /users/me failure with an invalid token."""
    token = "invalid_http_token"

    def raise_invalid_token():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired Firebase token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    app.dependency_overrides[get_current_user] = raise_invalid_token

    response = client.get(
        "/users/me", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 401
    assert "Invalid or expired Firebase token" in response.text

    del app.dependency_overrides[get_current_user]

# TODO: Add tests for the get_current_user dependency function itself?
# This would involve mocking verify_firebase_token and the db session directly.

# TODO: Add tests for the get_current_user_ws dependency function?
