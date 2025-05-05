# Tests for database utility functions

import pytest
from unittest.mock import patch, MagicMock
from sqlmodel import SQLModel, create_engine, Session

# Import functions to test
from api.database import create_db_and_tables, get_session, engine as db_engine

# --- Test Cases ---


def test_create_db_and_tables():
    """Tests the create_db_and_tables function."""
    # We need to patch SQLModel.metadata.create_all
    # Patch the *specific* engine instance used in the database module
    with patch('api.database.SQLModel.metadata.create_all') as mock_create_all, \
            patch('api.database.engine') as mock_engine_instance:

        create_db_and_tables()

        # Assert that create_all was called once with the engine instance
        mock_create_all.assert_called_once_with(mock_engine_instance)


def test_get_session():
    """Tests the get_session dependency generator."""
    # Patch the Session class used within get_session
    # Also patch the engine instance it uses
    mock_session_instance = MagicMock(spec=Session)
    with patch('api.database.Session') as mock_session_class, \
            patch('api.database.engine') as mock_engine_instance:

        mock_session_class.return_value.__enter__.return_value = mock_session_instance

        # Call the generator function
        gen = get_session()
        # Get the yielded session
        yielded_session = next(gen)

        # Assert that the Session was instantiated with the engine
        mock_session_class.assert_called_once_with(mock_engine_instance)
        # Assert that the yielded value is the mock session instance
        assert yielded_session == mock_session_instance

        # Assert that the session context manager __exit__ is called upon generator exit
        with pytest.raises(StopIteration):
            next(gen)
        # Check that the context manager exit was called
        mock_session_class.return_value.__exit__.assert_called_once()

# Potential future tests:
# - Test specific CRUD functions if they are added to database.py
# - Test database connection retry logic if implemented
# - Test engine configuration (e.g., pool size if configured)
