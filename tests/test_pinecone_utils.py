# Tests for Pinecone integration

import pytest
import pytest_asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import logging

# Import functions/clients to test from api.py
# Assuming tests are run from project root
from api import query_pinecone_index  # Import the query function
# We might need to test initialization indirectly or patch os.getenv

# --- Fixtures (if any needed specifically for Pinecone tests) ---
# Example: Fixture to provide a mock Pinecone client instance


@pytest.fixture
def mock_pinecone_instance():
    mock_client = MagicMock()
    # Mock the Index method to return another mock
    mock_index = MagicMock()
    mock_client.Index.return_value = mock_index
    # Mock the query method on the index mock
    mock_index.query = MagicMock()
    return mock_client, mock_index

# --- Test Cases ---

# TODO: Add test for Pinecone client initialization


@pytest.mark.asyncio
# Removed patched args from signature
async def test_query_pinecone_index_success(mock_pinecone_instance):
    """Tests successful query to Pinecone index (placeholder response)."""
    # Use patch as context manager
    with patch('api.pinecone_client') as mock_global_client_patch, \
            patch('api.PINECONE_INDEX_NAME', 'test-index'):  # No need for handle if not used

        _, mock_index = mock_pinecone_instance
        mock_global_client_patch.Index.return_value = mock_index

        transcript = "Tell me about project X"
        top_k = 2
        expected_results = [
            f"Placeholder context related to '{transcript[:20]}' 1",
            f"Placeholder context related to '{transcript[:20]}' 2"
        ]

        results = await query_pinecone_index(transcript, top_k=top_k)

        mock_global_client_patch.Index.assert_called_once_with('test-index')
        # mock_index.query.assert_called_once()
        assert results == expected_results


@pytest.mark.asyncio
# Removed patched arg from signature
async def test_query_pinecone_index_disabled():
    """Tests behavior when Pinecone client is None (disabled)."""
    with patch('api.pinecone_client', None):  # Patch directly in context
        results = await query_pinecone_index("test transcript")
        assert results == []


@pytest.mark.asyncio
# Removed patched args from signature
async def test_query_pinecone_index_no_index_name():
    """Tests behavior when PINECONE_INDEX_NAME is not set."""
    # Need a non-None client for the first check in the function
    with patch('api.pinecone_client', MagicMock()), \
            patch('api.PINECONE_INDEX_NAME', None):  # Patch index name in context
        results = await query_pinecone_index("test transcript")
        assert results == []


@pytest.mark.asyncio
# Removed patched args from signature
async def test_query_pinecone_index_query_error(mock_pinecone_instance):
    """Tests behavior when getting the index handle raises an exception."""
    # Note: We test error during index fetch, as query() is currently commented out
    with patch('api.pinecone_client') as mock_global_client_patch, \
            patch('api.PINECONE_INDEX_NAME', 'test-index'), \
            patch('api.logging.error') as mock_log_error:

        # Configure the patched client's Index() method to raise an error
        mock_global_client_patch.Index.side_effect = Exception(
            "Pinecone connection failed!")
        # _, mock_index = mock_pinecone_instance # No longer need mock_index here
        # mock_index.query.side_effect = Exception("Pinecone query failed!") # Moved side effect

        transcript = "This should fail"
        results = await query_pinecone_index(transcript)

        # Assertions
        # Assert that the *patched* client's Index method was called (attempted)
        mock_global_client_patch.Index.assert_called_once_with('test-index')
        # mock_index.query.assert_called_once() # Query is not called, remove this
        mock_log_error.assert_called_once()  # Error should have been logged
        # Check that the specific error was logged
        assert "Error querying Pinecone index" in mock_log_error.call_args[0][0]
        # Update expected exception message
        assert "Pinecone connection failed!" in mock_log_error.call_args[0][0]
        assert results == []
