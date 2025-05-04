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
# Use fixture for mocks, context manager for patching api globals
async def test_query_pinecone_index_success(mock_pinecone_instance):
    """Tests successful query to Pinecone index."""
    with patch('api.get_openai_embedding') as mock_get_embedding, \
            patch('api.pinecone_client') as mock_global_client_patch, \
            patch('api.PINECONE_INDEX_NAME', 'test-index'):

        # Configure mock embedding function
        dummy_embedding = [0.1] * 1536
        mock_get_embedding.return_value = dummy_embedding

        # Configure mock Pinecone client and index (using fixture)
        _, mock_index = mock_pinecone_instance
        mock_global_client_patch.Index.return_value = mock_index  # Link patch to fixture mock

        # Configure mock index query response
        mock_index.query.return_value = {
            'matches': [
                {'id': 'vec1', 'score': 0.9, 'metadata': {'text': 'mock context 1'}},
                {'id': 'vec2', 'score': 0.8, 'metadata': {'text': 'mock context 2'}}
            ]
        }

        transcript = "Tell me about project X"
        top_k = 2
        expected_results = ['mock context 1', 'mock context 2']

        results = await query_pinecone_index(transcript, top_k=top_k)

        # Assertions
        mock_get_embedding.assert_awaited_once_with(transcript)
        mock_global_client_patch.Index.assert_called_once_with('test-index')
        mock_index.query.assert_called_once()
        call_args, call_kwargs = mock_index.query.call_args
        assert call_kwargs['vector'] == dummy_embedding
        assert call_kwargs['top_k'] == top_k
        assert call_kwargs['include_metadata'] is True
        assert results == expected_results


@pytest.mark.asyncio
async def test_query_pinecone_index_embedding_fails():
    """Tests behavior when get_openai_embedding returns None."""
    # Need to patch pinecone_client so the initial check passes
    with patch('api.pinecone_client', MagicMock()), \
            patch('api.PINECONE_INDEX_NAME', 'test-index'), \
            patch('api.get_openai_embedding') as mock_get_embedding, \
            patch('api.logging.error') as mock_log_error:

        mock_get_embedding.return_value = None  # Simulate embedding failure
        results = await query_pinecone_index("test transcript")

    assert results == []
    mock_get_embedding.assert_awaited_once_with("test transcript")
    mock_log_error.assert_called_once()
    assert "Failed to generate embedding" in mock_log_error.call_args[0][0]


@pytest.mark.asyncio
async def test_query_pinecone_index_disabled():
    """Tests behavior when Pinecone client is None (disabled)."""
    with patch('api.pinecone_client', None):
        results = await query_pinecone_index("test transcript")
        assert results == []


@pytest.mark.asyncio
async def test_query_pinecone_index_no_index_name():
    """Tests behavior when PINECONE_INDEX_NAME is not set."""
    with patch('api.pinecone_client', MagicMock()), \
            patch('api.PINECONE_INDEX_NAME', None):
        results = await query_pinecone_index("test transcript")
        assert results == []


@pytest.mark.asyncio
async def test_query_pinecone_index_query_error(mock_pinecone_instance):
    """Tests behavior when index.query() raises an exception."""
    with patch('api.get_openai_embedding') as mock_get_embedding, \
            patch('api.pinecone_client') as mock_global_client_patch, \
            patch('api.PINECONE_INDEX_NAME', 'test-index'), \
            patch('api.logging.error') as mock_log_error:

        # Configure mock embedding
        dummy_embedding = [0.2] * 1536
        mock_get_embedding.return_value = dummy_embedding

        # Configure Pinecone mocks to raise error on query
        _, mock_index = mock_pinecone_instance
        mock_global_client_patch.Index.return_value = mock_index
        mock_index.query.side_effect = Exception("Pinecone query failed!")

        transcript = "This should fail"
        results = await query_pinecone_index(transcript)

        # Assertions
        mock_get_embedding.assert_awaited_once_with(transcript)
        mock_global_client_patch.Index.assert_called_once_with('test-index')
        mock_index.query.assert_called_once()
        mock_log_error.assert_called_once()
        assert "Error querying Pinecone index" in mock_log_error.call_args[0][0]
        assert "Pinecone query failed!" in mock_log_error.call_args[0][0]
        assert results == []
