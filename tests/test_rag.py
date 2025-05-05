# Tests for RAG (Pinecone + OpenAI) integration

import pytest
import pytest_asyncio
from unittest.mock import patch, MagicMock, AsyncMock, call
import logging
import json
import asyncio  # Import asyncio for patching to_thread

# Import functions/clients to test from api.rag
from api.rag import (
    get_openai_embedding,
    query_rag_system,
    # Remove imports of client instances, tests should mock/patch these
    # openai_client,
    # pinecone_index
    EMBEDDING_MODEL,
    LLM_MODEL
)

# --- Fixtures ---


@pytest.fixture
def mock_pinecone_index_fixture():
    """Provides a mock Pinecone index object."""
    mock_index = MagicMock()
    # Mock the actual query method used within the to_thread call
    mock_index.query = MagicMock()  # Synchronous mock is fine here
    return mock_index


@pytest.fixture
def mock_openai_client_fixture():
    """Provides a mock OpenAI client object."""
    mock_client = MagicMock()
    # Mock the specific methods used
    mock_client.embeddings = MagicMock()
    mock_client.embeddings.create = MagicMock()  # Synchronous mock
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = MagicMock()  # Synchronous mock
    return mock_client

# --- Helper for mocking asyncio.to_thread (Module Level) ---


async def mock_asyncio_to_thread(func, *args, **kwargs):
    # Directly call the mocked synchronous function in tests
    return func(*args, **kwargs)

# --- Test Cases for get_openai_embedding (moved from test_openai_utils) ---


@pytest.mark.asyncio
@patch('openai.resources.embeddings.Embeddings.create')
async def test_get_openai_embedding_success(mock_create, mock_openai_client_fixture):
    """Tests successful embedding generation."""
    # Patch the _openai_client variable inside the test function's scope
    # to ensure the patched create method is called on our mock client if accessed.
    # This might be redundant if the direct patch on Embeddings.create works.
    with patch('api.rag._openai_client', mock_openai_client_fixture):
        # Configure mock response
        mock_embedding_response = MagicMock()
        mock_embedding_response.data = [MagicMock()]
        mock_embedding_response.data[0].embedding = [0.1, 0.2, 0.3]
        # The mock_create passed by the decorator represents Embeddings.create
        mock_create.return_value = mock_embedding_response

        text_to_embed = "Embed this text"
        expected_embedding = [0.1, 0.2, 0.3]

        embedding = get_openai_embedding(text_to_embed)

        assert embedding == expected_embedding
        mock_create.assert_called_once()
        call_args, call_kwargs = mock_create.call_args
        assert call_kwargs['input'] == [text_to_embed]
        assert call_kwargs['model'] == EMBEDDING_MODEL


@pytest.mark.asyncio
@patch('openai.resources.embeddings.Embeddings.create')
async def test_get_openai_embedding_api_error(mock_create, mock_openai_client_fixture, caplog):
    """Tests behavior when the OpenAI embeddings API call raises an exception
       OR when the client isn't initialized in test mode.
    """
    # Configure the patched create method (though it won't be called in the RuntimeError path)
    mock_create.side_effect = Exception("Embedding API Error!")

    # Patch the internal client var to None to trigger the RuntimeError path
    with patch('api.rag._openai_client', None), \
            pytest.raises(RuntimeError, match="OpenAI client accessed but not initialized/mocked in test."):
        get_openai_embedding("this will fail")

    # In this specific path (RuntimeError), create is NOT called
    mock_create.assert_not_called()
    # Check the warning log from get_openai_embedding
    # Ensure logging level is set for caplog to capture warnings
    caplog.set_level(logging.WARNING)
    assert "OpenAI client not initialized in test" in caplog.text

# --- Test Cases for query_rag_system ---


@pytest.mark.asyncio
@patch('asyncio.to_thread', new_callable=AsyncMock)  # Mock asyncio.to_thread
async def test_query_rag_system_success(mock_to_thread, mock_openai_client_fixture, mock_pinecone_index_fixture, caplog):
    """Tests the full RAG query flow successfully."""
    caplog.set_level(logging.INFO)

    # --- Mock Setup ---
    dummy_embedding = [0.1] * 1536
    mock_pinecone_index_fixture.query.return_value = {
        'matches': [
            {'id': 'vec1', 'score': 0.9, 'metadata': {
                'text': 'Pinecone context 1'}},
            {'id': 'vec2', 'score': 0.8, 'metadata': {'text': 'Pinecone context 2'}}
        ]
    }

    mock_llm_completion = MagicMock()
    mock_llm_completion.choices = [MagicMock()]
    mock_llm_completion.choices[0].message.content = "  Final LLM Answer  "
    mock_openai_client_fixture.chat.completions.create.return_value = mock_llm_completion

    # --- Adjust side effect to compare function names or attributes ---
    async def to_thread_side_effect_success(func, *args, **kwargs):
        # Check if the function being passed is the mocked query or create method
        if func == mock_pinecone_index_fixture.query:
            print("DEBUG: Matched Pinecone query")  # Debug print
            return func(*args, **kwargs)
        # --- Compare directly with the mocked OpenAI create method ---
        elif func == mock_openai_client_fixture.chat.completions.create:
            print("DEBUG: Matched OpenAI create")  # Debug print
            # Directly call the create method on the *mocked* client fixture
            return mock_openai_client_fixture.chat.completions.create(*args, **kwargs)
        else:
            print(f"DEBUG: No match for {func}")  # Debug print
            raise ValueError(
                f"Unexpected function passed to mock_to_thread: {getattr(func, '__qualname__', func)}")
    mock_to_thread.side_effect = to_thread_side_effect_success

    # --- Input Data ---
    transcript = "What is the context?"
    conversation_history = [
        {"type": "user", "text": "Hello there"},
        {"type": "assistant", "text": "Hi! How can I help?"}
    ]
    expected_response = "Final LLM Answer"

    # --- Execute ---
    # Also patch initialize functions to prevent them running after client patches
    with patch('api.rag._openai_client', mock_openai_client_fixture), \
            patch('api.rag._pinecone_index', mock_pinecone_index_fixture), \
            patch('api.rag.get_openai_embedding', return_value=dummy_embedding) as mock_get_emb_call, \
            patch('api.rag.initialize_openai') as mock_init_openai, \
            patch('api.rag.initialize_pinecone') as mock_init_pinecone:
        actual_response = await query_rag_system(transcript, conversation_history)

    # --- Assertions ---
    assert actual_response == expected_response
    mock_get_emb_call.assert_called_once_with(transcript)
    assert mock_to_thread.await_count == 2  # Expect 2 calls now
    # Verify Pinecone query call
    mock_pinecone_index_fixture.query.assert_called_once_with(
        vector=dummy_embedding, top_k=3, include_metadata=True
    )
    # Verify LLM call (check essential args)
    mock_openai_client_fixture.chat.completions.create.assert_called_once()
    call_args, call_kwargs = mock_openai_client_fixture.chat.completions.create.call_args
    assert call_kwargs['model'] == LLM_MODEL  # Use the constant from rag.py
    # assert "Knowledge Base Context:\nPinecone context 1\nPinecone context 2" in call_kwargs['messages'][-1]['content']
    # Check the last message content more carefully
    last_message_content = call_kwargs['messages'][-1]['content']
    assert "User Query: What is the context?" in last_message_content
    assert "Pinecone context 1" in last_message_content
    assert "Pinecone context 2" in last_message_content
    # Check history is included
    assert len(call_kwargs['messages']) == 4  # System + 2 history + 1 user


@pytest.mark.asyncio
@patch('api.rag.get_openai_embedding')
async def test_query_rag_system_embedding_error(mock_get_emb, caplog):
    """Tests RAG query when embedding generation fails."""
    caplog.set_level(logging.ERROR)
    # Configure the patched function to raise an error
    mock_get_emb.side_effect = Exception("Embedding Failed!")

    # Remove the unnecessary inner patch for _openai_client
    response = await query_rag_system("test", [])

    # Check exact string (function should return early)
    assert response == "Sorry, I couldn't process that due to an embedding error."
    assert "Error getting OpenAI embedding: Embedding Failed!" in caplog.text
    mock_get_emb.assert_called_once_with("test")


@pytest.mark.asyncio
@patch('api.rag.get_openai_embedding', new_callable=AsyncMock)
@patch('asyncio.to_thread', new_callable=AsyncMock)
async def test_query_rag_system_pinecone_disabled(mock_to_thread, mock_get_emb, mock_openai_client_fixture, caplog):
    """Tests RAG query when Pinecone index is None."""
    caplog.set_level(logging.INFO)  # Capture INFO level logs
    dummy_embedding = [0.1] * 1536
    mock_get_emb.return_value = dummy_embedding

    mock_llm_completion = MagicMock()
    mock_llm_completion.choices = [MagicMock()]
    mock_llm_completion.choices[0].message.content = "LLM Answer without Pinecone"
    mock_openai_client_fixture.chat.completions.create.return_value = mock_llm_completion

    # Mock to_thread to only call the LLM method
    async def to_thread_side_effect_no_pinecone(func, *args, **kwargs):
        if func == mock_openai_client_fixture.chat.completions.create:
            return func(*args, **kwargs)  # Call the underlying mock
        else:
            # Should not be called for pinecone in this test
            raise ValueError("Unexpected function passed to mock_to_thread")
    mock_to_thread.side_effect = to_thread_side_effect_no_pinecone

    transcript = "Query without pinecone"
    # Patch internal vars: _pinecone_index=None
    with patch('api.rag._openai_client', mock_openai_client_fixture), \
            patch('api.rag._pinecone_index', None), \
            patch('api.rag.initialize_pinecone'), \
            patch('api.rag.initialize_openai'):  # Patch inits to be safe
        response = await query_rag_system(transcript, [])

    assert response == "LLM Answer without Pinecone"
    assert "Pinecone index not available or not found, skipping vector search." in caplog.text
    mock_get_emb.assert_called_once_with(transcript)
    mock_to_thread.assert_awaited_once()  # Only called for LLM
    # Verify sync method called via to_thread
    mock_openai_client_fixture.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
@patch('api.rag.get_openai_embedding', new_callable=AsyncMock)
@patch('asyncio.to_thread', new_callable=AsyncMock)
async def test_query_rag_system_pinecone_error(mock_to_thread, mock_get_emb, mock_openai_client_fixture, mock_pinecone_index_fixture, caplog):
    """Tests RAG query when Pinecone query fails."""
    caplog.set_level(logging.ERROR)  # Capture ERROR level logs
    dummy_embedding = [0.1] * 1536
    mock_get_emb.return_value = dummy_embedding

    pinecone_error = Exception("Pinecone Down!")
    mock_pinecone_index_fixture.query.side_effect = pinecone_error

    mock_llm_completion = MagicMock()
    mock_llm_completion.choices = [MagicMock()]
    mock_llm_completion.choices[0].message.content = "LLM Answer after Pinecone fail"
    mock_openai_client_fixture.chat.completions.create.return_value = mock_llm_completion

    # Configure mock_to_thread: Pinecone raises error, LLM returns result
    async def to_thread_side_effect_pc_fail(func, *args, **kwargs):
        # Explicitly call the mocked sync method to check assertions later
        if func == mock_pinecone_index_fixture.query:
            # Call the underlying mock which is configured to raise the error
            # The error is raised here, stopping execution before the explicit raise
            func(*args, **kwargs)
            # This line won't be reached if the above raises as expected
            raise pinecone_error  # Keep for safety, but shouldn't be needed
        elif func == mock_openai_client_fixture.chat.completions.create:
            # Call the underlying LLM mock
            return func(*args, **kwargs)
        else:
            raise ValueError("Unexpected function passed to mock_to_thread")
    mock_to_thread.side_effect = to_thread_side_effect_pc_fail

    transcript = "Query with pinecone fail"
    # Ensure the mock index is injected correctly
    with patch('api.rag._openai_client', mock_openai_client_fixture), \
            patch('api.rag._pinecone_index', mock_pinecone_index_fixture), \
            patch('api.rag.initialize_pinecone'), \
            patch('api.rag.initialize_openai'):
        response = await query_rag_system(transcript, [])

    assert response == "LLM Answer after Pinecone fail"
    assert "Error querying Pinecone: Pinecone Down!" in caplog.text
    mock_get_emb.assert_called_once_with(transcript)
    # Should still try both pinecone and llm
    assert mock_to_thread.await_count == 2
    mock_pinecone_index_fixture.query.assert_called_once()  # Verify attempt
    # Verify LLM was called
    mock_openai_client_fixture.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
@patch('api.rag.get_openai_embedding', new_callable=AsyncMock)
@patch('asyncio.to_thread', new_callable=AsyncMock)
async def test_query_rag_system_llm_error(mock_to_thread, mock_get_emb, mock_openai_client_fixture, mock_pinecone_index_fixture, caplog):
    """Tests RAG query when the final LLM call fails."""
    caplog.set_level(logging.ERROR)  # Capture ERROR level logs
    dummy_embedding = [0.1] * 1536
    mock_get_emb.return_value = dummy_embedding

    pinecone_query_response = {'matches': []}
    mock_pinecone_index_fixture.query.return_value = pinecone_query_response

    llm_error = Exception("LLM Unavailable!")
    mock_openai_client_fixture.chat.completions.create.side_effect = llm_error

    async def to_thread_side_effect_llm_fail(func, *args, **kwargs):
        # Explicitly call the mocked sync method to check assertions later
        if func == mock_pinecone_index_fixture.query:
            return func(*args, **kwargs)
        elif func == mock_openai_client_fixture.chat.completions.create:
            result = func(*args, **kwargs)  # This will raise the side_effect
            raise llm_error
        else:
            raise ValueError("Unexpected function passed to mock_to_thread")
    mock_to_thread.side_effect = to_thread_side_effect_llm_fail

    transcript = "Query with LLM fail"
    # Ensure mocks are injected
    with patch('api.rag._openai_client', mock_openai_client_fixture), \
            patch('api.rag._pinecone_index', mock_pinecone_index_fixture), \
            patch('api.rag.initialize_pinecone'), \
            patch('api.rag.initialize_openai'):
        response = await query_rag_system(transcript, [])

    # Check exact string
    assert response == "Sorry, I encountered an error trying to generate a response."
    assert "Error calling OpenAI LLM: LLM Unavailable!" in caplog.text
    mock_get_emb.assert_called_once_with(transcript)
    assert mock_to_thread.await_count == 2
    mock_pinecone_index_fixture.query.assert_called_once()
    mock_openai_client_fixture.chat.completions.create.assert_called_once()
