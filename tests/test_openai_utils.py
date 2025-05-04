# Tests for OpenAI integration

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import logging

# Import functions/clients to test from api.py
# Import the functions to test
from api import get_llm_response, get_openai_embedding

# --- Test Cases for get_openai_embedding ---


@pytest.mark.asyncio
@patch('api.openai_client')
async def test_get_openai_embedding_success(mock_global_openai_client):
    """Tests successful embedding generation."""
    # Configure mock response
    mock_embedding_response = MagicMock()
    mock_embedding_response.data = [MagicMock()]
    mock_embedding_response.data[0].embedding = [0.1, 0.2, 0.3]
    mock_global_openai_client.embeddings.create = AsyncMock(
        return_value=mock_embedding_response)

    text_to_embed = "Embed this text\nwith newline"
    expected_embedding = [0.1, 0.2, 0.3]
    expected_input_text = "Embed this text with newline"  # Newline replaced

    embedding = await get_openai_embedding(text_to_embed)

    # Assertions
    assert embedding == expected_embedding
    mock_global_openai_client.embeddings.create.assert_awaited_once()
    # Check that the input text had newline replaced
    call_args, call_kwargs = mock_global_openai_client.embeddings.create.await_args
    assert call_kwargs['input'] == [expected_input_text]
    assert call_kwargs['model'] == "text-embedding-3-small"  # Default model


@pytest.mark.asyncio
# REMOVED decorator @patch('api.openai_client', None)
async def test_get_openai_embedding_disabled():  # REMOVED argument mock_client_none
    """Tests behavior when OpenAI client is None (disabled)."""
    # Use context manager
    with patch('api.openai_client', None):
        embedding = await get_openai_embedding("test")
        assert embedding is None


@pytest.mark.asyncio
@patch('api.openai_client')
@patch('api.logging.error')
async def test_get_openai_embedding_api_error(mock_log_error, mock_global_openai_client):
    """Tests behavior when the OpenAI embeddings API call raises an exception."""
    mock_global_openai_client.embeddings.create = AsyncMock(
        side_effect=Exception("Embedding API Error!"))

    embedding = await get_openai_embedding("this will fail")

    # Assertions
    mock_global_openai_client.embeddings.create.assert_awaited_once()
    mock_log_error.assert_called_once()
    logged_message = mock_log_error.call_args[0][0]
    assert "Error generating OpenAI embedding" in logged_message
    assert "Embedding API Error!" in logged_message
    assert embedding is None


# --- Test Cases for get_llm_response ---

@pytest.mark.asyncio
@patch('api.openai_client')
async def test_get_llm_response_success(mock_global_openai_client):
    """Tests successful LLM response generation with transcript and context."""
    # Configure mock response for the actual API call
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.content = "Mocked LLM Response"
    mock_global_openai_client.chat.completions.create = AsyncMock(
        return_value=mock_completion)

    transcript = "Tell me about the weather."
    pinecone_context = ["Weather is sunny.", "Temperature is 25C."]
    expected_result = "Mocked LLM Response"

    # Call the modified function
    result = await get_llm_response(transcript, pinecone_context)

    # Assertions for actual API call
    assert result == expected_result
    mock_global_openai_client.chat.completions.create.assert_awaited_once()
    call_args, call_kwargs = mock_global_openai_client.chat.completions.create.await_args

    # Verify the structure of the messages argument
    assert 'messages' in call_kwargs
    messages = call_kwargs['messages']
    assert len(messages) == 2
    assert messages[0]['role'] == 'system'
    assert messages[1]['role'] == 'user'
    # Check if transcript and context are in the user message
    assert transcript in messages[1]['content']
    for context_item in pinecone_context:
        assert context_item in messages[1]['content']
    # Check other params
    assert call_kwargs['model'] == 'gpt-4'
    assert call_kwargs['temperature'] == 0.7
    assert call_kwargs['max_tokens'] == 150


@pytest.mark.asyncio
async def test_get_llm_response_disabled():
    """Tests behavior when OpenAI client is None (disabled)."""
    with patch('api.openai_client', None):
        # Call with dummy transcript/context
        result = await get_llm_response("test transcript", [])
        assert result is None


@pytest.mark.asyncio
@patch('api.openai_client')
@patch('api.logging.error')
async def test_get_llm_response_api_error(mock_log_error, mock_global_openai_client):
    """Tests behavior when the OpenAI API call raises an exception."""
    mock_global_openai_client.chat.completions.create = AsyncMock(
        side_effect=Exception("OpenAI API Error!"))

    transcript = "This should cause an error"
    pinecone_context = ["Some context"]
    # Call the modified function
    result = await get_llm_response(transcript, pinecone_context)

    # Assertions
    mock_global_openai_client.chat.completions.create.assert_awaited_once()
    mock_log_error.assert_called_once()
    logged_message = mock_log_error.call_args[0][0]
    assert "Error getting response from OpenAI" in logged_message
    assert "OpenAI API Error!" in logged_message
    assert result is None
