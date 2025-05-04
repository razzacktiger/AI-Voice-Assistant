# Tests for OpenAI integration

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import logging

# Import functions/clients to test from api.py
from api import get_llm_response  # Import the function to test

# --- Test Cases ---


@pytest.mark.asyncio
@patch('api.openai_client')
async def test_get_llm_response_success(mock_global_openai_client):
    """Tests successful LLM response generation."""
    # Configure mock response for the actual API call
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.content = "Mocked LLM Response"
    mock_global_openai_client.chat.completions.create = AsyncMock(
        return_value=mock_completion)

    prompt = "Explain quantum physics simply."
    expected_result = "Mocked LLM Response"

    result = await get_llm_response(prompt)

    # Assertions for actual API call
    assert result == expected_result
    mock_global_openai_client.chat.completions.create.assert_awaited_once()
    # Can add more specific checks on call args if needed
    # print(mock_global_openai_client.chat.completions.create.await_args)


@pytest.mark.asyncio
# @patch('api.openai_client', None) # REMOVED DECORATOR
async def test_get_llm_response_disabled():  # REMOVED ARGUMENT
    """Tests behavior when OpenAI client is None (disabled)."""
    # Use context manager instead
    with patch('api.openai_client', None):
        result = await get_llm_response("test prompt")
        assert result is None


@pytest.mark.asyncio
@patch('api.openai_client')
@patch('api.logging.error')
async def test_get_llm_response_api_error(mock_log_error, mock_global_openai_client):
    """Tests behavior when the OpenAI API call raises an exception."""
    mock_global_openai_client.chat.completions.create = AsyncMock(
        side_effect=Exception("OpenAI API Error!"))

    prompt = "This should cause an error"
    result = await get_llm_response(prompt)

    # Assertions
    mock_global_openai_client.chat.completions.create.assert_awaited_once()
    mock_log_error.assert_called_once()
    # Check the exception message within the single logged argument
    logged_message = mock_log_error.call_args[0][0]
    assert "Error getting response from OpenAI" in logged_message
    assert "OpenAI API Error!" in logged_message
    assert result is None
