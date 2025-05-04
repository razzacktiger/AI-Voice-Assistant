# Tests for standalone Deepgram event handlers

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from api import handle_deepgram_message  # Import the function to test
# Add other necessary imports later


@pytest.mark.asyncio
@patch('api.get_llm_response')
@patch('api.query_pinecone_index')
@patch('api.logging.info')
@patch('api.logging.warning')
async def test_handle_deepgram_message_handler(
    mock_log_warning, mock_log_info, mock_query_pinecone, mock_get_llm
):
    """Tests the handle_deepgram_message function with RAG integration."""

    # --- Setup Mocks ---
    mock_query_pinecone.return_value = ["mock context 1", "mock context 2"]
    mock_get_llm.return_value = "Mock LLM Answer"

    # --- Case 1: Valid transcript -> RAG -> LLM Success ---
    mock_result_valid = MagicMock()
    mock_result_valid.channel.alternatives = [MagicMock()]
    mock_result_valid.channel.alternatives[0].transcript = "Hello world"

    await handle_deepgram_message(mock_result_valid)

    # Assertions for RAG flow
    mock_log_info.assert_any_call("Deepgram -> Transcript: Hello world")
    mock_query_pinecone.assert_awaited_once_with("Hello world")
    mock_log_info.assert_any_call("Pinecone -> Found 2 context items.")
    mock_get_llm.assert_awaited_once_with(
        "Hello world", ["mock context 1", "mock context 2"])
    mock_log_info.assert_any_call("LLM -> Response: Mock LLM Answer")
    mock_log_warning.assert_not_called()

    # --- Reset Mocks for Next Case ---
    mock_log_info.reset_mock()
    mock_log_warning.reset_mock()
    mock_query_pinecone.reset_mock()
    mock_get_llm.reset_mock()

    # --- Case 2: Pinecone returns no context ---
    mock_query_pinecone.return_value = []
    mock_get_llm.return_value = "Answer without context"

    mock_result_valid.channel.alternatives[0].transcript = "Another query"

    await handle_deepgram_message(mock_result_valid)

    mock_log_info.assert_any_call("Deepgram -> Transcript: Another query")
    mock_query_pinecone.assert_awaited_once_with("Another query")
    mock_log_info.assert_any_call("Pinecone -> No relevant context found.")
    mock_get_llm.assert_awaited_once_with("Another query", [])
    mock_log_info.assert_any_call("LLM -> Response: Answer without context")
    mock_log_warning.assert_not_called()

    # --- Reset Mocks ---
    mock_log_info.reset_mock()
    mock_log_warning.reset_mock()
    mock_query_pinecone.reset_mock()
    mock_get_llm.reset_mock()

    # --- Case 3: LLM call fails ---
    mock_query_pinecone.return_value = ["context again"]
    mock_get_llm.return_value = None  # Simulate LLM failure

    mock_result_valid.channel.alternatives[0].transcript = "Query leading to LLM fail"

    await handle_deepgram_message(mock_result_valid)

    mock_log_info.assert_any_call(
        "Deepgram -> Transcript: Query leading to LLM fail")
    mock_query_pinecone.assert_awaited_once_with("Query leading to LLM fail")
    mock_get_llm.assert_awaited_once_with(
        "Query leading to LLM fail", ["context again"])
    mock_log_warning.assert_called_once_with("LLM -> Failed to get response.")
    # Ensure the success log for LLM wasn't called
    for call in mock_log_info.call_args_list:
        assert "LLM -> Response:" not in call.args[0]

    # --- Reset Mocks ---
    mock_log_info.reset_mock()
    mock_log_warning.reset_mock()
    mock_query_pinecone.reset_mock()
    mock_get_llm.reset_mock()

    # --- Case 4: Empty transcript (should return early) ---
    mock_result_empty = MagicMock()
    mock_result_empty.channel.alternatives = [MagicMock()]
    mock_result_empty.channel.alternatives[0].transcript = "   "

    await handle_deepgram_message(mock_result_empty)
    mock_log_info.assert_not_called()
    mock_log_warning.assert_not_called()
    mock_query_pinecone.assert_not_awaited()
    mock_get_llm.assert_not_awaited()

    # --- Reset Mocks ---
    # (No need to reset mocks that weren't called)

    # --- Case 5: Malformed result (missing alternatives - should warn and exit) ---
    mock_result_malformed1 = MagicMock()
    mock_result_malformed1.channel.alternatives = []

    await handle_deepgram_message(mock_result_malformed1)
    mock_log_info.assert_not_called()
    mock_log_warning.assert_called_once()
    assert "unexpected message structure" in mock_log_warning.call_args[0][0]
    mock_query_pinecone.assert_not_awaited()
    mock_get_llm.assert_not_awaited()
