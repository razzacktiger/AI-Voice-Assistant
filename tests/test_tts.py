# Tests for Deepgram TTS streaming functionality

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import logging
from fastapi import WebSocket

# Import the function and options class to test
# Import client class
from api.tts import generate_and_stream_tts, SpeakOptions, DeepgramClient

# --- Mocks and Fixtures ---


@pytest.fixture
def mock_websocket():
    """Provides a mock WebSocket object with an async send_bytes method."""
    ws = MagicMock(spec=WebSocket)
    ws.send_bytes = AsyncMock()
    return ws

# No longer need mock_deepgram_speak_stream fixture, will patch DeepgramClient instead

# --- Test Cases ---


@pytest.mark.asyncio
# Patch the DeepgramClient where it's imported in api.tts
@patch('api.tts.DeepgramClient')
async def test_generate_and_stream_tts_success(MockDeepgramClient, mock_websocket):
    """Tests successful TTS audio generation and streaming."""
    # --- Configure Mock Chain --- (Updated for speak.rest.v("1").stream() -> .stream.aiter_bytes())
    mock_client_instance = MagicMock(spec=DeepgramClient)
    mock_speak_router = MagicMock()  # Represents client.speak
    mock_rest_router = MagicMock()  # Represents client.speak.rest
    mock_version_router = MagicMock()  # Represents client.speak.rest.v("1")

    # Mock the final response object returned by stream()
    mock_speak_stream_response = MagicMock()

    # Mock the async iterator part
    async def async_iterator_bytes():
        yield b"audio_chunk_1"
        yield b""  # Empty chunk test
        yield b"audio_chunk_2"
    mock_speak_stream_response.stream = MagicMock()  # The .stream attribute
    mock_speak_stream_response.stream.aiter_bytes = MagicMock(
        return_value=async_iterator_bytes())  # Has aiter_bytes()

    # Configure the call chain: .stream() should return the mock response object
    mock_version_router.stream = MagicMock(
        return_value=mock_speak_stream_response)
    mock_rest_router.v = MagicMock(return_value=mock_version_router)
    mock_speak_router.rest = mock_rest_router
    mock_client_instance.speak = mock_speak_router
    MockDeepgramClient.return_value = mock_client_instance
    # -------------------------------------------\

    text_to_speak = "Stream this speech."

    # --- Mock global client variable directly ---
    # Since the client is initialized globally, we also need to patch the instance used
    with patch('api.tts.deepgram_client', mock_client_instance):
        await generate_and_stream_tts(text_to_speak, mock_websocket)

    # Verify the DeepgramClient instantiation mock was *not* called again (it's global)
    # MockDeepgramClient.assert_called_once() # This may not be true if it was initialized at module level before test

    # Verify the mock instance's methods were called
    mock_client_instance.speak.rest.v.assert_called_once_with("1")
    mock_version_router.stream.assert_called_once()  # stream() itself is synchronous
    # Verify the async iterator was entered
    mock_speak_stream_response.stream.aiter_bytes.assert_called_once()

    # Check stream() positional arguments
    call_args, call_kwargs = mock_version_router.stream.call_args
    # First positional arg (payload)
    assert call_args[0] == {"text": text_to_speak}
    options_arg = call_args[1]  # Second positional arg (options)
    # assert call_kwargs == {} # Ensure no unexpected keyword args were passed
    # assert call_kwargs.get('options') == None # Explicit check if needed

    assert isinstance(options_arg, SpeakOptions)
    assert options_arg.model == "aura-asteria-en"
    assert options_arg.encoding == "linear16"
    assert options_arg.container == "none"
    assert mock_websocket.send_bytes.await_count == 2
    mock_websocket.send_bytes.assert_any_await(b"audio_chunk_1")
    mock_websocket.send_bytes.assert_any_await(b"audio_chunk_2")


@pytest.mark.asyncio
@patch('api.tts.DeepgramClient')
async def test_generate_and_stream_tts_api_error(MockDeepgramClient, mock_websocket, caplog):
    """Tests behavior when the Deepgram speak.rest.v("1").stream API call raises an exception."""
    caplog.set_level(logging.INFO)

    # --- Configure Mock Chain to Raise Error --- (Updated for speak.rest.v().stream)
    mock_client_instance = MagicMock(spec=DeepgramClient)
    mock_speak_router = MagicMock()
    mock_rest_router = MagicMock()
    mock_version_router = MagicMock()

    # Configure the synchronous stream() call to raise the exception
    mock_version_router.stream = MagicMock(
        side_effect=Exception("Deepgram TTS Stream Error!"))
    mock_rest_router.v = MagicMock(return_value=mock_version_router)
    mock_speak_router.rest = mock_rest_router
    mock_client_instance.speak = mock_speak_router
    # MockDeepgramClient.return_value = mock_client_instance # Don't need this if patching instance below

    # -------------------------------------------\

    text_to_speak = "This will fail"

    # --- Mock global client variable directly ---
    with patch('api.tts.deepgram_client', mock_client_instance):
        await generate_and_stream_tts(text_to_speak, mock_websocket)

    # Verify the API call chain was attempted up to the point of error
    mock_client_instance.speak.rest.v.assert_called_once_with("1")
    mock_version_router.stream.assert_called_once()  # stream() is sync
    # Verify no audio bytes were sent
    mock_websocket.send_bytes.assert_not_awaited()
    # Verify error was logged
    assert "Error during Deepgram TTS generation or streaming: Deepgram TTS Stream Error!" in caplog.text

# Note: Testing the "disabled" case (DEEPGRAM_API_KEY not set) is harder
# as it raises an error on module import. Such tests usually require
# manipulating environment variables before import, which can be complex.
# It might be sufficient to rely on the initial check within the module itself.
