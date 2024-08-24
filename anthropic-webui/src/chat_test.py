import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types.chat import ChatCompletionUserMessageParam

from src.chat import (
    create_chat_completion,
    create_chat_completion_stream,
)


@pytest.mark.asyncio
@patch("anthropic.AsyncAnthropic")
async def test_create_chat_completion(mock_anthropic: AsyncMock) -> None:
    mock_response = MagicMock()
    mock_response.id = "123"
    mock_response.created = 1234567890
    mock_response.content[0].text = "Test response"
    mock_anthropic.messages.create = AsyncMock(return_value=mock_response)

    completion = await create_chat_completion(
        client=mock_anthropic,
        model="claude-3-5-sonnet-20240620",
        messages=[ChatCompletionUserMessageParam(role="user", content="Test message")],
    )

    mock_anthropic.messages.create.assert_awaited_once_with(
        max_tokens=4096,
        model="claude-3-5-sonnet-20240620",
        messages=[ChatCompletionUserMessageParam(role="user", content="Test message")],
    )
    completion_dict = completion.model_dump()
    assert completion_dict["object"] == "chat.completion"
    assert completion_dict["model"] == "claude-3-5-sonnet-20240620"
    assert completion_dict["choices"] == [
        {
            "index": 0,
            "logprobs": None,
            "message": {
                "role": "assistant",
                "content": "Test response",
                "function_call": None,
                "refusal": None,
                "tool_calls": None,
            },
            "finish_reason": "stop",
        }
    ]


@pytest.mark.asyncio
@patch("anthropic.AsyncAnthropic")
async def test_create_chat_completion_stream(mock_anthropic: AsyncMock) -> None:
    mock_stream = AsyncMock()
    mock_stream.id = "123"
    mock_stream.created = 1234567890

    mock_stream.text_stream = AsyncMock()
    mock_stream.text_stream.__aiter__.return_value = ["Test ", "response"]

    # Create a context manager mock
    mock_context_manager = AsyncMock()
    mock_context_manager.__aenter__.return_value = mock_stream
    mock_context_manager.__aexit__.return_value = None

    # Set the stream method to return the context manager mock
    mock_anthropic.messages.stream.return_value = mock_context_manager

    stream = await create_chat_completion_stream(
        client=mock_anthropic,
        model="claude-3-5-sonnet-20240620",
        messages=[ChatCompletionUserMessageParam(role="user", content="Test message")],
    )

    chunks = [chunk async for chunk in stream]

    mock_anthropic.messages.stream.assert_called_once_with(
        max_tokens=4096,
        model="claude-3-5-sonnet-20240620",
        messages=[ChatCompletionUserMessageParam(role="user", content="Test message")],
    )

    assert len(chunks) == 3  # Two content chunks + one final chunk

    for chunk in chunks[:-1]:
        assert chunk.startswith("data: ")
        chunk_data = json.loads(chunk.split("data: ")[1])
        assert chunk_data["model"] == "claude-3-5-sonnet-20240620"
        assert chunk_data["object"] == "chat.completion.chunk"
        assert len(chunk_data["choices"]) == 1
        assert "content" in chunk_data["choices"][0]["delta"]

    # Check the final chunk
    final_chunk = json.loads(chunks[-1].split("data: ")[1])
    assert final_chunk["choices"][0]["finish_reason"] == "stop"
