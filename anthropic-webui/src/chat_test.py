import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic.types import (
    ImageBlockParam,
    MessageParam,
    TextBlock,
    TextBlockParam,
)
from anthropic.types.image_block_param import Source
from openai.types.chat import ChatCompletionUserMessageParam

from src.chat import (
    create_chat_completion,
    create_chat_completion_stream,
    get_anthropic_image_source,
    get_anthropic_message_content,
    get_anthropic_messages,
)


def test_get_anthropic_image_source() -> None:
    url = "data:image/png;base64,iVBORw"
    source = get_anthropic_image_source(url)
    assert source == Source(
        data="iVBORw",
        media_type="image/png",
        type="base64",
    )


def test_get_anthropic_message_content_text() -> None:
    content = "Test message"
    message_content = get_anthropic_message_content(content)
    assert message_content == content


def test_get_anthropic_message_content_blocks() -> None:
    content = [
        {"type": "text", "text": "Test message"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw"}},
    ]
    message_content = get_anthropic_message_content(content)
    assert len(message_content) == 2
    assert message_content[0] == TextBlockParam(text="Test message", type="text")
    assert message_content[1] == ImageBlockParam(
        source=Source(data="iVBORw", media_type="image/png", type="base64"),
        type="image",
    )


def test_get_anthropic_messages_text() -> None:
    messages = [
        ChatCompletionUserMessageParam(role="user", content="Test message"),
    ]
    message_params = get_anthropic_messages(messages)
    assert len(message_params) == 1
    assert message_params[0] == MessageParam(
        content="Test message",
        role="user",
    )


def test_get_anthropic_messages_blocks() -> None:
    messages = [
        ChatCompletionUserMessageParam(role="user", content="Test message"),
        ChatCompletionUserMessageParam(
            role="assistant",
            content=[
                {"type": "text", "text": "Test response"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,iVBORw"},
                },
            ],
        ),
    ]
    message_params = get_anthropic_messages(messages)
    assert len(message_params) == 2
    assert message_params[0] == MessageParam(
        content="Test message",
        role="user",
    )
    assert message_params[1] == MessageParam(
        content=[
            TextBlockParam(text="Test response", type="text"),
            ImageBlockParam(
                source=Source(data="iVBORw", media_type="image/png", type="base64"),
                type="image",
            ),
        ],
        role="assistant",
    )


@pytest.mark.asyncio
@patch("anthropic.AsyncAnthropic")
async def test_create_chat_completion(mock_anthropic: AsyncMock) -> None:
    mock_response = MagicMock()
    mock_response.id = "123"
    mock_response.created = 1234567890
    mock_response.content = [TextBlock(text="Test response", type="text")]
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
