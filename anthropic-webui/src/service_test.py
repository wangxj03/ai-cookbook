import json
from unittest.mock import AsyncMock, patch

import pytest
from anthropic import AsyncAnthropic
from fastapi.testclient import TestClient
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion import Choice as ChatCompletionChoice

from src.service import MODELS, create_app


@pytest.fixture
def mock_anthropic() -> AsyncMock:
    mock = AsyncMock(spec=AsyncAnthropic)
    mock.messages = AsyncMock()
    mock.messages.create = AsyncMock()
    return mock


@pytest.fixture
def client(mock_anthropic: AsyncMock) -> TestClient:
    app = create_app(client=mock_anthropic)
    return TestClient(app)


def test_health(client: TestClient) -> None:
    response = client.get("/v1")
    assert response.status_code == 200
    assert response.json() == {"status": True}


def test_models(client: TestClient) -> None:
    response = client.get("/v1/models")
    assert response.status_code == 200
    assert response.json() == MODELS


@pytest.mark.asyncio
async def test_chat(mock_anthropic: AsyncMock, client: TestClient) -> None:
    with patch("src.service.create_chat_completion") as mock_create_chat_completion:
        mock_create_chat_completion.return_value = ChatCompletion(
            id="msg-123",
            object="chat.completion",
            created=1677652288,
            model="claude-3-5-sonnet-20240620",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant", content="Test response"
                    ),
                    finish_reason="stop",
                )
            ],
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-3-5-sonnet-20240620",
                "messages": [{"role": "user", "content": "Test message"}],
                "stream": False,
            },
        )

    mock_create_chat_completion.assert_called_once_with(
        client=mock_anthropic,
        model="claude-3-5-sonnet-20240620",
        messages=[ChatCompletionUserMessageParam(role="user", content="Test message")],
    )
    assert response.status_code == 200
    assert response.json() == {
        "id": "msg-123",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Test response",
                    "function_call": None,
                    "refusal": None,
                    "tool_calls": None,
                },
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "created": 1677652288,
        "model": "claude-3-5-sonnet-20240620",
        "object": "chat.completion",
        "service_tier": None,
        "system_fingerprint": None,
        "usage": None,
    }


@pytest.mark.asyncio
async def test_chat_stream(mock_anthropic: AsyncMock, client: TestClient) -> None:
    with patch(
        "src.service.create_chat_completion_stream"
    ) as mock_create_chat_completion_stream:
        mock_create_chat_completion_stream.return_value = (
            f"data: {json.dumps({'model': 'claude-3-5-sonnet-20240620', 'choices': [{'delta': {'content': 'Test'}}]})}\n\n"
            f"data: {json.dumps({'model': 'claude-3-5-sonnet-20240620', 'choices': [{'delta': {'content': ' response'}}]})}\n\n"
            f"data: {json.dumps({'model': 'claude-3-5-sonnet-20240620', 'choices': [{'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-3-5-sonnet-20240620",
                "messages": [{"role": "user", "content": "Test message"}],
                "stream": True,
            },
        )

    mock_create_chat_completion_stream.assert_called_once_with(
        client=mock_anthropic,
        model="claude-3-5-sonnet-20240620",
        messages=[ChatCompletionUserMessageParam(role="user", content="Test message")],
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    chunks = list(line for line in response.iter_lines() if line)
    assert len(chunks) == 3

    for chunk in chunks:
        assert chunk.startswith("data: ")
        chunk_data = json.loads(chunk.split("data: ")[1])
        assert chunk_data["model"] == "claude-3-5-sonnet-20240620"
        assert "choices" in chunk_data

    # Check the final chunk
    final_chunk = json.loads(chunks[-1].split("data: ")[1])
    assert final_chunk["choices"][0]["finish_reason"] == "stop"
