from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from genai.chat import (
    GuardrailException,
    chat_with_guardrails,
    check_moderation,
    check_topic,
    moderation_guardrail,
    topic_guardrail,
)
from genai.prompts import (
    ANIMAL_ADVICE_CRITERIA,
    ANIMAL_ADVICE_STEPS,
    DOMAIN,
    MODERATION_SYSTEM_PROMPT,
    TOPIC_GUARDRAIL_PROMPT,
)


@pytest.fixture
def mock_openai() -> AsyncMock:
    mock = AsyncMock(spec=AsyncOpenAI)
    mock.chat = AsyncMock()
    mock.chat.completions.create = AsyncMock()
    return mock


@pytest.mark.asyncio
async def test_check_topic(mock_openai: AsyncMock) -> None:
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "allowed"
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await check_topic(
        client=mock_openai, model="gpt-4o", content="On-topic message"
    )

    mock_openai.chat.completions.create.assert_awaited_once_with(
        model="gpt-4o",
        messages=[
            ChatCompletionSystemMessageParam(
                role="system", content=TOPIC_GUARDRAIL_PROMPT
            ),
            ChatCompletionUserMessageParam(role="user", content="On-topic message"),
        ],
        temperature=0.0,
    )
    assert result == "allowed"


@pytest.mark.asyncio
@patch("genai.chat.check_topic")
async def test_topic_guardrail_not_allowed(
    mock_check_topic: AsyncMock, mock_openai: AsyncMock
) -> None:
    mock_check_topic.return_value = "not_allowed"

    with pytest.raises(GuardrailException) as e:
        await topic_guardrail(
            client=mock_openai, model="gpt-4o", content="Off-topic message"
        )

    assert str(e.value) == "Only topics related to dogs or cats are allowed!"
    mock_check_topic.assert_awaited_once_with(
        client=mock_openai, model="gpt-4o", content="Off-topic message"
    )


@pytest.mark.asyncio
async def test_check_moderation(mock_openai: AsyncMock) -> None:
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "2"
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await check_moderation(
        client=mock_openai, model="gpt-4o", content="Assistant response"
    )

    content = MODERATION_SYSTEM_PROMPT.format(
        domain=DOMAIN,
        scoring_criteria=ANIMAL_ADVICE_CRITERIA,
        scoring_steps=ANIMAL_ADVICE_STEPS,
        content="Assistant response",
    )

    mock_openai.chat.completions.create.assert_awaited_once_with(
        model="gpt-4o",
        messages=[ChatCompletionUserMessageParam(role="user", content=content)],
        temperature=0.0,
    )
    assert result == "2"


@pytest.mark.asyncio
@patch("genai.chat.check_moderation")
async def test_moderation_guardrail_not_permitted(
    mock_check_moderation: AsyncMock, mock_openai: AsyncMock
) -> None:
    mock_check_moderation.return_value = "5"

    with pytest.raises(GuardrailException) as e:
        await moderation_guardrail(
            client=mock_openai, model="gpt-4o", content="Not permitted message"
        )

    assert (
        str(e.value) == "Response skipped because animal breeding advice was detected!"
    )
    mock_check_moderation.assert_awaited_once_with(
        client=mock_openai, model="gpt-4o", content="Not permitted message"
    )


@pytest.mark.asyncio
@patch("genai.chat.moderation_guardrail")
@patch("genai.chat.topic_guardrail")
async def test_chat_with_guardrails_success(
    mock_topic_guardrail: AsyncMock,
    mock_moderation_guardrail: AsyncMock,
    mock_openai: AsyncMock,
):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "On-topic response"
    mock_openai.chat.completions.create.return_value = mock_response
    mock_topic_guardrail.return_value = None
    mock_moderation_guardrail.return_value = None

    messages = [ChatCompletionUserMessageParam(role="user", content="On-topic message")]
    result = await chat_with_guardrails(
        client=mock_openai, model="gpt-4o", messages=messages
    )

    assert result == mock_response
    mock_topic_guardrail.assert_awaited_once_with(
        client=mock_openai, model="gpt-4o", content="On-topic message"
    )
    mock_openai.chat.completions.create.assert_awaited_once_with(
        model="gpt-4o", messages=messages
    )
    mock_moderation_guardrail.assert_awaited_once_with(
        client=mock_openai, model="gpt-4o", content="On-topic response"
    )


@pytest.mark.asyncio
@patch("genai.chat.topic_guardrail")
async def test_chat_with_guardrails_topic_bad(
    mock_topic_guardrail: AsyncMock,
    mock_openai: AsyncMock,
) -> None:
    mock_topic_guardrail.side_effect = GuardrailException("Topic not allowed")

    messages = [
        ChatCompletionUserMessageParam(role="user", content="Off-topic message")
    ]

    completion = await chat_with_guardrails(
        client=mock_openai, model="gpt-4o", messages=messages
    )

    mock_topic_guardrail.assert_awaited_once_with(
        client=mock_openai, model="gpt-4o", content="Off-topic message"
    )
    completion_dict = completion.model_dump()
    assert completion_dict["choices"][0]["index"] == 0
    assert completion_dict["choices"][0]["message"]["content"] == "Topic not allowed"
    assert completion_dict["choices"][0]["finish_reason"] == "stop"
    assert completion_dict["model"] == "gpt-4o"
    assert completion_dict["object"] == "chat.completion"


@pytest.mark.asyncio
@patch("genai.chat.moderation_guardrail")
@patch("genai.chat.topic_guardrail")
async def test_chat_with_guardrail_moderation_bad(
    mock_topic_guardrail: AsyncMock,
    mock_moderation_guardrail: AsyncMock,
    mock_openai: AsyncMock,
) -> None:
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "On-topic response"
    mock_openai.chat.completions.create.return_value = mock_response
    mock_topic_guardrail.return_value = None
    mock_moderation_guardrail.side_effect = GuardrailException("Response not permitted")

    messages = [ChatCompletionUserMessageParam(role="user", content="On-topic message")]

    completion = await chat_with_guardrails(
        client=mock_openai, model="gpt-4o", messages=messages
    )

    mock_moderation_guardrail.assert_awaited_once_with(
        client=mock_openai, model="gpt-4o", content="On-topic response"
    )
    completion_dict = completion.model_dump()
    assert completion_dict["choices"][0]["index"] == 0
    assert (
        completion_dict["choices"][0]["message"]["content"] == "Response not permitted"
    )
    assert completion_dict["choices"][0]["finish_reason"] == "stop"
    assert completion_dict["model"] == "gpt-4o"
    assert completion_dict["object"] == "chat.completion"
