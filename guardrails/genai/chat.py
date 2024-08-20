import asyncio
import time
from uuid import uuid4

from fastapi import HTTPException
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion import Choice as ChatCompletionChoice
from pydantic import BaseModel

from genai.prompts import (
    ANIMAL_ADVICE_CRITERIA,
    ANIMAL_ADVICE_STEPS,
    DOMAIN,
    MODERATION_GUARDRAIL_PROMPT,
    TOPIC_GUARDRAIL_PROMPT,
)


class TopicGuardrailResult(BaseModel):
    allowed: bool


class ModerationGuardrailResult(BaseModel):
    score: int


class GuardrailException(Exception):
    pass


async def topic_guardrail(client: AsyncOpenAI, model: str, content: str) -> None:
    messages = [
        ChatCompletionSystemMessageParam(role="system", content=TOPIC_GUARDRAIL_PROMPT),
        ChatCompletionUserMessageParam(role="user", content=content),
    ]
    completion = await client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        temperature=0.0,
        response_format=TopicGuardrailResult,
    )
    parsed = completion.choices[0].message.parsed

    if parsed is None or not parsed.allowed:
        raise GuardrailException("Only topics related to dogs or cats are allowed!")


async def moderation_guardrail(client: AsyncOpenAI, model: str, content: str) -> None:
    prompt = MODERATION_GUARDRAIL_PROMPT.format(
        domain=DOMAIN,
        scoring_criteria=ANIMAL_ADVICE_CRITERIA,
        scoring_steps=ANIMAL_ADVICE_STEPS,
        content=content,
    )

    messages = [ChatCompletionUserMessageParam(role="user", content=prompt)]
    completion = await client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        temperature=0.0,
        response_format=ModerationGuardrailResult,
    )
    parsed = completion.choices[0].message.parsed

    if parsed is None or parsed.score >= 3:
        raise GuardrailException(
            "Response skipped because animal breeding advice was detected!"
        )


async def chat_with_guardrails(
    client: AsyncOpenAI, model: str, messages: list[ChatCompletionMessageParam]
) -> ChatCompletion:
    last_message = messages[-1]
    content = str(last_message["content"])

    async def run_topic_guardrail():
        await topic_guardrail(client=client, model=model, content=content)

    async def run_completion_and_moderation_guardrail():
        completion = await client.chat.completions.create(
            model=model, messages=messages
        )
        content = completion.choices[0].message.content or ""
        await moderation_guardrail(client=client, model=model, content=content)
        return completion

    try:
        topic_task = asyncio.create_task(run_topic_guardrail())
        completion_task = asyncio.create_task(run_completion_and_moderation_guardrail())

        # Wait for topic check to complete
        await topic_task

        # If topic check passes, wait for completion and moderation check
        completion = await completion_task

        return completion

    except GuardrailException as e:
        completion_task.cancel()
        error_message = str(e)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

    # If a GuardrailException was caught, return an error completion
    return ChatCompletion(
        id=f"chatcomp-{uuid4()}",
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=error_message),
                finish_reason="stop",
            )
        ],
        created=int(time.time()),
        model=model,
        object="chat.completion",
    )
