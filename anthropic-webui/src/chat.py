import time
import uuid
from typing import AsyncIterator

from anthropic import AsyncAnthropic
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
)
from openai.types.chat.chat_completion import Choice as ChatCompletionChoice
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta


async def create_chat_completion(
    client: AsyncAnthropic,
    model: str,
    messages: list[ChatCompletionMessageParam],
) -> ChatCompletion:
    message = await client.messages.create(
        max_tokens=4096, messages=messages, model=model
    )
    completion = ChatCompletion(
        id=message.id,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionMessage(
                    content=message.content[0].text, role="assistant"
                ),
                finish_reason="stop",
            )
        ],
        created=int(time.time()),
        model=model,
        object="chat.completion",
    )
    return completion


async def create_chat_completion_stream(
    client: AsyncAnthropic,
    model: str,
    messages: list[ChatCompletionMessageParam],
) -> AsyncIterator[str]:
    async def stream_content() -> AsyncIterator[str]:
        async with client.messages.stream(
            max_tokens=4096,
            messages=messages,
            model=model,
        ) as stream:
            async for text in stream.text_stream:
                chunk = ChatCompletionChunk(
                    id=f"msg-{uuid.uuid4()}",
                    choices=[Choice(index=0, delta=ChoiceDelta(content=text))],
                    created=int(time.time()),
                    model=model,
                    object="chat.completion.chunk",
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

            # Send a final chunk to indicate the stream has ended
            chunk = ChatCompletionChunk(
                id=f"msg-{uuid.uuid4()}",
                choices=[Choice(index=0, delta=ChoiceDelta(), finish_reason="stop")],
                created=int(time.time()),
                model=model,
                object="chat.completion.chunk",
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

    return stream_content()
