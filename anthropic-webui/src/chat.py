import time
import uuid
from collections.abc import AsyncIterator, Iterable

from anthropic import AsyncAnthropic
from anthropic.types import (
    ImageBlockParam,
    MessageParam,
    TextBlock,
    TextBlockParam,
)
from anthropic.types.image_block_param import Source
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionContentPartParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
)
from openai.types.chat.chat_completion import Choice as ChatCompletionChoice
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta


def get_anthropic_image_source(url: str) -> Source:
    metadata, encoded_data = url.split(",", 1)
    media_type = metadata.split(";")[0].replace("data:", "")

    return Source(
        data=encoded_data,
        media_type=media_type,
        type="base64",
    )


def get_anthropic_message_content(
    content: str | Iterable[ChatCompletionContentPartParam],
) -> str | list[TextBlockParam | ImageBlockParam]:
    if isinstance(content, str):
        return content

    blocks: list[TextBlockParam | ImageBlockParam] = []
    for part in content:
        if part["type"] == "text":
            blocks.append(TextBlockParam(text=part["text"], type="text"))
        elif part["type"] == "image_url":
            image = part["image_url"]
            blocks.append(
                ImageBlockParam(
                    source=get_anthropic_image_source(image["url"]),
                    type="image",
                )
            )

    return blocks


def get_anthropic_messages(
    messages: list[ChatCompletionMessageParam],
) -> list[MessageParam]:
    return [
        MessageParam(
            content=get_anthropic_message_content(message["content"]),
            role=message["role"],
        )
        for message in messages
        if message["role"] in ["user", "assistant"]
    ]


async def create_chat_completion(
    client: AsyncAnthropic,
    model: str,
    messages: list[ChatCompletionMessageParam],
) -> ChatCompletion:
    message = await client.messages.create(
        max_tokens=4096, messages=get_anthropic_messages(messages), model=model
    )
    content = (
        message.content[0].text if isinstance(message.content[0], TextBlock) else ""
    )

    completion = ChatCompletion(
        id=message.id,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionMessage(content=content, role="assistant"),
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
            messages=get_anthropic_messages(messages),
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
