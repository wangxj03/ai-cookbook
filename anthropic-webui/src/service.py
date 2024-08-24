from typing import Any

from anthropic import AsyncAnthropic
from fastapi import FastAPI
from fastapi.responses import Response, StreamingResponse
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from src.chat import (
    create_chat_completion,
    create_chat_completion_stream,
)

MODELS = {
    "object": "list",
    "data": [
        {
            "id": "claude-3-5-sonnet-20240620",
            "model": "claude-3.5-sonnet",
            "object": "model",
            "owned_by": "anthropic",
        },
        {
            "id": "claude-3-haiku-20240307",
            "model": "claude-3-haiku",
            "object": "model",
            "owned_by": "anthropic",
        },
        {
            "id": "claude-3-sonnet-20240229",
            "model": "claude-3-sonnet",
            "object": "model",
            "owned_by": "anthropic",
        },
        {
            "id": "claude-3-opus-20240229",
            "model": "claude-3-opus",
            "object": "model",
            "owned_by": "anthropic",
        },
    ],
}


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatCompletionMessageParam]
    stream: bool = True


def create_app(client: AsyncAnthropic) -> FastAPI:
    app = FastAPI()

    @app.get("/v1")
    async def health() -> dict[str, Any]:
        return {"status": True}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return MODELS

    @app.post("/v1/chat/completions")
    async def chat(request: ChatCompletionRequest) -> Any:
        if request.stream:
            # Streaming when responding to user messages
            stream_content = await create_chat_completion_stream(
                client=client,
                model=request.model,
                messages=request.messages,
            )
            return StreamingResponse(
                content=stream_content, media_type="text/event-stream"
            )

        # Non-streaming when creating titles for conversations
        completion = await create_chat_completion(
            client=client,
            model=request.model,
            messages=request.messages,
        )
        return Response(
            content=completion.model_dump_json(), media_type="application/json"
        )

    return app
