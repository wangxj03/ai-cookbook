from fastapi import FastAPI
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from pydantic import BaseModel

from src.chat import chat_with_guardrails


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatCompletionMessageParam]


def create_fastapi_app(client: AsyncOpenAI) -> FastAPI:
    fastapi_app = FastAPI()

    @fastapi_app.post("/v1/chat/completions", response_model=ChatCompletion)
    async def chat(request: ChatCompletionRequest) -> ChatCompletion:
        return await chat_with_guardrails(
            client=client, model=request.model, messages=request.messages
        )

    return fastapi_app
