from typing import Any

import uvicorn
from absl import app, flags
from langfuse.openai import AsyncOpenAI

from genai.service import create_fastapi_app

FLAGS = flags.FLAGS
flags.DEFINE_integer("port", 8000, "Port number to run the FastAPI app on")


def main(argv: Any) -> None:
    del argv  # Unused

    client = AsyncOpenAI()
    fastapi_app = create_fastapi_app(client=client)
    uvicorn.run(
        fastapi_app,
        host="0.0.0.0",
        port=FLAGS.port,
        use_colors=True,
    )


if __name__ == "__main__":
    app.run(main)
