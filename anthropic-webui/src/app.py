from typing import Any

import uvicorn
from absl import app, flags
from anthropic import AsyncAnthropic

from src.service import create_app

FLAGS = flags.FLAGS
flags.DEFINE_integer("port", 8000, "Port number to run the FastAPI app on")


def main(argv: Any) -> None:
    del argv  # Unused

    client = AsyncAnthropic()
    fastapi_app = create_app(client=client)
    uvicorn.run(
        fastapi_app,
        host="0.0.0.0",
        port=FLAGS.port,
        use_colors=True,
    )


if __name__ == "__main__":
    app.run(main)
