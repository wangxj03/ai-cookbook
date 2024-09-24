import logging
from typing import Any

import uvicorn
from absl import app as absl_app
from absl import flags
from fastapi import FastAPI
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from starlette.staticfiles import StaticFiles

from backend.code_search import CodeSearcher
from backend.file_fetch import FileFetcher

logging.basicConfig(level=logging.DEBUG)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "qdrant_host",
    default="http://localhost:6333",
    help="Qdrant host to connect to.",
)
flags.DEFINE_string(
    "code_collection",
    default="qdrant-code",
    help="Qdrant collection name for code snippets.",
)
flags.DEFINE_string(
    "file_collection",
    default="qdrant-file",
    help="Qdrant collection name for files",
)
flags.DEFINE_integer("port", default=8000, help="Port number to run the FastAPI app on")


def main(argv):
    del argv  # Unused

    qdrant = AsyncQdrantClient(FLAGS.qdrant_host)
    openai = AsyncOpenAI()

    code_searcher = CodeSearcher(qdrant=qdrant, openai=openai)
    file_fetcher = FileFetcher(qdrant=qdrant)

    app = FastAPI()

    @app.get("/api/search")
    async def search(query: str) -> dict[str, Any]:
        logging.info(f"Searching with query: {query}")
        result = await code_searcher.search(
            query=query, collection_name=FLAGS.code_collection
        )
        return {"result": result}

    @app.get("/api/file")
    async def fetch(path: str) -> dict[str, Any]:
        logging.info(f"Fetching file at path: {path}")
        result = await file_fetcher.fetch(
            path=path, collection_name=FLAGS.file_collection
        )
        return {"result": result}

    # Need to clone https://github.com/qdrant/demo-code-search in the docker container, run `npm run build`
    # from the `frontend` directory, and then mount the `frontend/dist` directory.
    app.mount("/", StaticFiles(directory="./frontend/dist", html=True))

    uvicorn.run(app, host="0.0.0.0", port=FLAGS.port)


if __name__ == "__main__":
    absl_app.run(main)
