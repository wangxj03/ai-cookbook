from typing import Any

from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient


class CodeSearcher:
    def __init__(
        self,
        qdrant: AsyncQdrantClient,
        openai: AsyncOpenAI,
        embedding_model: str = "text-embedding-3-small",
    ):
        self.qdrant = qdrant
        self.openai = openai
        self.embedding_model = embedding_model

    async def search(
        self, query: str, collection_name: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        embedding_response = await self.openai.embeddings.create(
            input=query, model=self.embedding_model
        )
        embedding = embedding_response.data[0].embedding

        points = await self.qdrant.search(
            collection_name=collection_name,
            query_vector=embedding,
            limit=limit,
            with_payload=True,
        )

        results = []
        for point in points:
            payload = point.payload
            if point.payload is None:
                continue
            results.append(
                {
                    # "code_type": None,
                    "context": {
                        "file_name": payload["file_name"],
                        "file_path": payload["file_path"],
                        # "module": None,
                        "snippet": payload["text"],
                        # "struct_name": None,
                    },
                    # "docstring": None,
                    # "line": None,
                    "line_from": payload["start_line"],
                    "line_to": payload["end_line"],
                    # "name": None,
                    # "signature": None,
                }
            )

        return results
