from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models


class FileFetcher:
    def __init__(self, qdrant: AsyncQdrantClient) -> None:
        self.qdrant = qdrant

    async def fetch(
        self, path: str, collection_name: str, limit: int = 5
    ) -> list[dict]:
        # https://github.com/qdrant/qdrant-client/blob/d18cb1702f4cf8155766c7b32d1e4a68af11cd6a/qdrant_client/async_qdrant_client.py#L829
        points, _ = await self.qdrant.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="path",
                        match=models.MatchValue(value=path),
                    )
                ]
            ),
            limit=limit,
        )

        return [point.payload for point in points if point.payload is not None]
