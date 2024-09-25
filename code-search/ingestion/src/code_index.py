import logging

import pandas as pd
from absl import app, flags
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_file",
    default="/data/code_embeddings.parquet",
    help="Input parquet file of code chunk embeddings",
)
flags.DEFINE_string(
    "qdrant_host",
    default="http://localhost:6333",
    help="Qdrant host to connect to",
)
flags.DEFINE_string(
    "code_collection",
    default="qdrant-code",
    help="Qdrant collection name for code snippets",
)
flags.DEFINE_integer(
    "embedding_dim",
    # 1536 for `text-embedding-3-small` and 3072 for `text-embedding-3-large`
    default=1536,
    help="Embedding dimension",
)


def main(argv):
    del argv  # Unused.

    df = pd.read_parquet(FLAGS.input_file)
    logging.info(f"Loaded {len(df)} records from {FLAGS.input_file}")

    client = QdrantClient(FLAGS.qdrant_host)

    client.recreate_collection(
        collection_name=FLAGS.code_collection,
        vectors_config=VectorParams(
            size=FLAGS.embedding_dim,
            distance=Distance.COSINE,
        ),
    )
    logging.info(f"Recreated collection {FLAGS.code_collection}")

    # Record oriented upload
    client.upload_points(
        collection_name=FLAGS.code_collection,
        points=[
            PointStruct(
                id=idx,
                vector=row["embedding"][: FLAGS.embedding_dim].tolist(),
                payload=row.drop(["embedding"]).to_dict(),
            )
            for idx, row in df.iterrows()
        ],
    )
    logging.info(f"Uploaded {len(df)} points to collection {FLAGS.code_collection}")


if __name__ == "__main__":
    app.run(main)
