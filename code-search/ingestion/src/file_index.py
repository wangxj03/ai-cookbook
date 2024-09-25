import logging
import os
from typing import Any, Generator

from absl import app, flags
from qdrant_client import QdrantClient

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_dir",
    default=None,
    help="Input directory containing the rust files",
    required=True,
)
flags.DEFINE_string(
    "qdrant_host",
    default="http://localhost:6333",
    help="Qdrant host to connect to",
)
flags.DEFINE_string(
    "file_collection",
    default="qdrant-file",
    help="Qdrant collection name for files",
)


def walk(dir: str) -> Generator[dict[str, Any], None, None]:
    for root, _, files in os.walk(dir):
        for file in files:
            if not file.endswith(".rs"):
                continue

            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, dir)
            with open(file_path, mode="r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                yield {
                    "path": rel_path,
                    "code": lines,
                    "startline": 1,
                    "endline": len(lines),
                }


def main(argv):
    del argv  # Unused.

    files = list(walk(FLAGS.input_dir))
    logging.info(f"Found {len(files)} files in {FLAGS.input_dir}")

    client = QdrantClient(FLAGS.qdrant_host)
    client.recreate_collection(
        collection_name=FLAGS.file_collection,
        vectors_config={},
    )
    logging.info(f"Recreated collection {FLAGS.file_collection}")

    client.upload_collection(
        collection_name=FLAGS.file_collection,
        payload=files,
        vectors=[{}] * len(files),
        ids=None,
        batch_size=256,
    )
    logging.info(f"Uploaded {len(files)} files to collection {FLAGS.file_collection}")


if __name__ == "__main__":
    app.run(main)
