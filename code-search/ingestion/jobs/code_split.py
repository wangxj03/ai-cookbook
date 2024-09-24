import logging
import os
from collections.abc import Generator
from typing import Any

import pandas as pd
from absl import app, flags
from code_splitter import Language, TiktokenSplitter

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_dir", default=None, help="Input directory of rust files", required=True
)
flags.DEFINE_integer(
    "max_size",
    default=256,
    help="Maximum number of tokens for a single code chunk",
)
flags.DEFINE_string(
    "output_file",
    default="/data/code_chunks.parquet",
    help="Output parquet file with code chunk embeddings",
)


def walk(dir: str, max_size: int) -> Generator[dict[str, Any], None, None]:
    splitter = TiktokenSplitter(Language.Rust, max_size=max_size)

    for root, _, files in os.walk(dir):
        for file in files:
            if not file.endswith(".rs"):
                continue

            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, dir)

            with open(file_path, mode="r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            with open(file_path, mode="rb") as f:
                code = f.read()
                chunks = splitter.split(code)

                for chunk in chunks:
                    yield {
                        "file_path": rel_path,
                        "file_name": file,
                        "start_line": chunk.start,
                        "end_line": chunk.end,
                        "text": "\n".join(lines[chunk.start : chunk.end]),
                        "size": chunk.size,
                    }


def main(argv):
    del argv  # Unused.

    chunks = list(walk(dir=FLAGS.input_dir, max_size=FLAGS.max_size))
    logging.info(f"Found {len(chunks)} chunks in {FLAGS.input_dir}")

    # Convert to DataFrame and save to parquet
    df = pd.DataFrame(chunks)
    df.to_parquet(FLAGS.output_file)
    logging.info(f"Saved chunks to {FLAGS.output_file}")


if __name__ == "__main__":
    app.run(main)
