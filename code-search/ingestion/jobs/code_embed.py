import logging

import pandas as pd
from absl import app, flags
from datasets import Dataset
from openai import OpenAI

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_file",
    default="/data/code_chunks.parquet",
    help="Input parquet file of code chunks",
)
flags.DEFINE_string(
    "output_file",
    default="/data/code_embeddings.parquet",
    help="Output parquet file with code chunk embeddings",
)
flags.DEFINE_string(
    "model",
    default="text-embedding-3-small",
    help="OpenAI embedding model to use",
)


def main(argv):
    del argv  # Unused.

    df = pd.read_parquet(FLAGS.input_file)
    ds = Dataset.from_pandas(df)
    logging.info(f"Loaded dataset with {len(ds)} records from {FLAGS.input_file}")

    client = OpenAI()
    model = FLAGS.model

    def create_embeddings(batch: list[str], model: str) -> list[list[float]]:
        response = client.embeddings.create(input=batch, model=model)
        embeddings = [item.embedding for item in response.data]
        return embeddings

    ds = ds.filter(
        lambda x: len(x["text"]) > 0,
    )

    ds = ds.map(
        lambda x: {
            "embedding": create_embeddings(batch=x["text"], model=model),
        },
        batched=True,
        batch_size=32,
    )

    ds.to_parquet(FLAGS.output_file)
    logging.info(f"Saved embeddings to {FLAGS.output_file}")


if __name__ == "__main__":
    app.run(main)
