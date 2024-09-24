#!/bin/bash

python jobs/code_split.py --input_dir=/qdrant --output_file=/data/code_chunks.parquet
python jobs/code_embed.py --input_file=/data/code_chunks.parquet --output_file=/data/code_embeddings.parquet
python jobs/code_index.py --qdrant_host=http://qdrant:6333 --input_file=/data/code_embeddings.parquet
python jobs/file_index.py --qdrant_host=http://qdrant:6333 --input_dir=/qdrant
