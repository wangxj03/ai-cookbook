#!/bin/bash

python src/code_split.py --input_dir=/qdrant --output_file=/data/code_chunks.parquet
python src/code_embed.py --input_file=/data/code_chunks.parquet --output_file=/data/code_embeddings.parquet
python src/code_index.py --qdrant_host=http://qdrant:6333 --input_file=/data/code_embeddings.parquet
python src/file_index.py --qdrant_host=http://qdrant:6333 --input_dir=/qdrant
