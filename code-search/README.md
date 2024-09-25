# Semantic Code Search

This application is adapted from https://github.com/qdrant/demo-code-search/tree/master to demonstrate semantic code search using Qdrant.

## Ingestion

The [`ingestion`](./ingestion/) directory contains the code and configuration files for the ingestion pipeline, responsible for processing and indexing a Rust codebase into a vector database (Qdrant). The ingestion pipeline follows these steps:

1. **Code Splitting**: Split code into smaller chunks using the `code_splitter` library.
2. **Code Embedding**: Generate embeddings for code chunks using OpenAI.
3. **Code Indexing**: Index code chunk embeddings into a Qdrant collection (`qdrant-code`), along with associated metadata (file path, line numbers, etc.).
4. **File Indexing**: Index code files into a separate Qdrant collection (`qdrant-file`) for retrieving full file content when needed.

Compared to the original implementation, we directly use OpenAI's embedding model instead of the open-source models. This significantly reduces the complexity of the ingestion pipeline.

To run the ingestion pipeline in a containerized environment, execute the following command from the `ingestion` directory:

```sh
just run
```

Both Qdrant collections will be stored in the `qdrant-storage` volume and mounted to the `qdrant` container when starting the code search server.

## Backend

The [`backend`](./backend/) directory contains the backend code for the semantic code search server. It is built using FastAPI and handles REST requests to interact with the Qdrant vector database. It exposes two endpoints:

- **`GET /api/search`**: Searches for code snippets based on a query.
- **`GET /api/file`**: Fetches the full content of a file based on its path.

To start the server, run the following command from the `backend` directory:

```sh
just run
```

Visit http://localhost:8000 to access the code search interface.
