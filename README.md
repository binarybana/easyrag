# LLM RAG

A RAG (Retrieval Augmented Generation) implementation using LlamaIndex for document processing, Gemini for embeddings, and LanceDB for vector storage.

## Setup

This project uses `uv` for dependency management and `direnv` for environment management. To get started:

1. Install dependencies:
```bash
# Create and activate a new virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e .
```

2. Set up environment:
```bash
# Create .env file with your Google API key
echo "GOOGLE_API_KEY=your_key_here" > .env

# Allow direnv to load the environment
direnv allow
```

## Usage

### Data Ingestion

```bash
python -m llm_rag.ingest --source /path/to/source --type [code|url|pdf]
```

### Search Server

```bash
python -m llm_rag.search --db /path/to/lancedb
