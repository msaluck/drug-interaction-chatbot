# Drug Interaction Chatbot - AI Agent Instructions

## Project Overview

This is a local RAG (Retrieval-Augmented Generation) chatbot for querying drug-drug interactions. It runs offline using `llama-cpp-python` and uses a CSV knowledge base (`data/interactions_seed.csv`) instead of a vector database. The architecture relies on FastAPI for the interface and specialized regex/fuzzy matching for retrieval.

## Architecture & Core Components

- **Framework**: FastAPI (`src/main.py`) serving a LangChain pipeline (`src/chatbot.py`).
- **Data Source**: `data/interactions_seed.csv` loaded into a Pandas DataFrame.
- **Retrieval Mechanism** (`src/chatbot.py`):
  - **NOT** vector-based. Uses a custom 2-step process:
  1. **Regex Extraction**: Attempts to pull drug names from user queries using defined patterns.
  2. **Fuzzy Matching**: Uses `rapidfuzz` to match extracted terms against the CSV vocabulary.
- **LLM**: Local GGUF models via `llama-cpp-python`.
- **API Models**: Pydantic models in `src/main.py` define the request contract (`Question` model containing `mode`, `audience`, `verbosity`, etc.).

## Critical Workflows

### 1. Setup & Execution

- **One-Step Start**: Use the PowerShell helper which handles venv creation, dependency updates, and server execution.
  ```powershell
  .\scripts\start.ps1
  ```
- **Configuration**: Requires a `.env` file in the root with:
  ```properties
  LLAMA_CPP_MODEL=path/to/your/model.gguf
  ```
- **Manual Run**:
  ```bash
  uvicorn src.main:app --reload
  ```

### 2. Testing

- Run unit tests (configured for the retrieval logic):
  ```bash
  # Run from root
  python -m pytest tests/
  ```

## Code Conventions & Patterns

### explicit-resource-loading

- **Pattern**: Models and DataFrames are initialized at startup in `src/main.py` to prevent reloading on every request.
- **Example**:
  ```python
  # src/main.py
  chatbot_chain, db = create_chatbot_chain()
  ```

### custom-retrieval-logic

- **Retrieval**: Do not implement vector stores or embeddings unless specifically requested. Maintain the existing `retrieve_interaction_info` function in `src/chatbot.py` which queries the Pandas DataFrame directly.
- **Fuzzy Search**: Always use `rapidfuzz` for string matching to handle typos in drug names.

### api-response-formatting

- The `/ask` endpoint supports two modes controlled by the client:
  - `mode="data"`: Returns raw dictionary/text from the CSV (Bypasses LLM).
  - `mode="conversation"`: Puts the retrieved data into a prompt context for the LLM to generate a natural language response.

## Dependencies

- **Core**: `fastapi`, `uvicorn`, `langchain`, `llama-cpp-python`, `rapidfuzz`, `pandas`.
- **Environment**: Windows is the primary dev environment (evident from `.ps1` scripts and Visual Studio Build Tools requirements).
