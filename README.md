# Drug Interaction Chatbot (LangChain + llama.cpp + FastAPI)

A robust, local Retrieval-Augmented Generation (RAG) chatbot designed to answer questions about drug–drug interactions. It combines the power of local LLMs (via `llama-cpp-python`) with precise data retrieval from a CSV knowledge base.

## 🚀 Features

- **Local Inference**: Runs entirely offline using `llama.cpp` (no external API keys).
- **Robust Retrieval**:
  - **Regex Patterns**: Accurately extracts drug names from common query formats.
  - **Fuzzy Matching**: Handles typos and spelling errors using `rapidfuzz`.
  - **Conversational Fallback**: Scans user input against the entire drug vocabulary if specific patterns aren't found.
- **Customizable Output**:
  - **Modes**: Choose between `conversation` (LLM-generated natural language) or `data` (raw database records).
  - **Audience Adaptation**: Tailor responses for a `patient` (simple, reassuring) or a `pharmacist` (clinical, technical).
  - **Formatting**: Request output in `text` or `json`.
  - **Verbosity**: Control response length (`concise`, `normal`, `detailed`).
  - **Multi-language Support**: Generate responses in different languages.
- **Streaming**: Real-time token streaming via Server-Sent Events (SSE).
- **Observability**: Integrated logging for debugging and monitoring.

## 📂 Repository Structure

```
data/
    interactions_seed.csv        # Knowledge base (Drug A, Drug B, Severity, Description)
src/
    chatbot.py                   # Core logic: Retrieval, Fuzzy Matching, Prompt Engineering
    main.py                      # FastAPI application & Pydantic models
    __init__.py
tests/
    test_retriever.py            # Unit tests
requirements.txt                 # Python dependencies
README.md                        # Documentation
```

## 🛠️ Prerequisites

- **Python 3.10+**
- **Build Tools**: CMake + Visual Studio Build Tools (for compiling `llama-cpp-python` on Windows).
- **Model**: A GGUF format LLM (e.g., Mistral, Llama 3, Phi-3).

## 📦 Installation

1.  **Clone the repository**:

    ```bash
    git clone <repository-url>
    cd drug_interaction_chatbot
    ```

2.  **Create a Virtual Environment**:

    ```powershell
    python -m venv .venv
    .\.venv\Scripts\Activate
    ```

3.  **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Download a Model**:
    Download a GGUF model (e.g., `mistral-7b-instruct-v0.2.Q4_K_M.gguf`) from Hugging Face and place it in a `models/` directory.

5.  **Configure Environment**:
    Create a `.env` file in the root directory:
    ```env
    LLAMA_CPP_MODEL=models/mistral-7b-instruct-v0.2.Q4_K_M.gguf
    LLAMA_CPP_CTX=4096
    LLAMA_CPP_THREADS=8
    LLAMA_CPP_TEMPERATURE=0.2
    ```

## 🚀 Running the Server

Start the FastAPI server using Uvicorn:

```powershell
uvicorn src.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

## 📖 API Documentation

### 1. POST `/ask` (Standard Response)

Get a complete response in a single JSON object.

**Endpoint**: `http://127.0.0.1:8000/ask`

**Request Body Parameters**:

| Parameter   | Type     | Default          | Description                                                      |
| :---------- | :------- | :--------------- | :--------------------------------------------------------------- |
| `text`      | `string` | **Required**     | The user's question (e.g., "Can I take Aspirin with Warfarin?"). |
| `mode`      | `string` | `"conversation"` | `"conversation"` for LLM answer, `"data"` for raw CSV rows.      |
| `audience`  | `string` | `"patient"`      | `"patient"` (simple) or `"pharmacist"` (technical).              |
| `language`  | `string` | `"English"`      | Target language for the response.                                |
| `verbosity` | `string` | `"normal"`       | `"concise"`, `"normal"`, or `"detailed"`.                        |
| `format`    | `string` | `"text"`         | `"text"` or `"json"`.                                            |

**Example Request (Patient Mode):**

```json
{
  "text": "I have a headache, can I take ibuprofen if I'm on lithium?",
  "audience": "patient",
  "verbosity": "concise"
}
```

**Example Request (Pharmacist Mode - Raw Data):**

```json
{
  "text": "interaction lithium ibuprofen",
  "mode": "data"
}
```

### 2. POST `/stream` (Streaming Response)

Stream the response token-by-token using Server-Sent Events (SSE). Useful for real-time UI updates.

**Endpoint**: `http://127.0.0.1:8000/stream`

**Request Body**: Same as `/ask`.

**Example Client (Python):**

```python
import requests

url = "http://127.0.0.1:8000/stream"
payload = {"text": "Explain the interaction between Warfarin and Aspirin", "audience": "pharmacist"}

with requests.post(url, json=payload, stream=True) as r:
    for chunk in r.iter_content(1024):
        print(chunk.decode("utf-8"), end="")
```

## 🔍 Logging

The application uses Python's built-in `logging` module.

- **INFO**: Startup events, incoming requests, retrieval results.
- **ERROR**: Exceptions and processing failures.

Logs are output to the console (stderr) by default.

## 🧪 Testing

Run the included tests to verify retrieval logic:

```powershell
pytest tests/
```

## 📝 License

[Your License Here]
