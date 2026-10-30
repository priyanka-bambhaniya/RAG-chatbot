# RAG Production System

A Retrieval-Augmented Generation chatbot: FastAPI backend + Streamlit frontend, using OpenAI for embeddings/generation and ChromaDB as the vector store.

## Structure

```
rag-production-system/
├── backend/
│   ├── app/
│   │   ├── config.py     # settings loaded from .env
│   │   ├── pipeline.py   # ingestion, chunking, embedding, retrieval, generation
│   │   └── main.py       # FastAPI endpoints
│   └── requirements.txt
├── frontend/
│   ├── ui.py             # Streamlit chat UI
│   └── requirements.txt
├── .env
└── README.md
```

## Setup

1. Set your OpenAI key in `.env`:
   ```
   OPENAI_API_KEY=sk-...
   ```

2. Create a virtual environment and install dependencies for each part:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r backend\requirements.txt
   pip install -r frontend\requirements.txt
   ```

## Run

> **Activate the venv in every new terminal first.** Otherwise the global Python is used and imports fail (e.g. `No module named 'rank_bm25'`). Your prompt shows `(venv)` when it's active. The commands below are for PowerShell; in Command Prompt (cmd) use `venv\Scripts\activate.bat` instead.

Start the backend (from the `backend` directory, so the `app` package resolves):
```powershell
cd D:\learning\rag-chatbot\RAG-chatbot
.\venv\Scripts\Activate.ps1
cd backend
uvicorn app.main:app --reload --port 8000
```

In a second terminal, start the frontend:
```powershell
cd D:\learning\rag-chatbot\RAG-chatbot
.\venv\Scripts\Activate.ps1
cd frontend
streamlit run ui.py
```

See [COMMANDS.md](COMMANDS.md) for a quick command cheat sheet.

Open the Streamlit URL, upload PDF/TXT/DOCX files in the sidebar to ingest them, then ask questions in the chat box.

## Notes

- Vector data persists on disk under `backend/chroma_data/` (gitignored).
- Supported document types: `.pdf`, `.txt`, `.docx`.
- PDF pages with charts, images, card layouts or untitled tables are read by Gemini from a page image (`LLM_PROVIDER=gemini` only). Set `PDF_VISION` in `.env` to `auto` (default, visual pages only), `all` or `off`. Readings are cached in `backend/vision_cache/`, so re-ingesting the same file makes no API calls.
- Tune chunking/retrieval behavior via `.env` (`CHUNK_SIZE`, `CHUNK_OVERLAP`, `RETRIEVAL_TOP_K`) or the model names (`OPENAI_CHAT_MODEL`, `OPENAI_EMBEDDING_MODEL`).
