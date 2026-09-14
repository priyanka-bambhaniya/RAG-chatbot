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

Start the backend (from the `backend` directory, so the `app` package resolves):
```powershell
cd backend
uvicorn app.main:app --reload --port 8000
```

In a second terminal, start the frontend:
```powershell
cd frontend
streamlit run ui.py
```

Open the Streamlit URL, upload PDF/TXT/DOCX files in the sidebar to ingest them, then ask questions in the chat box.

## Notes

- Vector data persists on disk under `backend/chroma_data/` (gitignored).
- Supported document types: `.pdf`, `.txt`, `.docx`.
- Tune chunking/retrieval behavior via `.env` (`CHUNK_SIZE`, `CHUNK_OVERLAP`, `RETRIEVAL_TOP_K`) or the model names (`OPENAI_CHAT_MODEL`, `OPENAI_EMBEDDING_MODEL`).
