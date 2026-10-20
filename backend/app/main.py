import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from app.pipeline import SUPPORTED_EXTENSIONS, pipeline

app = FastAPI(title="RAG Production System")


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


class IngestResponse(BaseModel):
    filename: str
    chunks_ingested: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile) -> IngestResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        # Ingestion blocks (PDF parsing, embedding calls, retry sleeps); run it off the event loop
        # so the server keeps answering other requests meanwhile
        chunk_count = await run_in_threadpool(
            pipeline.ingest_file, tmp_path, source_name=file.filename
        )
    except Exception as exc:
        # Surface the real cause (e.g. embedding quota exhausted) instead of a bare 500
        raise HTTPException(status_code=502, detail=f"Ingestion failed: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    return IngestResponse(filename=file.filename or "unknown", chunks_ingested=chunk_count)


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty")

    result = pipeline.query(request.question, top_k=request.top_k)
    return QueryResponse(answer=result["answer"], sources=result["sources"])
