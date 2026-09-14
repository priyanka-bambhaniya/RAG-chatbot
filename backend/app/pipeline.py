import uuid
from pathlib import Path

import chromadb
from docx import Document as DocxDocument
from openai import OpenAI
from pypdf import PdfReader

from app.config import settings

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using only the provided "
    "context. If the context does not contain the answer, say you don't know "
    "instead of guessing. Cite the source filename(s) you relied on."
)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}


def extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".docx":
        doc = DocxDocument(str(file_path))
        return "\n".join(p.text for p in doc.paragraphs)
    if suffix == ".txt":
        return file_path.read_text(encoding="utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type: {suffix}")


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return chunks


class RAGPipeline:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self.collection = self.chroma_client.get_or_create_collection(
            name=settings.chroma_collection_name
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=settings.openai_embedding_model, input=texts
        )
        return [item.embedding for item in response.data]

    def ingest_file(self, file_path: Path, source_name: str | None = None) -> int:
        source_name = source_name or file_path.name
        text = extract_text(file_path)
        chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            return 0

        embeddings = self._embed(chunks)
        ids = [f"{source_name}-{uuid.uuid4().hex[:8]}-{i}" for i in range(len(chunks))]
        metadatas = [{"source": source_name, "chunk_index": i} for i in range(len(chunks))]

        self.collection.add(
            ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas
        )
        return len(chunks)

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        top_k = top_k or settings.retrieval_top_k
        query_embedding = self._embed([query])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=top_k
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        return [
            {"text": doc, "source": meta.get("source"), "distance": dist}
            for doc, meta, dist in zip(documents, metadatas, distances)
        ]

    def generate_answer(self, query: str, context_chunks: list[dict]) -> str:
        if not context_chunks:
            context_block = "No relevant context was found."
        else:
            context_block = "\n\n".join(
                f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
            )

        response = self.client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{context_block}\n\nQuestion: {query}",
                },
            ],
        )
        return response.choices[0].message.content or ""

    def query(self, question: str, top_k: int | None = None) -> dict:
        context_chunks = self.retrieve(question, top_k)
        answer = self.generate_answer(question, context_chunks)
        sources = sorted({c["source"] for c in context_chunks if c["source"]})
        return {"answer": answer, "sources": sources, "context": context_chunks}


pipeline = RAGPipeline()
