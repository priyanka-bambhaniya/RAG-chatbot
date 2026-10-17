import re
import time
import uuid
from pathlib import Path

import chromadb
from docx import Document as DocxDocument
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
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
        self.provider = settings.llm_provider.lower()
        if self.provider == "gemini":
            if not settings.gemini_api_key:
                raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")
            self.genai_client = genai.Client(api_key=settings.gemini_api_key)
        elif self.provider == "openai":
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
            self.openai_client = OpenAI(api_key=settings.openai_api_key)
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER '{settings.llm_provider}'")

        self.chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self.collection = self.chroma_client.get_or_create_collection(
            name=settings.chroma_collection_name
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self.provider == "gemini":
            embeddings: list[list[float]] = []
            # Gemini caps embed_content at 100 items per request; smaller batches stay under per-minute quota
            batch_size = min(settings.embed_batch_size, 100)
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                response = self._gemini_embed_with_retry(batch)
                embeddings.extend(embedding.values for embedding in response.embeddings)
            return embeddings
        response = self.openai_client.embeddings.create(
            model=settings.openai_embedding_model, input=texts
        )
        return [item.embedding for item in response.data]

    def _gemini_embed_with_retry(self, batch: list[str]):
        for attempt in range(settings.embed_max_retries + 1):
            try:
                return self.genai_client.models.embed_content(
                    model=settings.gemini_embedding_model, contents=batch
                )
            except genai_errors.ClientError as exc:
                if exc.code != 429 or attempt == settings.embed_max_retries:
                    raise
                # Honor the server's suggested retryDelay (e.g. "23s"), else back off exponentially
                match = re.search(r"retryDelay['\"]?:\s*['\"](\d+(?:\.\d+)?)s", str(exc))
                delay = float(match.group(1)) + 1 if match else min(2 ** attempt * 5, 60)
                time.sleep(delay)

    def ingest_file(self, file_path: Path, source_name: str | None = None) -> int:
        source_name = source_name or file_path.name
        self.collection.delete(where={"source": source_name})
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

        user_content = f"Context:\n{context_block}\n\nQuestion: {query}"

        if self.provider == "gemini":
            response = self.genai_client.models.generate_content(
                model=settings.gemini_chat_model,
                contents=user_content,
                config=genai_types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
            )
            return response.text or ""

        response = self.openai_client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        return response.choices[0].message.content or ""

    def query(self, question: str, top_k: int | None = None) -> dict:
        context_chunks = self.retrieve(question, top_k)
        answer = self.generate_answer(question, context_chunks)
        sources = sorted({c["source"] for c in context_chunks if c["source"]})
        return {"answer": answer, "sources": sources, "context": context_chunks}


pipeline = RAGPipeline()
