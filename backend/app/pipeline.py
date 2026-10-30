import logging
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
from rank_bm25 import BM25Okapi

from app.config import settings
from app.pdf_layout import Segment, parse_pdf

SYSTEM_PROMPT = """You answer questions about the user's uploaded documents using only the numbered context excerpts.

Rules:
- Use only facts stated in the excerpts. If they do not contain the answer, say you don't know; never guess or use outside knowledge.
- Excerpts come from prose, boxes, tables and figure descriptions. Table excerpts give one row per line with every value labelled by its column, e.g. "China: 2023: 5.4; 2026f: 4.2". For numbers, prefer table rows and quote the column exactly.
- In forecast tables "e" means estimate and "f" means forecast (2026f = forecast for 2026). Values under a heading like "Percentage-point differences from January 2026 projections" are revisions to an earlier forecast, not growth rates. ".." means not available.
- Apply notes attached to a row, such as figures reported on a fiscal-year basis.
- When excerpts state the same figure with different precision, give the most specific one.
- Figure excerpts carry a chart's title, summary, panel titles and notes, but not the plotted values; don't infer values from them.
- Cite each fact as (filename, p. N), adding the table, figure or box name when the excerpt has one."""

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}
# Reference lists match almost any query on vocabulary but rarely answer one
EXCLUDED_KINDS = ["references"]
RRF_K = 60  # reciprocal-rank-fusion constant
STOPWORDS = set(
    "a an and are as at be by for from has have how in is it its of on or that the this to was were what "
    "when where which who will with does did do than into about".split()
)

# Use uvicorn's logger so progress lines show up in the server console
logger = logging.getLogger("uvicorn.error")


def extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
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


def _split_long(text: str, chunk_size: int) -> list[str]:
    """Split an oversized paragraph on sentence ends, repeating one sentence for overlap."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces, current = [], []
    for sentence in sentences:
        if current and len(" ".join(current)) + len(sentence) > chunk_size:
            pieces.append(" ".join(current))
            current = current[-1:]
        current.append(sentence)
    if current:
        pieces.append(" ".join(current))
    return pieces


def _page_label(start: int, end: int) -> str:
    if not start:
        return ""
    return f"p. {start}" if start == end else f"pp. {start}-{end}"


def build_chunks(segments: list[Segment], chunk_size: int) -> list[tuple[str, dict]]:
    """Pack consecutive paragraphs of one section into chunks; tables and figures stay whole.

    Each chunk starts with its section path and pages so both the embedding and the
    LLM see where the text sits in the document.
    """
    chunks: list[tuple[str, dict]] = []
    buffer: list[Segment] = []

    def emit(body: str, first: Segment, page_end: int) -> None:
        header = " | ".join(p for p in [first.section, _page_label(first.page_start, page_end)] if p)
        text = f"[{header}]\n{body}" if header else body
        chunks.append(
            (
                text,
                {
                    "kind": first.kind,
                    "section": first.section[:500],
                    "page_start": first.page_start,
                    "page_end": page_end,
                    "table_id": first.table_id,
                },
            )
        )

    def flush() -> None:
        if buffer:
            emit("\n\n".join(s.text for s in buffer), buffer[0], max(s.page_end for s in buffer))
            buffer.clear()

    for segment in segments:
        if segment.standalone:
            flush()
            emit(segment.text, segment, segment.page_end)
            continue
        if buffer and (
            segment.kind != buffer[0].kind
            or segment.section != buffer[0].section
            or sum(len(s.text) for s in buffer) + len(segment.text) > chunk_size
        ):
            flush()
        if len(segment.text) > chunk_size:
            for piece in _split_long(segment.text, chunk_size):
                emit(piece, segment, segment.page_end)
            continue
        buffer.append(segment)
    flush()
    return chunks


def _tokenize(text: str) -> list[str]:
    tokens = []
    for token in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower()):
        if token in STOPWORDS:
            continue
        tokens.append(token)
        # "2026f" (forecast) should also match a question about "2026"
        if match := re.fullmatch(r"((?:19|20)\d{2})[a-z]{1,2}", token):
            tokens.append(match.group(1))
    return tokens


class RAGPipeline:
    def __init__(self) -> None:
        self.provider = settings.llm_provider.lower()
        self.embedding_provider = (settings.embedding_provider or self.provider).lower()
        if self.provider not in ("openai", "gemini"):
            raise ValueError(f"Unsupported LLM_PROVIDER '{settings.llm_provider}'")
        if self.embedding_provider not in ("local", "openai", "gemini"):
            raise ValueError(f"Unsupported EMBEDDING_PROVIDER '{settings.embedding_provider}'")

        needed = {self.provider, self.embedding_provider}
        if "gemini" in needed:
            if not settings.gemini_api_key:
                raise ValueError("GEMINI_API_KEY is required for the Gemini provider")
            # The SDK's built-in retries can stall a request for minutes on 503s; retries and
            # model fallback are handled here instead so the user gets an answer or a clear error fast
            self.genai_client = genai.Client(
                api_key=settings.gemini_api_key,
                http_options=genai_types.HttpOptions(
                    retry_options=genai_types.HttpRetryOptions(attempts=1)
                ),
            )
        if "openai" in needed:
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required for the OpenAI provider")
            self.openai_client = OpenAI(api_key=settings.openai_api_key)
        self._local_model = None

        self.chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=chromadb.config.Settings(anonymized_telemetry=False),
        )
        # Embedding sizes differ per provider, so each provider gets its own collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=f"{settings.chroma_collection_name}-{self.embedding_provider}",
            metadata={"hnsw:space": "cosine"},
        )
        self._bm25: tuple[BM25Okapi, list[tuple[str, str, dict]]] | None = None

    # ----- embeddings ---------------------------------------------------------

    def _embed(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        if self.embedding_provider == "local":
            if self._local_model is None:
                from fastembed import TextEmbedding  # loads ONNX runtime; only when used

                logger.info("Loading local embedding model %s", settings.local_embedding_model)
                self._local_model = TextEmbedding(
                    settings.local_embedding_model, cache_dir=settings.local_model_cache_dir
                )
            vectors = (
                self._local_model.query_embed(texts) if is_query else self._local_model.passage_embed(texts)
            )
            return [vector.tolist() for vector in vectors]

        if self.embedding_provider == "gemini":
            embeddings: list[list[float]] = []
            # Gemini caps embed_content at 100 items per request; smaller batches stay under per-minute quota
            batch_size = min(settings.embed_batch_size, 100)
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                response = self._gemini_embed_with_retry(batch)
                embeddings.extend(embedding.values for embedding in response.embeddings)
                logger.info("Embedded %d/%d chunks", len(embeddings), len(texts))
            return embeddings

        embeddings = []
        for i in range(0, len(texts), 256):
            response = self.openai_client.embeddings.create(
                model=settings.openai_embedding_model, input=texts[i : i + 256]
            )
            embeddings.extend(item.embedding for item in response.data)
        return embeddings

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
                if delay > settings.embed_max_retry_wait:
                    # Daily quota exhausted (Gemini asks to wait hours); fail now with a clear reason
                    raise RuntimeError(
                        f"Gemini embedding quota exhausted; it resets in about {delay / 3600:.1f}h. "
                        "Wait for the reset, use another API key, or set EMBEDDING_PROVIDER=local."
                    ) from exc
                logger.warning("Gemini rate limit hit; retrying in %.0fs", delay)
                time.sleep(delay)

    # ----- ingestion ----------------------------------------------------------

    def ingest_file(self, file_path: Path, source_name: str | None = None) -> int:
        source_name = source_name or file_path.name
        if file_path.suffix.lower() == ".pdf":
            segments = parse_pdf(file_path)
        else:
            text = extract_text(file_path)
            segments = [
                Segment(chunk, "text", 0, 0, "")
                for chunk in chunk_text(text, settings.chunk_size, settings.chunk_overlap)
            ]
        chunks = build_chunks(segments, settings.chunk_size)
        if not chunks:
            return 0

        logger.info("Ingesting %s: %d chunks to embed", source_name, len(chunks))
        started = time.time()
        embeddings = self._embed([text for text, _ in chunks])
        logger.info("Embedded %s in %.0fs", source_name, time.time() - started)

        # Replace the previous version only once the new one is fully embedded
        self.collection.delete(where={"source": source_name})
        run = uuid.uuid4().hex[:8]
        for start in range(0, len(chunks), 1000):
            batch = chunks[start : start + 1000]
            self.collection.add(
                ids=[f"{source_name}-{run}-{start + i}" for i in range(len(batch))],
                embeddings=embeddings[start : start + len(batch)],
                documents=[text for text, _ in batch],
                metadatas=[
                    {**meta, "source": source_name, "chunk_index": start + i}
                    for i, (_, meta) in enumerate(batch)
                ],
            )
        self._bm25 = None
        return len(chunks)

    def list_documents(self) -> dict[str, int]:
        """Ingested files and their chunk counts."""
        counts: dict[str, int] = {}
        for meta in self.collection.get(include=["metadatas"])["metadatas"]:
            source = meta.get("source", "unknown")
            counts[source] = counts.get(source, 0) + 1
        return dict(sorted(counts.items()))

    def delete_document(self, source_name: str) -> int:
        """Remove every chunk of one ingested file; returns how many were removed."""
        ids = self.collection.get(where={"source": source_name})["ids"]
        if ids:
            self.collection.delete(ids=ids)
            self._bm25 = None
        return len(ids)

    # ----- retrieval ----------------------------------------------------------

    def _keyword_index(self) -> tuple[BM25Okapi, list[tuple[str, str, dict]]] | None:
        if self._bm25 is None:
            data = self.collection.get(include=["documents", "metadatas"])
            rows = [
                (id_, doc, meta)
                for id_, doc, meta in zip(data["ids"], data["documents"], data["metadatas"])
                if meta.get("kind") not in EXCLUDED_KINDS
            ]
            if not rows:
                return None
            self._bm25 = (BM25Okapi([_tokenize(doc) for _, doc, _ in rows]), rows)
        return self._bm25

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """Hybrid search: vectors catch paraphrases, BM25 catches names, years and numbers.

        The two rankings are merged with reciprocal rank fusion.
        """
        top_k = top_k or settings.retrieval_top_k
        count = self.collection.count()
        if count == 0:
            return []
        candidates = min(count, max(top_k * 4, 20))

        scores: dict[str, float] = {}
        found: dict[str, tuple[str, dict]] = {}

        results = self.collection.query(
            query_embeddings=[self._embed([query], is_query=True)[0]],
            n_results=candidates,
            where={"kind": {"$nin": EXCLUDED_KINDS}},
        )
        for rank, (id_, doc, meta) in enumerate(
            zip(results["ids"][0], results["documents"][0], results["metadatas"][0])
        ):
            scores[id_] = scores.get(id_, 0) + 1 / (RRF_K + rank + 1)
            found[id_] = (doc, meta)

        index = self._keyword_index()
        tokens = _tokenize(query)
        if index and tokens:
            bm25, rows = index
            keyword_scores = bm25.get_scores(tokens)
            best = sorted(range(len(rows)), key=lambda i: keyword_scores[i], reverse=True)[:candidates]
            for rank, i in enumerate(i for i in best if keyword_scores[i] > 0):
                id_, doc, meta = rows[i]
                scores[id_] = scores.get(id_, 0) + 1 / (RRF_K + rank + 1)
                found[id_] = (doc, meta)

        ranked = sorted(scores, key=scores.get, reverse=True)
        hits = [self._hit(*found[id_], score=scores[id_]) for id_ in ranked]
        # Rows of one table collapse into a single block, so fill the freed slots further down
        return self._expand_tables(hits)[:top_k]

    @staticmethod
    def _hit(doc: str, meta: dict, score: float = 0.0) -> dict:
        return {
            "text": doc,
            "source": meta.get("source"),
            "kind": meta.get("kind", "text"),
            "section": meta.get("section", ""),
            "page_start": meta.get("page_start", 0),
            "page_end": meta.get("page_end", 0),
            "table_id": meta.get("table_id", ""),
            "score": score,
        }

    def _expand_tables(self, hits: list[dict]) -> list[dict]:
        """Swap matched table rows for the whole table, so comparisons across rows work."""
        tables = []
        for hit in hits:
            key = (hit["source"], hit["table_id"])
            if hit["kind"] == "table" and hit["table_id"] and key not in tables:
                tables.append(key)
        expanded: dict[tuple, dict] = {}
        for source, table_id in tables[: settings.max_expanded_tables]:
            data = self.collection.get(
                where={"$and": [{"source": source}, {"table_id": table_id}]},
                include=["documents", "metadatas"],
            )
            rows = sorted(zip(data["documents"], data["metadatas"]), key=lambda r: r[1]["chunk_index"])
            if not rows:
                continue
            lines = [rows[0][0]]  # overview: title, columns, row names, notes
            for doc, _ in rows[1:]:
                body = doc.split("\n", 1)[-1]
                lines.append(body.split(" — ", 1)[-1] if " — " in body else body)
            text = "\n".join(lines)
            if len(text) <= settings.max_table_chars:
                expanded[(source, table_id)] = self._hit(text, rows[0][1])

        out, used = [], set()
        for hit in hits:
            key = (hit["source"], hit["table_id"])
            if key in expanded:
                if key not in used:
                    out.append(expanded[key])
                    used.add(key)
                continue
            out.append(hit)
        return out

    # ----- generation ---------------------------------------------------------

    def generate_answer(self, query: str, context_chunks: list[dict]) -> str:
        if not context_chunks:
            context_block = "No relevant context was found."
        else:
            blocks = []
            for i, chunk in enumerate(context_chunks, start=1):
                where = ", ".join(
                    p for p in [chunk["source"], _page_label(chunk["page_start"], chunk["page_end"]), chunk["kind"]] if p
                )
                blocks.append(f"[{i}] ({where})\n{chunk['text']}")
            context_block = "\n\n".join(blocks)

        user_content = f"Context:\n{context_block}\n\nQuestion: {query}"

        if self.provider == "gemini":
            return self._gemini_generate(user_content)

        response = self.openai_client.chat.completions.create(
            model=settings.openai_chat_model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        return response.choices[0].message.content or ""

    def _gemini_generate(self, user_content: str) -> str:
        """Try the main chat model, then each fallback, retrying briefly on overload or rate limits."""
        fallbacks = [m.strip() for m in settings.gemini_fallback_models.split(",") if m.strip()]
        models = [settings.gemini_chat_model] + [m for m in fallbacks if m != settings.gemini_chat_model]
        config = genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
            # No tools are passed, so AFC is unused; disabling it silences the SDK warning
            automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
        )
        last_error: Exception | None = None
        for model in models:
            for attempt in range(settings.chat_retries_per_model):
                try:
                    response = self.genai_client.models.generate_content(
                        model=model, contents=user_content, config=config
                    )
                    if model != settings.gemini_chat_model:
                        logger.info("Answered with fallback model %s", model)
                    return response.text or ""
                except genai_errors.APIError as exc:
                    if exc.code not in (429, 500, 503, 504):
                        raise
                    last_error = exc
                    logger.warning("Gemini %s returned %s (attempt %d)", model, exc.code, attempt + 1)
                    if attempt + 1 < settings.chat_retries_per_model:
                        time.sleep(3 * (attempt + 1))
        raise RuntimeError(
            f"All Gemini chat models are busy or out of quota ({', '.join(models)}). "
            "Try again in a few minutes, add models to GEMINI_FALLBACK_MODELS, or set LLM_PROVIDER=openai."
        ) from last_error

    def query(self, question: str, top_k: int | None = None) -> dict:
        context_chunks = self.retrieve(question, top_k)
        answer = self.generate_answer(question, context_chunks)
        sources = []
        for chunk in context_chunks:
            if not chunk["source"]:
                continue
            label = " ".join(p for p in [chunk["source"], _page_label(chunk["page_start"], chunk["page_end"])] if p)
            if label not in sources:
                sources.append(label)
        return {"answer": answer, "sources": sources, "context": context_chunks}


pipeline = RAGPipeline()
