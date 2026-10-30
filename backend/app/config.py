from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT_DIR / ".env"


class Settings(BaseSettings):
    llm_provider: str = "openai"  # "openai" or "gemini"

    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    gemini_api_key: str = ""
    gemini_chat_model: str = "gemini-3.6-flash"
    gemini_embedding_model: str = "gemini-embedding-001"
    # Comma-separated chat models to try when the main one is overloaded (503) or rate-limited (429)
    gemini_fallback_models: str = ""
    chat_retries_per_model: int = 2

    # "local" runs a small embedding model on this machine: free, no quota, no API key.
    # "openai" / "gemini" use the provider's embedding model. Changing it needs a re-ingest.
    embedding_provider: str = "local"
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Kept out of %TEMP%, which Windows may clean, so the ~70MB model downloads only once
    local_model_cache_dir: str = str(ROOT_DIR / "backend" / "model_cache")

    chroma_persist_dir: str = str(ROOT_DIR / "backend" / "chroma_data")
    chroma_collection_name: str = "documents"

    chunk_size: int = 1200
    chunk_overlap: int = 150
    retrieval_top_k: int = 8
    # A matched table row pulls in its whole table (up to this size) for cross-row questions
    max_expanded_tables: int = 2
    max_table_chars: int = 12000

    # Gemini free tier rate-limits embeddings per minute; keep batches small and retry on 429
    embed_batch_size: int = 50
    embed_max_retries: int = 6
    # Longer suggested waits mean a daily quota hit, not a per-minute one: fail instead of sleeping
    embed_max_retry_wait: int = 120

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH), env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
