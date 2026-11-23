"""Centralized application settings for AI and ingestion."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.constants import (
    CHAT_MODEL,
    DEFAULT_PINECONE_INDEX_NAME,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    RERANKER_MODEL,
)

BACKEND_ROOT = Path(__file__).resolve().parents[2]  # .../backend
DATA_DIR = BACKEND_ROOT / "data"
MARKDOWN_DIR = DATA_DIR / "markdown"
MARKDOWN_MANIFEST = MARKDOWN_DIR / "manifest.json"
INGESTION_CACHE_DIR = DATA_DIR / "ingestion_cache"
INGESTION_REPORT_PATH = DATA_DIR / "ingestion_report.json"
BM25_CACHE_PATH = DATA_DIR / "ingestion_cache" / "bm25_encoder.json"


class Settings(BaseSettings):
    """Typed settings loaded from environment variables and optional .env file."""

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    database_url: str

    chat_temperature: float = Field(default=0.1, alias="CHAT_TEMPERATURE")
    chat_max_tokens: int = Field(default=4096, alias="CHAT_MAX_TOKENS")

    # Amazon AWS
    aws_region: str = Field(default="eu-north-1", alias="AWS_REGION")
    aws_access_key_id: str = Field(default="", alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(default="", alias="AWS_SECRET_ACCESS_KEY")

    # Amazon Bedrock
    bedrock_chat_model: str = Field(
        default=CHAT_MODEL, alias="AMAZON_BEDROCK_CHAT_MODEL"
    )
    bedrock_embedding_model: str = Field(
        default=EMBEDDING_MODEL, alias="AMAZON_BEDROCK_EMBEDDING_MODEL"
    )
    embedding_dimensions: int = Field(
        default=EMBEDDING_DIMENSIONS, alias="EMBEDDING_DIMENSIONS"
    )
    bedrock_ingest_mode: str = Field(default="batch", alias="BEDROCK_INGEST_MODE")
    aws_s3_embedding_bucket: str = Field(default="", alias="AWS_S3_EMBEDDING_BUCKET")
    aws_s3_embedding_input_prefix: str = Field(
        default="bedrock/embeddings/input/", alias="AWS_S3_EMBEDDING_INPUT_PREFIX"
    )
    aws_s3_embedding_output_prefix: str = Field(
        default="bedrock/embeddings/output/", alias="AWS_S3_EMBEDDING_OUTPUT_PREFIX"
    )
    bedrock_batch_role_arn: str = Field(default="", alias="BEDROCK_BATCH_ROLE_ARN")
    bedrock_batch_job_name_prefix: str = Field(
        default="hebbia-titan-embed", alias="BEDROCK_BATCH_JOB_NAME_PREFIX"
    )
    bedrock_batch_poll_seconds: float = Field(
        default=60.0, alias="BEDROCK_BATCH_POLL_SECONDS"
    )
    bedrock_batch_min_records: int = Field(
        default=100, alias="BEDROCK_BATCH_MIN_RECORDS"
    )

    # Pinecone Vector Store
    pinecone_api_key: str = Field(default="", alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field(
        default=DEFAULT_PINECONE_INDEX_NAME, alias="PINECONE_INDEX_NAME"
    )
    pinecone_cloud: str = Field(default="aws", alias="PINECONE_CLOUD")
    pinecone_region: str = Field(default="us-east-1", alias="PINECONE_REGION")

    # Chunking & Embedding
    chunk_size_tokens: int = Field(default=1000, alias="CHUNK_SIZE_TOKENS")
    chunk_overlap_tokens: int = Field(default=150, alias="CHUNK_OVERLAP_TOKENS")
    embedding_batch_size: int = Field(default=32, alias="EMBEDDING_BATCH_SIZE")
    embedding_batch_pause_seconds: float = Field(
        default=0.0, alias="EMBEDDING_BATCH_PAUSE_SECONDS"
    )
    max_consecutive_rate_limits: int = Field(
        default=5, alias="MAX_CONSECUTIVE_RATE_LIMITS"
    )
    rate_limit_max_backoff_seconds: float = Field(
        default=120.0, alias="RATE_LIMIT_MAX_BACKOFF_SECONDS"
    )
    pinecone_upsert_batch_size: int = Field(
        default=100, alias="PINECONE_UPSERT_BATCH_SIZE"
    )

    # RAG Retrieval & Reranking
    agent_max_iterations: int = Field(default=6, alias="AGENT_MAX_ITERATIONS")
    retrieval_top_k: int = Field(default=30, alias="RETRIEVAL_TOP_K")
    rerank_top_n: int = Field(default=7, alias="RERANK_TOP_N")
    context_token_budget: int = Field(default=20000, alias="CONTEXT_TOKEN_BUDGET")
    max_query_rewrites: int = Field(default=4, alias="MAX_QUERY_REWRITES")
    max_retrieval_calls: int = Field(default=8, alias="MAX_RETRIEVAL_CALLS")
    reranker_model: str = Field(default=RERANKER_MODEL, alias="RERANKER_MODEL")

    # Request Timeout & Retries
    request_timeout_seconds: float = Field(
        default=60.0, alias="REQUEST_TIMEOUT_SECONDS"
    )
    max_retries: int = Field(default=8, alias="MAX_RETRIES")
    min_document_chars: int = Field(default=200, alias="MIN_DOCUMENT_CHARS")

    # Data Paths
    markdown_dir: Path = Field(default=MARKDOWN_DIR)
    markdown_manifest: Path = Field(default=MARKDOWN_MANIFEST)
    ingestion_cache_dir: Path = Field(default=INGESTION_CACHE_DIR)
    ingestion_report_path: Path = Field(default=INGESTION_REPORT_PATH)
    bm25_cache_path: Path = Field(default=BM25_CACHE_PATH)

    def require_aws(self) -> None:
        if not self.aws_region.strip():
            raise ValueError("AWS_REGION is required for Amazon Bedrock.")
        if not self.bedrock_embedding_model.strip():
            raise ValueError("AMAZON_BEDROCK_EMBEDDING_MODEL is required.")

    def require_bedrock_batch(self) -> None:
        self.require_aws()
        if not self.aws_s3_embedding_bucket.strip():
            raise ValueError(
                "AWS_S3_EMBEDDING_BUCKET is required for Bedrock batch ingest."
            )
        if not self.bedrock_batch_role_arn.strip():
            raise ValueError(
                "BEDROCK_BATCH_ROLE_ARN is required for Bedrock batch ingest."
            )

    def require_pinecone(self) -> None:
        if not self.pinecone_api_key.strip():
            raise ValueError("PINECONE_API_KEY is required for Pinecone vector store.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
