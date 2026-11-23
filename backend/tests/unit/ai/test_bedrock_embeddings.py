"""Unit tests for Bedrock Titan embeddings (mocked AWS client)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from ai.providers.bedrock_embeddings import BedrockTitanEmbeddings
from config import Settings


def _settings() -> Settings:
    return Settings(
        aws_region="eu-north-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
        bedrock_embedding_model="amazon.titan-embed-text-v2:0",
        embedding_dimensions=1024,
        max_retries=1,
        embedding_batch_size=2,
        embedding_batch_pause_seconds=0.0,
    )


def test_embed_query_returns_configured_dimensions() -> None:
    vector = [0.01] * 1024
    body = MagicMock()
    body.read.return_value = json.dumps({"embedding": vector}).encode("utf-8")
    client = MagicMock()
    client.invoke_model.return_value = {"body": body}

    emb = BedrockTitanEmbeddings(_settings())
    with patch.object(emb, "_runtime_client", return_value=client):
        result = emb.embed_query("hello")

    assert len(result) == 1024
    assert emb.dimension == 1024
    kwargs = client.invoke_model.call_args.kwargs
    assert kwargs["modelId"] == "amazon.titan-embed-text-v2:0"
    payload = json.loads(kwargs["body"])
    assert payload["dimensions"] == 1024


def test_embed_documents_batches() -> None:
    vector = [0.02] * 1024
    body = MagicMock()
    body.read.return_value = json.dumps({"embedding": vector}).encode("utf-8")
    client = MagicMock()
    client.invoke_model.return_value = {"body": body}

    emb = BedrockTitanEmbeddings(_settings())
    with patch.object(emb, "_runtime_client", return_value=client):
        results = emb.embed_documents(["a", "b", "c"])

    assert len(results) == 3
    assert client.invoke_model.call_count == 3


def test_pipeline_embed_chunks_no_cache() -> None:
    from langchain_core.documents import Document

    from ai.ingestion.pipeline import embed_chunks

    settings = _settings()
    chunks = [
        Document(
            page_content="Revenue grew.",
            metadata={"chunk_id": "c1", "ticker": "AAPL"},
        ),
        Document(
            page_content="Margins improved.",
            metadata={"chunk_id": "c2", "ticker": "AAPL"},
        ),
    ]
    fake = MagicMock()
    fake.embed_batch.return_value = [[0.1] * 1024, [0.2] * 1024]
    fake.dimension = 1024

    records, summary = embed_chunks(chunks, fake, settings)
    assert len(records) == 2
    assert summary["successful_embeddings"] == 2
    assert "cache_hits" not in summary
    assert summary["embedding_model"] == "amazon.titan-embed-text-v2:0"
