"""Unit tests for Bedrock batch JSONL helpers (no live AWS)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from config import Settings
from langchain_core.documents import Document

from ai.ingestion.bedrock_batch import (
    build_jsonl_records,
    embed_chunks_via_batch,
    parse_batch_output_line,
)


def _chunk(chunk_id: str, text: str = "hello") -> Document:
    return Document(
        page_content=text,
        metadata={"chunk_id": chunk_id, "ticker": "AAPL", "source_document": "x.md"},
    )


def test_build_jsonl_records_uses_chunk_id_as_record_id() -> None:
    body, by_id = build_jsonl_records(
        [_chunk("aapl-1"), _chunk("aapl-2", "world")],
        dimensions=1024,
    )
    lines = [json.loads(line) for line in body.strip().splitlines()]
    assert len(lines) == 2
    assert lines[0]["recordId"] == "aapl-1"
    assert lines[0]["modelInput"] == {
        "inputText": "hello",
        "dimensions": 1024,
        "normalize": True,
    }
    assert set(by_id) == {"aapl-1", "aapl-2"}


def test_parse_batch_output_success_and_error() -> None:
    ok_line = json.dumps(
        {
            "recordId": "aapl-1",
            "modelOutput": {"embedding": [0.1] * 1024, "inputTextTokenCount": 1},
        }
    )
    record_id, vector, error = parse_batch_output_line(
        ok_line, expected_dimensions=1024
    )
    assert record_id == "aapl-1"
    assert vector is not None and len(vector) == 1024
    assert error is None

    err_line = json.dumps(
        {
            "recordId": "aapl-2",
            "error": {"errorCode": 400, "errorMessage": "bad request"},
        }
    )
    record_id, vector, error = parse_batch_output_line(
        err_line, expected_dimensions=1024
    )
    assert record_id == "aapl-2"
    assert vector is None
    assert "bad request" in (error or "")


def test_embed_chunks_via_batch_mocked_aws() -> None:
    settings = Settings(
        aws_region="eu-north-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
        bedrock_embedding_model="amazon.titan-embed-text-v2:0",
        embedding_dimensions=1024,
        aws_s3_embedding_bucket="test-bucket",
        bedrock_batch_role_arn="arn:aws:iam::123456789012:role/batch",
        bedrock_batch_poll_seconds=0.01,
        bedrock_batch_min_records=2,
    )
    chunks = [_chunk("c1", "one"), _chunk("c2", "two")]
    out_line_1 = json.dumps(
        {"recordId": "c1", "modelOutput": {"embedding": [0.1] * 1024}}
    )
    out_line_2 = json.dumps(
        {"recordId": "c2", "modelOutput": {"embedding": [0.2] * 1024}}
    )

    s3 = MagicMock()
    bedrock = MagicMock()
    bedrock.create_model_invocation_job.return_value = {
        "jobArn": "arn:aws:bedrock:eu-north-1:123:model-invocation-job/abc"
    }
    bedrock.get_model_invocation_job.return_value = {"status": "Completed"}
    s3.get_paginator.return_value.paginate.return_value = [
        {
            "Contents": [
                {"Key": "bedrock/embeddings/output/run/input.jsonl.out"},
                {"Key": "bedrock/embeddings/output/run/manifest.json.out"},
            ]
        }
    ]
    body = MagicMock()
    body.read.return_value = f"{out_line_1}\n{out_line_2}\n".encode()
    s3.get_object.return_value = {"Body": body}

    def _client(service_name: str, **_kwargs: object) -> MagicMock:
        if service_name == "s3":
            return s3
        if service_name == "bedrock":
            return bedrock
        raise AssertionError(service_name)

    with patch("boto3.client", side_effect=_client):
        records, summary = embed_chunks_via_batch(chunks, settings)

    assert len(records) == 2
    assert summary["ingest_mode"] == "batch"
    assert summary["successful_embeddings"] == 2
    s3.put_object.assert_called_once()
    bedrock.create_model_invocation_job.assert_called_once()


def test_embed_chunks_via_batch_enforces_min_records() -> None:
    settings = Settings(
        aws_region="eu-north-1",
        aws_s3_embedding_bucket="test-bucket",
        bedrock_batch_role_arn="arn:aws:iam::123456789012:role/batch",
        bedrock_batch_min_records=100,
    )
    with pytest.raises(ValueError, match="at least 100"):
        embed_chunks_via_batch([_chunk("only-one")], settings)
