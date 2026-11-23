"""Amazon Bedrock asynchronous batch inference for Titan corpus embeddings."""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import boto3
from langchain_core.documents import Document

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = frozenset(
    {"Completed", "Failed", "Stopped", "Expired", "PartiallyCompleted"}
)


def _aws_client_kwargs(settings: Settings) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"region_name": settings.aws_region}
    if settings.aws_access_key_id.strip() and settings.aws_secret_access_key.strip():
        kwargs["aws_access_key_id"] = settings.aws_access_key_id
        kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
    return kwargs


def _normalize_prefix(prefix: str) -> str:
    cleaned = prefix.strip().lstrip("/")
    if cleaned and not cleaned.endswith("/"):
        cleaned = f"{cleaned}/"
    return cleaned


def _safe_job_name(prefix: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    raw = f"{prefix}-{stamp}-{suffix}"
    cleaned = re.sub(r"[^A-Za-z0-9-]", "-", raw).strip("-")
    return cleaned[:63] or f"hebbia-embed-{suffix}"


def build_jsonl_records(
    chunks: list[Document],
    *,
    dimensions: int,
) -> tuple[str, dict[str, Document]]:
    """Build JSONL body and chunk lookup keyed by recordId (chunk_id)."""
    lines: list[str] = []
    by_id: dict[str, Document] = {}
    for chunk in chunks:
        chunk_id = str(chunk.metadata.get("chunk_id") or "").strip()
        if not chunk_id:
            raise ValueError(
                "Chunk missing chunk_id metadata; cannot build batch recordId"
            )
        if chunk_id in by_id:
            raise ValueError(f"Duplicate chunk_id in batch input: {chunk_id}")
        record = {
            "recordId": chunk_id,
            "modelInput": {
                "inputText": chunk.page_content,
                "dimensions": dimensions,
                "normalize": True,
            },
        }
        lines.append(json.dumps(record, ensure_ascii=False))
        by_id[chunk_id] = chunk
    return "\n".join(lines) + ("\n" if lines else ""), by_id


def parse_batch_output_line(
    line: str,
    *,
    expected_dimensions: int,
) -> tuple[str, list[float] | None, str | None]:
    """Parse one output JSONL line → (recordId, embedding|None, error|None)."""
    payload = json.loads(line)
    record_id = str(payload.get("recordId") or "").strip()
    if not record_id:
        raise ValueError(f"Batch output line missing recordId: {line[:200]}")

    if "error" in payload and payload["error"]:
        err = payload["error"]
        if isinstance(err, dict):
            message = str(err.get("errorMessage") or err.get("message") or err)
        else:
            message = str(err)
        return record_id, None, message

    model_output = payload.get("modelOutput") or {}
    if not isinstance(model_output, dict):
        return record_id, None, f"Malformed modelOutput: {model_output!r}"

    vector = model_output.get("embedding")
    if not isinstance(vector, list) or not vector:
        return record_id, None, f"Missing embedding in modelOutput: {model_output!r}"

    floats = [float(value) for value in vector]
    if len(floats) != expected_dimensions:
        return (
            record_id,
            None,
            f"Expected {expected_dimensions} dimensions, received {len(floats)}",
        )
    return record_id, floats, None


def _list_output_jsonl_keys(s3: Any, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents") or []:
            key = str(item.get("Key") or "")
            name = key.rsplit("/", 1)[-1]
            if not key or name == "manifest.json.out":
                continue
            if name.endswith(".jsonl.out") or name.endswith(".jsonl"):
                keys.append(key)
    # Prefer explicit .jsonl.out artifacts when present.
    out_keys = [key for key in keys if key.endswith(".jsonl.out")]
    return out_keys or keys


def embed_chunks_via_batch(
    chunks: list[Document],
    settings: Settings | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Embed corpus chunks with Bedrock CreateModelInvocationJob + S3 JSONL."""

    settings = settings or get_settings()
    settings.require_bedrock_batch()

    if not chunks:
        return [], {
            "total_chunks": 0,
            "successful_embeddings": 0,
            "failed_embeddings": 0,
            "embedded_this_run": 0,
            "pending_remaining": 0,
            "stopped_early": False,
            "vector_dimensions": settings.embedding_dimensions,
            "embedding_model": settings.bedrock_embedding_model,
            "ingest_mode": "batch",
            "failures": [],
        }

    min_records = max(1, settings.bedrock_batch_min_records)
    if len(chunks) < min_records:
        raise ValueError(
            f"Bedrock batch jobs require at least {min_records} records; "
            f"got {len(chunks)}. Add more chunks or lower BEDROCK_BATCH_MIN_RECORDS "
            "only if your account quota allows it."
        )

    logger.info("Starting Bedrock batch embedding...")
    logger.info("Model: %s", settings.bedrock_embedding_model)
    logger.info("Dimensions: %s", settings.embedding_dimensions)
    logger.info("Mode: asynchronous batch inference")
    logger.info("Total chunks: %s", len(chunks))

    jsonl_body, by_id = build_jsonl_records(
        chunks,
        dimensions=settings.embedding_dimensions,
    )
    logger.info("Wrote JSONL records: %s", len(by_id))

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    input_prefix = (
        _normalize_prefix(settings.aws_s3_embedding_input_prefix) + f"{run_id}/"
    )
    output_prefix = (
        _normalize_prefix(settings.aws_s3_embedding_output_prefix) + f"{run_id}/"
    )
    input_key = f"{input_prefix}input.jsonl"
    bucket = settings.aws_s3_embedding_bucket.strip()
    input_uri = f"s3://{bucket}/{input_key}"
    output_uri = f"s3://{bucket}/{output_prefix}"

    client_kwargs = _aws_client_kwargs(settings)
    s3 = boto3.client("s3", **client_kwargs)
    bedrock = boto3.client("bedrock", **client_kwargs)

    s3.put_object(
        Bucket=bucket,
        Key=input_key,
        Body=jsonl_body.encode("utf-8"),
        ContentType="application/jsonl",
    )
    logger.info("Uploaded input to %s", input_uri)

    job_name = _safe_job_name(settings.bedrock_batch_job_name_prefix)
    try:
        create_response = bedrock.create_model_invocation_job(
            roleArn=settings.bedrock_batch_role_arn.strip(),
            modelId=settings.bedrock_embedding_model.strip(),
            jobName=job_name,
            inputDataConfig={
                "s3InputDataConfig": {
                    "s3Uri": input_uri,
                    "s3InputFormat": "JSONL",
                }
            },
            outputDataConfig={
                "s3OutputDataConfig": {
                    "s3Uri": output_uri,
                }
            },
        )
    except Exception as exc:  # noqa: BLE001 — surface AWS account gates clearly
        message = str(exc)
        if "not authorized to perform this action" in message.lower():
            raise RuntimeError(
                "Bedrock batch inference is blocked at the AWS account level "
                "(not an IAM/S3/.env misconfiguration — input JSONL already uploaded). "
                "InvokeModel access does not include CreateModelInvocationJob. "
                "Open an AWS Support case requesting Bedrock batch inference "
                f"for model {settings.bedrock_embedding_model!r} in region "
                f"{settings.aws_region!r}, then re-run ingestion. "
                "Temporary workaround: set BEDROCK_INGEST_MODE=ondemand "
                "(may hit on-demand throttling). Original error: "
                f"{message}"
            ) from exc
        raise
    job_arn = str(create_response.get("jobArn") or "")
    if not job_arn:
        raise RuntimeError(
            f"create_model_invocation_job returned no jobArn: {create_response!r}"
        )
    logger.info("Submitted batch job: %s", job_arn)

    status = "Submitted"
    message = ""
    while status not in _TERMINAL_STATUSES:
        time.sleep(max(5.0, settings.bedrock_batch_poll_seconds))
        detail = bedrock.get_model_invocation_job(jobIdentifier=job_arn)
        status = str(detail.get("status") or "Unknown")
        message = str(detail.get("message") or "")
        logger.info("Job status: %s%s", status, f" ({message})" if message else "")

    if status not in {"Completed", "PartiallyCompleted"}:
        raise RuntimeError(
            f"Bedrock batch job ended with status={status} message={message!r} "
            f"arn={job_arn}"
        )

    output_keys = _list_output_jsonl_keys(s3, bucket, output_prefix)
    if not output_keys:
        raise RuntimeError(
            f"No batch output JSONL found under s3://{bucket}/{output_prefix}"
        )

    embeddings_by_id: dict[str, list[float]] = {}
    failed: list[dict[str, Any]] = []
    for key in output_keys:
        obj = s3.get_object(Bucket=bucket, Key=key)
        raw = obj["Body"].read().decode("utf-8")
        for line in raw.splitlines():
            if not line.strip():
                continue
            record_id, vector, error = parse_batch_output_line(
                line,
                expected_dimensions=settings.embedding_dimensions,
            )
            if error or vector is None:
                chunk = by_id.get(record_id)
                failed.append(
                    {
                        "chunk_id": record_id,
                        "ticker": (chunk.metadata.get("ticker") if chunk else None),
                        "source_document": (
                            chunk.metadata.get("source_document") if chunk else None
                        ),
                        "error": error or "missing embedding",
                    }
                )
                continue
            embeddings_by_id[record_id] = vector

    records: list[dict[str, Any]] = []
    for chunk_id, chunk in by_id.items():
        vector = embeddings_by_id.get(chunk_id)
        if vector is None:
            if not any(item["chunk_id"] == chunk_id for item in failed):
                failed.append(
                    {
                        "chunk_id": chunk_id,
                        "ticker": chunk.metadata.get("ticker"),
                        "source_document": chunk.metadata.get("source_document"),
                        "error": "No embedding returned for recordId",
                    }
                )
            continue
        records.append(
            {
                "id": chunk_id,
                "values": vector,
                "metadata": chunk.metadata,
                "text": chunk.page_content,
                "ticker": chunk.metadata.get("ticker"),
            }
        )

    summary: dict[str, Any] = {
        "total_chunks": len(chunks),
        "successful_embeddings": len(records),
        "failed_embeddings": len(failed),
        "embedded_this_run": len(records),
        "pending_remaining": max(0, len(chunks) - len(records)),
        "stopped_early": False,
        "vector_dimensions": settings.embedding_dimensions,
        "embedding_model": settings.bedrock_embedding_model,
        "ingest_mode": "batch",
        "batch_job_arn": job_arn,
        "batch_job_status": status,
        "s3_input_uri": input_uri,
        "s3_output_uri": output_uri,
        "failures": failed,
    }
    logger.info(
        "Parsed embeddings: %s / %s (failed=%s dim=%s)",
        summary["successful_embeddings"],
        summary["total_chunks"],
        summary["failed_embeddings"],
        summary["vector_dimensions"],
    )
    return records, summary
