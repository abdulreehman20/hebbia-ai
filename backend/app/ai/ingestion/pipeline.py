"""End-to-end Phase 2 ingestion orchestration."""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from typing import Any

from langchain_core.documents import Document

from app.ai.ingestion.bedrock_batch import embed_chunks_via_batch
from app.ai.ingestion.chunker import chunk_documents
from app.ai.ingestion.cleaner import clean_documents
from app.ai.ingestion.loader import load_documents
from app.ai.ingestion.pinecone_store import ensure_index, namespace_stats, upsert_chunks
from app.core.config import Settings, get_settings
from app.core.constants import COMPANY_NAMESPACES, INGESTION_VERSION
from app.providers.bedrock_embeddings import BedrockTitanEmbeddings

logger = logging.getLogger(__name__)


def embed_chunks(
    chunks: list[Document],
    embeddings: BedrockTitanEmbeddings,
    settings: Settings,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Embed all chunks with Bedrock Titan (no local embedding cache)."""
    logger.info("Starting embedding generation...")
    logger.info("Model: %s", settings.bedrock_embedding_model)
    logger.info("Dimensions: %s", settings.embedding_dimensions)
    logger.info(
        "Total chunks: %s (batch_size=%s pause=%.1fs)",
        len(chunks),
        settings.embedding_batch_size,
        settings.embedding_batch_pause_seconds,
    )

    failed: list[dict[str, Any]] = []
    embedded_now = 0
    consecutive_failures = 0
    stopped_early = False
    prepared: list[dict[str, Any] | None] = [None] * len(chunks)

    batch_size = max(1, settings.embedding_batch_size)
    try:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [chunk.page_content for chunk in batch]
            try:
                vectors = embeddings.embed_batch(texts)
                for offset, vector in enumerate(vectors):
                    chunk = batch[offset]
                    chunk_id = str(chunk.metadata.get("chunk_id"))
                    prepared[start + offset] = {
                        "id": chunk_id,
                        "values": vector,
                        "metadata": chunk.metadata,
                        "text": chunk.page_content,
                        "ticker": chunk.metadata.get("ticker"),
                    }
                    embedded_now += 1
                consecutive_failures = 0
                logger.info("Generated embeddings: %s / %s", embedded_now, len(chunks))
                if (
                    start + batch_size < len(chunks)
                    and settings.embedding_batch_pause_seconds > 0
                ):
                    time.sleep(settings.embedding_batch_pause_seconds)
            except Exception as exc:
                consecutive_failures += 1
                logger.exception("Embedding batch failed at offset %s", start)
                for chunk in batch:
                    failed.append(
                        {
                            "chunk_id": chunk.metadata.get("chunk_id"),
                            "ticker": chunk.metadata.get("ticker"),
                            "source_document": chunk.metadata.get("source_document"),
                            "error": str(exc),
                        }
                    )
                if consecutive_failures >= settings.max_consecutive_rate_limits:
                    remaining = len(chunks) - start - len(batch)
                    stopped_early = True
                    logger.warning(
                        "Stopping embedding early after %s consecutive batch failures. "
                        "Re-run ingestion to continue (%s chunks still pending).",
                        consecutive_failures,
                        remaining,
                    )
                    break
                cool_down = min(
                    5.0 * consecutive_failures,
                    settings.rate_limit_max_backoff_seconds,
                )
                logger.info("Cooling down %.1fs before next batch...", cool_down)
                time.sleep(cool_down)
    except KeyboardInterrupt:
        logger.warning(
            "Interrupted during embedding — partial results will be upserted if any."
        )
        raise

    records = [row for row in prepared if row is not None]
    summary = {
        "total_chunks": len(chunks),
        "successful_embeddings": len(records),
        "failed_embeddings": len(failed),
        "embedded_this_run": embedded_now,
        "pending_remaining": max(0, len(chunks) - len(records)),
        "stopped_early": stopped_early,
        "vector_dimensions": embeddings.dimension or settings.embedding_dimensions,
        "embedding_model": settings.bedrock_embedding_model,
        "failures": failed,
    }
    logger.info(
        "Embedding generation completed. success=%s failed=%s dim=%s stopped_early=%s",
        summary["successful_embeddings"],
        summary["failed_embeddings"],
        summary["vector_dimensions"],
        stopped_early,
    )
    return records, summary


def run_ingestion(settings: Settings | None = None) -> dict[str, Any]:
    """Execute load → clean → chunk → embed → upsert and return a report."""
    settings = settings or get_settings()
    settings.require_aws()
    settings.require_pinecone()

    logger.info("Starting SEC document ingestion...")
    documents, load_failures = load_documents(settings)
    cleaned, rejected, audit = clean_documents(documents, settings)
    chunks, chunk_summary = chunk_documents(cleaned, settings)

    ingest_mode = (settings.bedrock_ingest_mode or "batch").strip().lower()
    dimension = settings.embedding_dimensions

    if ingest_mode == "batch":
        records, embed_summary = embed_chunks_via_batch(chunks, settings)
        dimension = int(
            embed_summary.get("vector_dimensions") or settings.embedding_dimensions
        )
    elif ingest_mode in {"ondemand", "on-demand", "invoke"}:
        embeddings = BedrockTitanEmbeddings(settings)
        dimension = embeddings.probe_dimension()
        logger.info("Measured embedding dimension: %s", dimension)
        records, embed_summary = embed_chunks(chunks, embeddings, settings)
        embed_summary = {**embed_summary, "ingest_mode": "ondemand"}
    else:
        raise ValueError(
            f"Unsupported BEDROCK_INGEST_MODE={settings.bedrock_ingest_mode!r}. "
            "Use 'batch' (default) or 'ondemand'."
        )

    if embed_summary.get("stopped_early"):
        logger.warning(
            "Partial embedding run. Upserting %s available vectors; "
            "re-run ingestion to continue.",
            len(records),
        )

    index = ensure_index(dimension, settings)
    upsert_summary = upsert_chunks(index, records, settings)
    stats = namespace_stats(index)

    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "ingestion_version": INGESTION_VERSION,
        "embedding_model": settings.bedrock_embedding_model,
        "vector_dimensions": dimension,
        "bedrock_ingest_mode": embed_summary.get("ingest_mode", ingest_mode),
        "pinecone_index": settings.pinecone_index_name,
        "company_namespaces": COMPANY_NAMESPACES,
        "source_markdown_documents": len(documents) + len(load_failures),
        "loaded_documents": len(documents),
        "load_failures": load_failures,
        "rejected_during_validation": rejected,
        "quality_audit": audit,
        "chunking": chunk_summary,
        "embeddings": embed_summary,
        "upserts": upsert_summary,
        "pinecone_namespace_counts": stats,
        "expected_namespaces": sorted(COMPANY_NAMESPACES.values()),
    }

    settings.ingestion_report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = settings.ingestion_report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(settings.ingestion_report_path)
    logger.info("Wrote ingestion report to %s", settings.ingestion_report_path)
    return report
