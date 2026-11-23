"""Amazon Bedrock Titan Embed Text V2 client (LangChain Embeddings compatible)."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from langchain_core.embeddings import Embeddings

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


class BedrockTitanEmbeddings(Embeddings):
    """Generate embeddings with amazon.titan-embed-text-v2:0 via Bedrock Runtime."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.settings.require_aws()
        self.model = self.settings.bedrock_embedding_model
        self.dimensions = self.settings.embedding_dimensions
        self.max_retries = self.settings.max_retries
        self._dimension: int | None = None
        self._client: Any | None = None

    @property
    def dimension(self) -> int | None:
        return self._dimension

    def _runtime_client(self) -> Any:
        if self._client is not None:
            return self._client
        import boto3

        kwargs: dict[str, Any] = {"region_name": self.settings.aws_region}
        if (
            self.settings.aws_access_key_id.strip()
            and self.settings.aws_secret_access_key.strip()
        ):
            kwargs["aws_access_key_id"] = self.settings.aws_access_key_id
            kwargs["aws_secret_access_key"] = self.settings.aws_secret_access_key
        self._client = boto3.client("bedrock-runtime", **kwargs)
        return self._client

    def _embed_one(self, text: str) -> list[float]:
        payload = {
            "inputText": text,
            "dimensions": self.dimensions,
            "normalize": True,
        }
        last_error: Exception | None = None
        client = self._runtime_client()
        for attempt in range(self.max_retries + 1):
            try:
                response = client.invoke_model(
                    modelId=self.model,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(payload),
                )
                body = response.get("body")
                raw = body.read() if hasattr(body, "read") else body
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                parsed = json.loads(raw)
                vector = parsed.get("embedding")
                if not isinstance(vector, list) or not vector:
                    raise ValueError(f"Malformed Titan embedding response: {parsed!r}")
                floats = [float(value) for value in vector]
                if len(floats) != self.dimensions:
                    raise ValueError(
                        f"Expected {self.dimensions} dimensions, received {len(floats)}"
                    )
                if self._dimension is None:
                    self._dimension = len(floats)
                elif self._dimension != len(floats):
                    raise ValueError(
                        f"Embedding dimension mismatch: expected {self._dimension}, "
                        f"received {len(floats)}"
                    )
                return floats
            except Exception as exc:  # noqa: BLE001 — retry transient Bedrock failures
                last_error = exc
                if attempt >= self.max_retries:
                    break
                delay = min(2**attempt, self.settings.rate_limit_max_backoff_seconds)
                message = str(exc)
                if "Throttling" in type(exc).__name__ or "Throttling" in message:
                    delay = max(delay, 5.0)
                logger.warning(
                    "Bedrock embedding failed (attempt %s/%s): %s; retrying in %.1fs",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                    delay,
                )
                time.sleep(delay)
        raise RuntimeError(
            f"Bedrock Titan embeddings failed after retries: {last_error}"
        )

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed texts one-by-one (Titan V2 invoke_model is single-input)."""
        return [self._embed_one(text) for text in texts]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        batch_size = max(1, self.settings.embedding_batch_size)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors.extend(self.embed_batch(batch))
            if (
                start + batch_size < len(texts)
                and self.settings.embedding_batch_pause_seconds > 0
            ):
                time.sleep(self.settings.embedding_batch_pause_seconds)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)

    def probe_dimension(self) -> int:
        if self._dimension is not None:
            return self._dimension
        self.embed_query("dimension probe")
        if self._dimension is None:
            raise RuntimeError("Failed to determine embedding dimension")
        return self._dimension
