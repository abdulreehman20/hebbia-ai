"""Shared constants for the financial research backend."""

from __future__ import annotations

COMPANY_NAMESPACES: dict[str, str] = {
    "AAPL": "apple",
    "MSFT": "microsoft",
    "GOOGL": "google",
    "AMZN": "amazon",
    "NVDA": "nvidia",
    "META": "meta",
    "WMT": "walmart",
    "ORCL": "oracle",
}

SUPPORTED_TICKERS: frozenset[str] = frozenset(COMPANY_NAMESPACES)

COMPANY_NAMES: dict[str, str] = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com, Inc.",
    "NVDA": "NVIDIA Corporation",
    "META": "Meta Platforms, Inc.",
    "WMT": "Walmart Inc.",
    "ORCL": "Oracle Corporation",
}

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
# LEGACY - Old Code / OpenRouter
# CHAT_MODEL = "nvidia/nemotron-3.5-super-vl:free"
CHAT_MODEL = "amazon.nova-lite-v1:0"
EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
EMBEDDING_DIMENSIONS = 1024

INGESTION_VERSION = "phase2-v1"
DEFAULT_PINECONE_INDEX_NAME = "hebbia-ai"
