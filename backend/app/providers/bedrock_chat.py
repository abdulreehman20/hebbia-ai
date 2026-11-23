"""Amazon Bedrock chat model factory for planning, tools, and generation."""

from __future__ import annotations

from typing import Any

from langchain_aws import ChatBedrockConverse

from app.core.config import Settings, get_settings


def get_chat_model(
    settings: Settings | None = None,
    *,
    temperature: float | None = None,
    streaming: bool = False,
) -> ChatBedrockConverse:
    """Return a Bedrock Converse chat model for agentic RAG."""
    settings = settings or get_settings()
    settings.require_aws()
    if not settings.bedrock_chat_model.strip():
        raise ValueError(
            "AMAZON_BEDROCK_CHAT_MODEL is required for Bedrock chat model."
        )

    kwargs: dict[str, Any] = {
        "model": settings.bedrock_chat_model,
        "region_name": settings.aws_region,
        "temperature": settings.chat_temperature
        if temperature is None
        else temperature,
        "max_tokens": settings.chat_max_tokens,
        "streaming": streaming,
    }
    if settings.aws_access_key_id.strip() and settings.aws_secret_access_key.strip():
        kwargs["aws_access_key_id"] = settings.aws_access_key_id
        kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
    return ChatBedrockConverse(**kwargs)
