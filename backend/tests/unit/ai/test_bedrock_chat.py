"""Unit tests for Bedrock chat factory (mocked LangChain AWS client)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from config import Settings


def _settings() -> Settings:
    return Settings(
        aws_region="eu-north-1",
        aws_access_key_id="test-key",
        aws_secret_access_key="test-secret",
        bedrock_chat_model="amazon.nova-lite-v1:0",
        bedrock_embedding_model="amazon.titan-embed-text-v2:0",
        chat_temperature=0.1,
        chat_max_tokens=2048,
    )


def test_get_chat_model_uses_bedrock_converse() -> None:
    from ai.providers.bedrock_chat import get_chat_model

    fake = MagicMock(name="ChatBedrockConverse")
    with patch(
        "ai.providers.bedrock_chat.ChatBedrockConverse", return_value=fake
    ) as ctor:
        model = get_chat_model(_settings(), temperature=0.0, streaming=True)

    assert model is fake
    kwargs = ctor.call_args.kwargs
    assert kwargs["model"] == "amazon.nova-lite-v1:0"
    assert kwargs["region_name"] == "eu-north-1"
    assert kwargs["temperature"] == 0.0
    assert kwargs["max_tokens"] == 2048
    assert kwargs["streaming"] is True
    assert kwargs["aws_access_key_id"] == "test-key"
    assert kwargs["aws_secret_access_key"] == "test-secret"
