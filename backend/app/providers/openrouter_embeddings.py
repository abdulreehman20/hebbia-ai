# LEGACY - Old Code / OpenRouter

# """Deprecated OpenRouter embeddings path.
#
# Embeddings now use Amazon Bedrock Titan via bedrock_embeddings.BedrockTitanEmbeddings
# """
#
# from __future__ import annotations
#
#
# class OpenRouterEmbeddings:  # pragma: no cover - deprecated shim
#     def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
#         raise RuntimeError(
#             "OpenRouter embeddings are retired. Use "
#             "ai.providers.bedrock_embeddings.BedrockTitanEmbeddings instead."
#         )
