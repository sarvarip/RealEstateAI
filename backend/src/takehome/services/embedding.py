from __future__ import annotations

import structlog
from openai import AzureOpenAI

from takehome.config import settings

logger = structlog.get_logger()

_client: AzureOpenAI | None = None


def _get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        _client = AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version="2024-06-01",
        )
    return _client


def embed_texts(texts: list[str]) -> list[list[float] | None]:
    """Embed a batch of texts using Azure OpenAI.

    Returns a list of embedding vectors, one per input text.
    If Azure OpenAI keys are not configured, returns [None, ...] so the app
    still works in full-context mode without embeddings.
    """
    if not texts:
        return []

    if not settings.embeddings_enabled:
        logger.warning(
            "Azure OpenAI keys not configured — skipping embedding",
            count=len(texts),
        )
        return [None] * len(texts)

    client = _get_client()
    all_embeddings: list[list[float] | None] = []
    batch_size = 16

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            input=batch,
            model=settings.azure_embedding_deployment,
            dimensions=settings.embedding_dimensions,
        )
        batch_embeddings: list[list[float] | None] = [
            item.embedding for item in response.data
        ]
        all_embeddings.extend(batch_embeddings)

    logger.info("Embedded texts", count=len(texts), dimensions=settings.embedding_dimensions)
    return all_embeddings


def embed_query(text: str) -> list[float]:
    """Embed a single query string.

    Raises RuntimeError if embeddings are not enabled (RAG should not be
    reachable without embeddings, but this guard prevents silent failures).
    """
    if not settings.embeddings_enabled:
        raise RuntimeError(
            "embed_query called but Azure OpenAI keys are not configured. "
            "RAG mode requires embeddings."
        )
    results = embed_texts([text])
    assert results[0] is not None
    return results[0]
