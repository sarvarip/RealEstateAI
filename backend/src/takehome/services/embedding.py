from __future__ import annotations

import asyncio

import structlog
from openai import AsyncAzureOpenAI, AzureOpenAI

from takehome.config import settings

logger = structlog.get_logger()

# Sync client — used during document upload (background task, not in the
# event loop's hot path).
_sync_client: AzureOpenAI | None = None

# Async client — used by agent tools so embedding calls don't block the
# event loop (the root cause of "can't use other chats while agent runs").
_async_client: AsyncAzureOpenAI | None = None


def _get_sync_client() -> AzureOpenAI:
    global _sync_client
    if _sync_client is None:
        _sync_client = AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version="2024-06-01",
        )
    return _sync_client


def _get_async_client() -> AsyncAzureOpenAI:
    global _async_client
    if _async_client is None:
        _async_client = AsyncAzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version="2024-06-01",
        )
    return _async_client


def embed_texts(texts: list[str]) -> list[list[float] | None]:
    """Embed a batch of texts using Azure OpenAI (sync).

    Used during document upload where blocking is acceptable.
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

    client = _get_sync_client()
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


async def async_embed_query(text: str) -> list[float]:
    """Embed a single query string without blocking the event loop.

    Uses the async Azure OpenAI client so other requests (listing chats,
    loading messages) can be served while the embedding call is in flight.

    Raises RuntimeError if embeddings are not enabled.
    """
    if not settings.embeddings_enabled:
        raise RuntimeError(
            "embed_query called but Azure OpenAI keys are not configured. "
            "RAG mode requires embeddings."
        )
    client = _get_async_client()
    response = await client.embeddings.create(
        input=[text],
        model=settings.azure_embedding_deployment,
        dimensions=settings.embedding_dimensions,
    )
    return response.data[0].embedding


def embed_query(text: str) -> list[float]:
    """Embed a single query string (sync wrapper).

    If called from within a running event loop, delegates to the async version
    to avoid blocking. Otherwise falls back to the sync client.
    """
    if not settings.embeddings_enabled:
        raise RuntimeError(
            "embed_query called but Azure OpenAI keys are not configured. "
            "RAG mode requires embeddings."
        )
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        raise RuntimeError(
            "embed_query (sync) called inside a running event loop. "
            "Use async_embed_query instead."
        )

    results = embed_texts([text])
    assert results[0] is not None
    return results[0]
