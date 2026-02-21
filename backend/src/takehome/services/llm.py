from __future__ import annotations

import re
from collections.abc import AsyncIterator
from dataclasses import dataclass

import structlog
from pydantic_ai import Agent
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from takehome.config import settings  # noqa: F401 — triggers ANTHROPIC_API_KEY export
from takehome.db.models import Document, DocumentChunk
from takehome.services.chunking import estimate_tokens
from takehome.services.embedding import embed_query

logger = structlog.get_logger()

agent = Agent(
    "anthropic:claude-haiku-4-5-20251001",
    system_prompt=(
        "You are a helpful legal document assistant for commercial real estate lawyers. "
        "You help lawyers review and understand documents during due diligence.\n\n"
        "IMPORTANT INSTRUCTIONS:\n"
        "- Answer questions based ONLY on the document content provided.\n"
        "- When referencing specific content, ALWAYS cite the source using this exact format: "
        "[Doc: <filename> | Page <number>]. For example: [Doc: lease.pdf | Page 5].\n"
        "- Every factual claim must have at least one citation.\n"
        "- If the answer is not in the provided documents, say so clearly. "
        "Do not fabricate information.\n"
        "- Be concise and precise. Lawyers value accuracy over verbosity.\n"
        "- When multiple documents are provided, cross-reference information between them "
        "and note any discrepancies."
    ),
)


@dataclass
class RetrievalContext:
    """The document context assembled for the LLM prompt."""

    text: str
    mode: str  # "full" or "rag"
    chunk_count: int
    doc_count: int


async def build_document_context(
    session: AsyncSession,
    conversation_id: str,
    user_message: str,
    documents: list[Document],
    rag_threshold_override: int | None = None,
) -> RetrievalContext:
    """Build the document context for the LLM, using full-context or RAG as appropriate."""
    if not documents:
        return RetrievalContext(
            text="No documents have been uploaded yet. If the user asks about a document, "
            "let them know they need to upload one first.",
            mode="none",
            chunk_count=0,
            doc_count=0,
        )

    threshold = rag_threshold_override if rag_threshold_override is not None else settings.rag_token_threshold
    total_text = "\n\n".join(d.extracted_text or "" for d in documents)
    total_tokens = estimate_tokens(total_text)

    logger.info("Token estimate", total_tokens=total_tokens, threshold=threshold)

    if not settings.embeddings_enabled:
        if total_tokens >= threshold:
            logger.warning(
                "Embeddings disabled but token count exceeds threshold — "
                "using full-context anyway",
                total_tokens=total_tokens,
                threshold=threshold,
            )
        return _build_full_context(documents)

    if total_tokens < threshold:
        return _build_full_context(documents)

    return await _build_rag_context(session, conversation_id, user_message)


def _build_full_context(documents: list[Document]) -> RetrievalContext:
    """Full-context mode: include all document text with clear labels."""
    parts: list[str] = []
    total_chunks = 0

    for doc in documents:
        if doc.extracted_text:
            parts.append(
                f"<document filename=\"{doc.filename}\">\n"
                f"{doc.extracted_text}\n"
                f"</document>"
            )
            total_chunks += 1

    logger.info(
        "Using full-context mode",
        doc_count=len(documents),
        total_chars=sum(len(d.extracted_text or "") for d in documents),
    )

    return RetrievalContext(
        text="\n\n".join(parts),
        mode="full",
        chunk_count=total_chunks,
        doc_count=len(documents),
    )


async def _build_rag_context(
    session: AsyncSession,
    conversation_id: str,
    user_message: str,
) -> RetrievalContext:
    """RAG mode: embed query, retrieve top-K relevant chunks via pgvector."""
    query_embedding = embed_query(user_message)

    stmt = text("""
        SELECT dc.id, dc.content, dc.page_number, dc.chunk_index,
               d.filename, d.id as document_id,
               dc.embedding <=> :query_embedding AS distance
        FROM document_chunks dc
        JOIN documents d ON dc.document_id = d.id
        WHERE d.conversation_id = :conversation_id
          AND dc.embedding IS NOT NULL
        ORDER BY dc.embedding <=> :query_embedding
        LIMIT :top_k
    """)

    result = await session.execute(
        stmt,
        {
            "query_embedding": str(query_embedding),
            "conversation_id": conversation_id,
            "top_k": settings.rag_top_k,
        },
    )
    rows = result.fetchall()

    if not rows:
        return RetrievalContext(
            text="Documents were uploaded but no searchable content was found.",
            mode="rag",
            chunk_count=0,
            doc_count=0,
        )

    parts: list[str] = []
    doc_names: set[str] = set()
    for row in rows:
        filename = row.filename
        page = row.page_number
        content = row.content
        doc_names.add(filename)
        parts.append(
            f"<chunk filename=\"{filename}\" page=\"{page}\">\n"
            f"{content}\n"
            f"</chunk>"
        )

    logger.info(
        "Using RAG mode",
        chunks_retrieved=len(rows),
        doc_count=len(doc_names),
        top_distance=round(rows[0].distance, 4) if rows else None,
    )

    return RetrievalContext(
        text="\n\n".join(parts),
        mode="rag",
        chunk_count=len(rows),
        doc_count=len(doc_names),
    )


async def generate_title(user_message: str) -> str:
    """Generate a 3-5 word conversation title from the first user message."""
    result = await agent.run(
        f"Generate a concise 3-5 word title for a conversation that starts with: '{user_message}'. "
        "Return only the title, nothing else."
    )
    title = str(result.output).strip().strip('"').strip("'")
    if len(title) > 100:
        title = title[:97] + "..."
    return title


async def chat_with_documents(
    user_message: str,
    context: RetrievalContext,
    conversation_history: list[dict[str, str]],
) -> AsyncIterator[str]:
    """Stream a response to the user's message with document context.

    Uses the pre-built RetrievalContext which may be full-context or RAG-retrieved.
    """
    prompt_parts: list[str] = []

    prompt_parts.append(
        f"The following is the relevant content from {context.doc_count} document(s) "
        f"({context.mode} mode, {context.chunk_count} sections):\n\n"
        f"{context.text}\n"
    )

    if conversation_history:
        prompt_parts.append("Previous conversation:\n")
        for msg in conversation_history:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                prompt_parts.append(f"User: {content}\n")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}\n")
        prompt_parts.append("\n")

    prompt_parts.append(f"User: {user_message}")

    full_prompt = "\n".join(prompt_parts)

    async with agent.run_stream(full_prompt) as result:
        async for chunk in result.stream_text(delta=True):
            yield chunk


def count_sources_cited(response: str) -> int:
    """Count structured citations in the format [Doc: ... | Page ...]."""
    structured = len(re.findall(r"\[Doc:\s*[^]]+\|\s*Page\s*\d+\]", response, re.IGNORECASE))
    if structured > 0:
        return structured
    patterns = [
        r"section\s+\d+",
        r"clause\s+\d+",
        r"page\s+\d+",
        r"paragraph\s+\d+",
    ]
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, response, re.IGNORECASE))
    return count
