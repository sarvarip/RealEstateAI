from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import structlog
from pydantic import BaseModel
from pydantic_ai import Agent
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from takehome.config import settings  # noqa: F401 — triggers ANTHROPIC_API_KEY export
from takehome.db.models import Document
from takehome.services.chunking import estimate_tokens
from takehome.services.embedding import embed_query

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Structured output models
# ---------------------------------------------------------------------------


class CitationRef(BaseModel):
    """A single citation produced by the LLM."""

    filename: str
    page: int
    quote: str


class AnswerSegment(BaseModel):
    """One logical piece of the answer, optionally backed by citations."""

    text: str
    citations: list[CitationRef] = []


class StructuredAnswer(BaseModel):
    """The full structured response: answer broken into citable segments."""

    segments: list[AnswerSegment]


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

_STRUCTURED_SYSTEM_PROMPT = (
    "You are a helpful legal document assistant for commercial real estate lawyers. "
    "You help lawyers review and understand documents during due diligence.\n\n"
    "IMPORTANT INSTRUCTIONS:\n"
    "- Answer questions based ONLY on the document content provided.\n"
    "- Break your answer into logical segments. Each segment should be one or two "
    "sentences covering a single factual point.\n"
    "- For every segment that makes a factual claim, provide at least one citation.\n"
    "- Each citation must include:\n"
    "  • filename — the exact filename of the source document\n"
    "  • page — the page number where the information appears\n"
    "  • quote — a verbatim excerpt (10-50 words) copied directly from the document\n"
    "- Segments that are purely transitional (e.g. 'Based on the documents provided:') "
    "may have an empty citations list.\n"
    "- If the answer is not in the provided documents, return a single segment saying so.\n"
    "- Be concise and precise. Lawyers value accuracy over verbosity.\n"
    "- When multiple documents are provided, cross-reference information between them "
    "and note any discrepancies."
)

structured_agent = Agent(
    "anthropic:claude-opus-4-5-20251101",
    output_type=StructuredAnswer,
    instructions=_STRUCTURED_SYSTEM_PROMPT,
    retries=2,
)

_title_agent = Agent(
    "anthropic:claude-haiku-4-5-20251001",
    instructions="Generate a concise 3-5 word title. Return only the title, nothing else.",
)

# ---------------------------------------------------------------------------
# Retrieval context (unchanged)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Title generation (uses cheap Haiku)
# ---------------------------------------------------------------------------


async def generate_title(user_message: str) -> str:
    """Generate a 3-5 word conversation title from the first user message."""
    result = await _title_agent.run(
        f"Generate a concise 3-5 word title for a conversation that starts with: '{user_message}'. "
        "Return only the title, nothing else."
    )
    title = str(result.output).strip().strip('"').strip("'")
    if len(title) > 100:
        title = title[:97] + "..."
    return title


# ---------------------------------------------------------------------------
# Core Q&A — single structured-output call
# ---------------------------------------------------------------------------


@dataclass
class Citation:
    """A single verified citation ready for storage / frontend."""

    index: int
    filename: str
    page: int
    quote: str
    document_id: str | None = None
    verified: bool = False


@dataclass
class StructuredResult:
    """Complete result from a single structured-output LLM call."""

    content: str
    segments: list[dict[str, object]]
    citations: list[Citation] = field(default_factory=list)


async def answer_with_citations(
    user_message: str,
    context: RetrievalContext,
    conversation_history: list[dict[str, str]],
    documents: list[Document],
) -> StructuredResult:
    """Answer a question with structured citations in a single LLM call.

    Returns a StructuredResult with the plain-text content, serialisable
    segments (each with text + citations), and a flat list of Citation objects.
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

    result = await structured_agent.run(full_prompt)
    answer: StructuredAnswer = result.output

    doc_map = _build_doc_map(documents)

    flat_citations: list[Citation] = []
    serialised_segments: list[dict[str, object]] = []

    for seg in answer.segments:
        seg_citations: list[dict[str, object]] = []
        for cref in seg.citations:
            matched_doc = doc_map.get(cref.filename.lower())
            cite = Citation(
                index=len(flat_citations) + 1,
                filename=cref.filename,
                page=cref.page,
                quote=cref.quote,
                document_id=matched_doc.id if matched_doc else None,
            )
            flat_citations.append(cite)
            seg_citations.append({
                "index": cite.index,
                "document_id": cite.document_id,
                "filename": cite.filename,
                "page": cite.page,
                "quote": cite.quote,
                "verified": cite.verified,
            })

        serialised_segments.append({
            "text": seg.text,
            "citations": seg_citations,
        })

    flat_citations = verify_citations(flat_citations, documents)
    flat_citations = await verify_citations_secondary(flat_citations, documents)

    _sync_verified_flags(serialised_segments, flat_citations)

    content = " ".join(seg.text for seg in answer.segments)

    return StructuredResult(
        content=content,
        segments=serialised_segments,
        citations=flat_citations,
    )


def _build_doc_map(documents: list[Document]) -> dict[str, Document]:
    """Build a case-insensitive lookup from filename (with/without extension) to Document."""
    doc_map: dict[str, Document] = {}
    for doc in documents:
        doc_map[doc.filename.lower()] = doc
        base = doc.filename.rsplit(".", 1)[0].lower()
        doc_map[base] = doc
    return doc_map


def _sync_verified_flags(
    segments: list[dict[str, object]], citations: list[Citation]
) -> None:
    """Push verified flags from flat Citation list back into serialised segments."""
    cite_by_index: dict[int, Citation] = {c.index: c for c in citations}
    for seg in segments:
        for sc in seg["citations"]:
            verified_cite = cite_by_index.get(sc["index"])
            if verified_cite:
                sc["verified"] = verified_cite.verified
                sc["quote"] = verified_cite.quote
                sc["page"] = verified_cite.page


# ---------------------------------------------------------------------------
# Citation verification
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace for matching."""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _get_page_text(doc: Document, page: int) -> str:
    """Extract the text for a specific page from a document's extracted_text."""
    if not doc.extracted_text:
        return ""
    marker = f"--- Page {page} ---"
    idx = doc.extracted_text.find(marker)
    if idx == -1:
        return ""
    start = idx + len(marker)
    next_marker = doc.extracted_text.find("\n\n--- Page ", start)
    if next_marker == -1:
        return doc.extracted_text[start:]
    return doc.extracted_text[start:next_marker]


def verify_citations(citations: list[Citation], documents: list[Document]) -> list[Citation]:
    """Verify each citation's quote against the source document text."""
    doc_by_id: dict[str, Document] = {d.id: d for d in documents}

    for cite in citations:
        if not cite.quote or not cite.document_id:
            continue

        doc = doc_by_id.get(cite.document_id)
        if not doc:
            continue

        norm_quote = _normalize(cite.quote)
        if len(norm_quote) < 5:
            cite.verified = True
            continue

        for page_offset in [0, -1, 1]:
            page_text = _get_page_text(doc, cite.page + page_offset)
            if not page_text:
                continue
            norm_page = _normalize(page_text)
            if norm_quote in norm_page:
                cite.verified = True
                if page_offset != 0:
                    cite.page = cite.page + page_offset
                break

    return citations


async def verify_citations_secondary(
    citations: list[Citation], documents: list[Document]
) -> list[Citation]:
    """For unverified citations, make a targeted LLM call to check the source."""
    doc_by_id: dict[str, Document] = {d.id: d for d in documents}
    unverified = [c for c in citations if not c.verified and c.document_id and c.quote]

    if not unverified:
        return citations

    logger.info("Running secondary citation verification", count=len(unverified))

    verification_agent = Agent(
        "anthropic:claude-haiku-4-5-20251001",
    )

    for cite in unverified:
        doc = doc_by_id.get(cite.document_id or "")
        if not doc:
            continue

        pages_text = ""
        for offset in [-1, 0, 1]:
            pt = _get_page_text(doc, cite.page + offset)
            if pt:
                pages_text += f"\n--- Page {cite.page + offset} ---\n{pt}"

        if not pages_text.strip():
            continue

        try:
            result = await verification_agent.run(
                f"I need to verify a citation. The claim quotes:\n"
                f'"{cite.quote}"\n\n'
                f"Here is the text from the document around page {cite.page}:\n"
                f"{pages_text}\n\n"
                f"Does this text contain information matching the quoted claim? "
                f"Reply with ONLY one of:\n"
                f'VERIFIED: "<exact quote from the text>" (if you find matching content)\n'
                f"UNVERIFIED (if the claim is not supported by this text)"
            )
            answer = str(result.output).strip()
            if answer.upper().startswith("VERIFIED"):
                cite.verified = True
                quote_match = re.search(r'"([^"]+)"', answer)
                if quote_match:
                    cite.quote = quote_match.group(1)
        except Exception:
            logger.exception("Secondary verification failed", citation_index=cite.index)

    return citations
