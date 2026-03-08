from __future__ import annotations

import re
import unicodedata
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import structlog
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from sqlalchemy import text as sa_text
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
# Agent dependencies
# ---------------------------------------------------------------------------

ToolCallbackT = Callable[[str, dict[str, object]], Awaitable[None]]


@dataclass
class AgentDeps:
    """Runtime dependencies injected into agent tool calls."""

    session: AsyncSession
    conversation_id: str
    documents: list[Document]
    on_tool_call: ToolCallbackT | None = None


# ---------------------------------------------------------------------------
# Agentic agent with tools
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a helpful legal document assistant for commercial real estate lawyers. "
    "You help lawyers review and understand documents during due diligence.\n\n"
    "ANSWER FORMAT:\n"
    "- Answer questions based ONLY on the document content provided or retrieved.\n"
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

SEARCH_TOP_K = 10

agentic_agent = Agent(
    "anthropic:claude-opus-4-5-20251101",
    deps_type=AgentDeps,
    output_type=StructuredAnswer,
    instructions=_SYSTEM_PROMPT,
    retries=2,
)


@agentic_agent.tool
async def search_documents(ctx: RunContext[AgentDeps], query: str) -> str:
    """Semantic search across all conversation documents. Returns the most relevant chunks.

    Call this to find content related to a specific topic. You may call it
    multiple times with different queries to gather all the information needed.

    Args:
        query: A natural-language search query (e.g. "current annual rent").
    """
    if ctx.deps.on_tool_call:
        await ctx.deps.on_tool_call("searching", {"query": query})

    if not settings.embeddings_enabled:
        return "Embeddings are not available. Use the full document text provided above."

    try:
        query_embedding = embed_query(query)
    except RuntimeError:
        return "Embeddings are not available. Use the full document text provided above."

    stmt = sa_text("""
        SELECT dc.content, dc.page_number, d.filename,
               dc.embedding <=> :query_embedding AS distance
        FROM document_chunks dc
        JOIN documents d ON dc.document_id = d.id
        WHERE d.conversation_id = :conversation_id
          AND dc.embedding IS NOT NULL
        ORDER BY dc.embedding <=> :query_embedding
        LIMIT :top_k
    """)

    result = await ctx.deps.session.execute(
        stmt,
        {
            "query_embedding": str(query_embedding),
            "conversation_id": ctx.deps.conversation_id,
            "top_k": SEARCH_TOP_K,
        },
    )
    rows = result.fetchall()

    if not rows:
        return "No matching content found. Try a different query."

    parts: list[str] = []
    for row in rows:
        score = round(1 - row.distance, 3)
        parts.append(
            f'<chunk filename="{row.filename}" page="{row.page_number}" score="{score}">\n'
            f"{row.content}\n"
            f"</chunk>"
        )

    logger.info(
        "search_documents tool called",
        query=query,
        chunks_returned=len(rows),
        top_score=round(1 - rows[0].distance, 3),
    )

    return "\n\n".join(parts)


@agentic_agent.tool
async def get_page(ctx: RunContext[AgentDeps], filename: str, page: int) -> str:
    """Get the full text of a specific page from a document.

    Use this to read the complete page context — especially to verify exact
    quote wording before citing. Chunks from search_documents are excerpts;
    this returns the entire page.

    Args:
        filename: The document filename (as shown in the document list).
        page: The 1-based page number.
    """
    if ctx.deps.on_tool_call:
        await ctx.deps.on_tool_call("reading", {"filename": filename, "page": page})

    doc_map = _build_doc_map(ctx.deps.documents)
    doc = doc_map.get(filename.lower())
    if not doc:
        return f"Document '{filename}' not found."

    page_text = _get_page_text(doc, page)
    if not page_text:
        return f"Page {page} not found in '{filename}'."

    logger.info("get_page tool called", filename=filename, page=page)

    return f'<page filename="{doc.filename}" page="{page}">\n{page_text.strip()}\n</page>'


@agentic_agent.tool
async def get_report_guidelines(ctx: RunContext[AgentDeps], report_type: str = "report_on_title") -> str:
    """Get the structure and guidelines for generating a formal legal report.

    Call this when the user asks for a report, writeup, summary document,
    or any structured documentation of the property/lease. The guidelines
    specify which sections to include and what to cover in each.

    Args:
        report_type: Type of report. Currently supported: "report_on_title".
    """
    if ctx.deps.on_tool_call:
        await ctx.deps.on_tool_call("planning", {"report_type": report_type})

    logger.info("get_report_guidelines tool called", report_type=report_type)

    return _REPORT_GUIDELINES.get(report_type, _REPORT_GUIDELINES["report_on_title"])


_REPORT_GUIDELINES: dict[str, str] = {
    "report_on_title": (
        "You are generating a Report on Title for a commercial property.\n\n"
        "STRUCTURE: Produce the following sections in order. For each section, "
        "use search_documents() to find relevant content, then get_page() to verify "
        "exact quotes. Create a header segment (e.g. '**1. Property**') with no "
        "citations, followed by one or more content segments with citations.\n\n"
        "SECTIONS:\n"
        "1. **Property** — Address, description, and floor area.\n"
        "2. **Title** — Title number, registered proprietor, class of title.\n"
        "3. **Tenure** — Freehold or leasehold, term, commencement date.\n"
        "4. **Parties** — Current landlord and tenant (and any predecessors).\n"
        "5. **Rent** — Current passing rent, review dates, review mechanism.\n"
        "6. **Rent Review** — Basis of review (open market, RPI, etc.), most recent review.\n"
        "7. **Break Clauses** — Any tenant or landlord break rights, conditions, notice periods.\n"
        "8. **Permitted Use** — Authorised use under the lease.\n"
        "9. **Alienation** — Assignment, subletting, sharing restrictions.\n"
        "10. **Repair & Insurance** — Repairing obligations, insurance responsibility.\n"
        "11. **Service Charge** — Basis, cap, any disputes.\n"
        "12. **Key Dates** — Summary of critical upcoming dates.\n\n"
        "GUIDELINES:\n"
        "- Skip any section where the documents contain no relevant information, "
        "but note it was not found.\n"
        "- If a section has information across multiple documents, cross-reference "
        "and note any discrepancies.\n"
        "- Every factual claim must have a citation with a verbatim quote.\n"
        "- The user may have asked to focus on or skip certain sections — follow "
        "their instructions.\n"
    ),
}


# ---------------------------------------------------------------------------
# Title generation (uses cheap Haiku)
# ---------------------------------------------------------------------------

_title_agent = Agent(
    "anthropic:claude-haiku-4-5-20251001",
    instructions="Generate a concise 3-5 word title. Return only the title, nothing else.",
)


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
# Core Q&A — agentic structured-output call with tools
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
    """Complete result from the agentic LLM pipeline."""

    content: str
    segments: list[dict[str, object]]
    citations: list[Citation] = field(default_factory=list)


def _build_message_history(
    conversation_history: list[dict[str, str]],
) -> list[ModelMessage]:
    """Convert raw conversation history into PydanticAI's native message array."""
    messages: list[ModelMessage] = []
    for msg in conversation_history:
        if msg["role"] == "user":
            messages.append(ModelRequest(parts=[UserPromptPart(content=msg["content"])]))
        elif msg["role"] == "assistant":
            messages.append(ModelResponse(parts=[TextPart(content=msg["content"])]))
    return messages


def _build_document_list(documents: list[Document]) -> str:
    """Build a human-readable document list with page counts for the prompt."""
    lines: list[str] = []
    for doc in documents:
        lines.append(f"  - {doc.filename} ({doc.page_count or '?'} pages)")
    return "\n".join(lines)


def _build_full_text(documents: list[Document]) -> str:
    """Build the full document text for full-context mode."""
    parts: list[str] = []
    for doc in documents:
        if doc.extracted_text:
            parts.append(
                f'<document filename="{doc.filename}">\n'
                f"{doc.extracted_text}\n"
                f"</document>"
            )
    return "\n\n".join(parts)


async def answer_with_citations(
    user_message: str,
    session: AsyncSession,
    conversation_id: str,
    conversation_history: list[dict[str, str]],
    documents: list[Document],
    rag_threshold_override: int | None = None,
    on_tool_call: ToolCallbackT | None = None,
) -> StructuredResult:
    """Answer a question using the agentic pipeline with search tools.

    Determines full-context vs agentic mode based on token threshold.
    In full-context mode, provides all document text plus tools.
    In agentic mode, provides only document metadata — the agent uses
    search_documents() and get_page() to retrieve what it needs.
    """
    deps = AgentDeps(
        session=session,
        conversation_id=conversation_id,
        documents=documents,
        on_tool_call=on_tool_call,
    )

    if not documents:
        user_prompt = (
            "No documents have been uploaded yet. If the user asks about a document, "
            "let them know they need to upload one first.\n\n"
            f"Question: {user_message}"
        )
        mode = "none"
    else:
        threshold = (
            rag_threshold_override
            if rag_threshold_override is not None
            else settings.rag_token_threshold
        )
        total_text = "\n\n".join(d.extracted_text or "" for d in documents)
        total_tokens = estimate_tokens(total_text)
        doc_list = _build_document_list(documents)

        logger.info("Token estimate", total_tokens=total_tokens, threshold=threshold)

        use_full = total_tokens < threshold or not settings.embeddings_enabled

        if use_full:
            full_text = _build_full_text(documents)
            user_prompt = (
                f"You have {len(documents)} document(s):\n{doc_list}\n\n"
                f"Full document text follows. Answer from this text. "
                f"You may use get_page() to verify exact quote wording "
                f"on a specific page before citing.\n\n"
                f"{full_text}\n\n"
                f"Question: {user_message}"
            )
            mode = "full"
            logger.info(
                "Using full-context mode (tools available)",
                doc_count=len(documents),
                total_tokens=total_tokens,
            )
        else:
            user_prompt = (
                f"You have {len(documents)} document(s):\n{doc_list}\n\n"
                "Use search_documents() to find relevant content — you may call it "
                "multiple times with different queries to gather all the information "
                "you need. Use get_page() to read the full text of a specific page "
                "and verify exact quote wording before citing.\n\n"
                f"Question: {user_message}"
            )
            mode = "agentic"
            logger.info(
                "Using agentic mode (tools required)",
                doc_count=len(documents),
                total_tokens=total_tokens,
            )

    message_history = _build_message_history(conversation_history)

    result = await agentic_agent.run(
        user_prompt,
        deps=deps,
        message_history=message_history,
    )
    answer: StructuredAnswer = result.output

    logger.info("Agent completed", mode=mode, segments=len(answer.segments))

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
        seg_cites: list[dict[str, object]] = seg["citations"]  # type: ignore[assignment]
        for sc in seg_cites:
            verified_cite = cite_by_index.get(sc["index"])  # type: ignore[arg-type]
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
