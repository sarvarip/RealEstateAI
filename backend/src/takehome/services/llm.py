from __future__ import annotations

import asyncio
import json
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
from pydantic_graph.nodes import End
from sqlalchemy import select
from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from takehome.config import settings  # noqa: F401 — triggers ANTHROPIC_API_KEY export
from takehome.db.models import Document, Message
from takehome.services.chunking import estimate_tokens
from takehome.services.embedding import async_embed_query

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


class PlannedSection(BaseModel):
    """A single section proposed by the planning agent (Stage 2).

    The search_query is used in Stage 3 for programmatic RAG — it is NOT
    executed by the planning agent itself (which has no tools).
    """

    title: str
    description: str
    search_query: str  # 3-6 keywords for programmatic embedding search


class SectionPlan(BaseModel):
    """Structured output from the planning LLM — list of report sections.

    The planning agent produces this in a single LLM call with no tool access,
    guaranteeing bounded execution time.
    """

    sections: list[PlannedSection]


# ---------------------------------------------------------------------------
# Agent dependencies
# ---------------------------------------------------------------------------

ToolCallbackT = Callable[[str, dict[str, object]], Awaitable[None]]


@dataclass
class ReportSection:
    """Definition of a single report section."""

    id: str
    title: str
    description: str


REPORT_SECTIONS: list[ReportSection] = [
    ReportSection("property", "Property", "Address, description, and floor area"),
    ReportSection("title", "Title", "Title number, registered proprietor, class of title"),
    ReportSection("tenure", "Tenure", "Freehold or leasehold, term, commencement date"),
    ReportSection("parties", "Parties", "Current landlord and tenant (and any predecessors)"),
    ReportSection("rent", "Rent", "Current passing rent, review dates, review mechanism"),
    ReportSection("rent_review", "Rent Review", "Basis of review (open market, RPI, etc.), most recent review"),
    ReportSection("break_clauses", "Break Clauses", "Tenant or landlord break rights, conditions, notice periods"),
    ReportSection("permitted_use", "Permitted Use", "Authorised use under the lease"),
    ReportSection("alienation", "Alienation", "Assignment, subletting, sharing restrictions"),
    ReportSection("repair_insurance", "Repair & Insurance", "Repairing obligations, insurance responsibility"),
    ReportSection("service_charge", "Service Charge", "Basis, cap, any disputes"),
    ReportSection("key_dates", "Key Dates", "Summary of critical upcoming dates"),
]


@dataclass
class AgentDeps:
    """Runtime dependencies injected into agent tool calls via RunContext.

    Shared across all tools within a single agent run. Mutable fields
    (doc_summary, phase, report_result) are set by tools during execution.

    Uses session_factory instead of a long-lived session so that DB connections
    are held only for the duration of each query, not the entire LLM run.
    """

    session_factory: async_sessionmaker[AsyncSession]
    conversation_id: str
    documents: list[Document]
    user_message: str = ""
    on_tool_call: ToolCallbackT | None = None  # SSE status event callback
    doc_summary: str | None = None  # populated by summary agent (Stage 1)
    phase: str | None = None  # current pipeline phase for SSE labels
    report_result: StructuredResult | None = None  # set by report tool → triggers Agent.iter() termination


# ---------------------------------------------------------------------------
# Agentic agent with tools
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a helpful legal document assistant for commercial real estate lawyers. "
    "You help lawyers review and understand documents during due diligence.\n\n"
    "TOOL ROUTING:\n"
    "- If the user asks for a report, writeup, analysis, summary document, or any "
    "comprehensive structured output, call generate_comprehensive_report() and "
    "return its result. Do NOT call search_documents() first.\n"
    "- If the user asks to add, remove, or change sections in an existing report plan, "
    "call modify_plan() with their request. Do NOT call generate_comprehensive_report().\n\n"
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

SEARCH_TOP_K = 5

agentic_agent = Agent(
    "anthropic:claude-opus-4-6",
    deps_type=AgentDeps,
    output_type=StructuredAnswer,
    instructions=_SYSTEM_PROMPT,
    retries=4,
)


@agentic_agent.tool
async def search_documents(ctx: RunContext[AgentDeps], query: str) -> str:
    """Semantic search across all conversation documents. Returns the most relevant chunks.

    Call this to find content related to a specific topic. You may call it
    multiple times with different queries to gather all the information needed.

    Args:
        query: A natural-language search query (e.g. "current annual rent").
    """
    # Defence-in-depth: if the report pipeline already ran and stored its
    # result, short-circuit so the agent can't waste tokens on extra calls.
    # The primary guard is Agent.iter() in answer_with_citations; this is backup.
    if ctx.deps.report_result is not None:
        return "Report already generated. Produce your final output now."

    # Emit SSE status event so the frontend can show "Searching: <query>"
    if ctx.deps.on_tool_call:
        details: dict[str, object] = {"query": query}
        if ctx.deps.phase:
            details["phase"] = ctx.deps.phase
        await ctx.deps.on_tool_call("searching", details)

    if not settings.embeddings_enabled:
        return "Embeddings are not available. Use the full document text provided above."

    try:
        query_embedding = await async_embed_query(query)
    except RuntimeError:
        return "Embeddings are not available. Use the full document text provided above."

    # Cosine distance search via pgvector (<=> operator).
    # Returns the top-K most similar chunks across all documents in this conversation.
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

    async with ctx.deps.session_factory() as session:
        result = await session.execute(
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
    # Defence-in-depth short-circuit (see search_documents for explanation)
    if ctx.deps.report_result is not None:
        return "Report already generated. Produce your final output now."

    if ctx.deps.on_tool_call:
        details: dict[str, object] = {"filename": filename, "page": page}
        if ctx.deps.phase:
            details["phase"] = ctx.deps.phase
        await ctx.deps.on_tool_call("reading", details)

    doc_map = _build_doc_map(ctx.deps.documents)
    doc = doc_map.get(filename.lower())
    if not doc:
        return f"Document '{filename}' not found."

    page_text = _get_page_text(doc, page)
    if not page_text:
        return f"Page {page} not found in '{filename}'."

    logger.info("get_page tool called", filename=filename, page=page)

    return f'<page filename="{doc.filename}" page="{page}">\n{page_text.strip()}\n</page>'


# ---------------------------------------------------------------------------
# Document summary agent (pre-report context gathering)
# ---------------------------------------------------------------------------
# Stage 1 of the report pipeline. Reads the first few pages of each document
# to produce a concise overview. This gives the planning agent (Stage 2)
# enough context to propose meaningful report sections without blind exploration.
# Only has access to get_page — no search tool needed since it reads sequentially.

_summary_agent = Agent(
    "anthropic:claude-opus-4-6",
    deps_type=AgentDeps,
    output_type=str,
    instructions=(
        "You are a document analyst preparing a brief overview of legal documents. "
        "You will be given document titles and page counts. Use get_page() to read "
        "the first 1-3 pages of each document to understand what it is about. "
        "Do NOT read more than 3 pages per document — the first few pages are enough. "
        "Produce a concise summary (2-4 sentences per document) covering: "
        "document type, key parties, property, and main terms."
    ),
    retries=2,
)

_summary_agent.tool(get_page)


async def _generate_doc_summaries(deps: AgentDeps) -> str:
    """Run the summary agent to produce a brief overview of each document."""
    doc_list = "\n".join(
        f"  - {doc.filename} ({doc.page_count or '?'} pages)"
        for doc in deps.documents
    )
    prompt = (
        f"You have {len(deps.documents)} document(s):\n{doc_list}\n\n"
        "Read the first 1-3 pages of each document to understand what it contains, "
        "then provide a brief summary of each."
    )

    if deps.on_tool_call:
        await deps.on_tool_call("summarising", {"doc_count": len(deps.documents)})

    summary_deps = AgentDeps(
        session_factory=deps.session_factory,
        conversation_id=deps.conversation_id,
        documents=deps.documents,
        on_tool_call=deps.on_tool_call,
        phase="summary",
    )

    result = await _summary_agent.run(prompt, deps=summary_deps)
    summary = str(result.output)

    logger.info("Document summaries generated", length=len(summary), summary=summary)
    return summary


# ---------------------------------------------------------------------------
# Planning agent (no tools — structured output only)
# ---------------------------------------------------------------------------
# Stage 2 of the report pipeline. Has NO tools — this is deliberate.
# Agents with tools will use them unpredictably. By removing tools entirely,
# we guarantee a single bounded LLM call that produces a structured SectionPlan.
# The search_query field in each PlannedSection is executed programmatically
# in Stage 3 (one DB query per section), not by the agent.

_planning_agent = Agent(
    "anthropic:claude-opus-4-6",
    output_type=SectionPlan,
    instructions=(
        "You are a legal document analyst planning a Report on Title. "
        "Given a document summary and user request, decide which sections "
        "the report should contain. For each section, provide a title, "
        "a brief description, and a search query to find the most relevant "
        "passage in the documents. Choose 5-10 sections. "
        "Each search_query should be 3-6 keywords that would match "
        "relevant document content (e.g. 'current annual rent review dates')."
    ),
    retries=2,
)


# ---------------------------------------------------------------------------
# Report generation tool (summary + plan + programmatic search)
# ---------------------------------------------------------------------------
# This tool is called by the main agent when the user asks for a report.
# It orchestrates the full 3-stage pipeline:
#   Stage 1: _summary_agent reads first pages → doc summary
#   Stage 2: _planning_agent proposes sections → SectionPlan (no tools)
#   Stage 3: _search_and_build_proposal runs one RAG query per section (programmatic)
#
# After completion, it sets ctx.deps.report_result. The Agent.iter() loop in
# answer_with_citations checks this after each step and terminates immediately,
# preventing the main agent from making any further LLM calls.

@agentic_agent.tool
async def generate_comprehensive_report(ctx: RunContext[AgentDeps], report_type: str = "report_on_title") -> str:
    """Generate a comprehensive structured report from the uploaded documents.

    Call this when the user asks for a report, writeup, analysis, summary
    document, or any comprehensive structured output about the property or
    lease.

    Args:
        report_type: Type of report. Currently supported: "report_on_title".
    """
    # Stage 1: Summary agent reads first pages of each document
    if ctx.deps.on_tool_call:
        await ctx.deps.on_tool_call("summarising", {"doc_count": len(ctx.deps.documents)})

    doc_summary = await _generate_doc_summaries(ctx.deps)
    ctx.deps.doc_summary = doc_summary

    # Stage 2: Planning agent (no tools) proposes sections with search queries
    if ctx.deps.on_tool_call:
        await ctx.deps.on_tool_call("planning", {"phase": "planning"})

    sections_list = "\n".join(
        f'  - {s.title} — {s.description}' for s in REPORT_SECTIONS
    )
    planning_prompt = (
        f"DOCUMENT OVERVIEW:\n{doc_summary}\n\n"
        f"USER REQUEST:\n{ctx.deps.user_message}\n\n"
        f"REFERENCE SECTIONS (use as a starting point, adapt as needed):\n{sections_list}\n\n"
        "Choose which sections to include in the report."
    )
    plan_result = await _planning_agent.run(planning_prompt)
    plan: SectionPlan = plan_result.output

    logger.info(
        "Planning agent completed",
        section_count=len(plan.sections),
        sections=[s.title for s in plan.sections],
    )

    # Stage 3: Programmatic search — one embed+query per section, no agent involved.
    # This guarantees exactly N searches for N sections (bounded, deterministic).
    ctx.deps.phase = "planning"

    proposal = await _search_and_build_proposal(
        plan=plan,
        session_factory=ctx.deps.session_factory,
        conversation_id=ctx.deps.conversation_id,
        documents=ctx.deps.documents,
        on_tool_call=ctx.deps.on_tool_call,
        doc_summary=doc_summary,
    )

    # Store result so Agent.iter() in answer_with_citations can detect it
    # and terminate the main agent immediately after this tool returns.
    ctx.deps.report_result = proposal
    logger.info("Report pipeline completed", sections=len(plan.sections))
    return "Report proposal generated and returned to the user."


@agentic_agent.tool
async def modify_plan(ctx: RunContext[AgentDeps], user_request: str) -> str:
    """Modify an existing report proposal based on user feedback.

    Call this when the user asks to add, remove, or change sections
    in a previously generated report plan. Do NOT call
    generate_comprehensive_report — this tool reuses the existing
    document summary and re-runs only the planning step.

    Args:
        user_request: What the user wants changed (e.g. "add a section about break clauses").
    """
    if ctx.deps.on_tool_call:
        await ctx.deps.on_tool_call("planning", {"phase": "modifying plan"})

    # --- Load the most recent proposal from the DB ---
    async with ctx.deps.session_factory() as session:
        stmt = (
            select(Message)
            .where(Message.conversation_id == ctx.deps.conversation_id)
            .where(Message.role == "assistant")
            .order_by(Message.created_at.desc())
        )
        result = await session.execute(stmt)
        recent_messages = list(result.scalars().all())

    doc_summary: str | None = None
    existing_sections: list[dict[str, str]] = []

    for msg in recent_messages:
        if not msg.citations_json:
            continue
        try:
            raw = json.loads(msg.citations_json)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(raw, dict) or raw.get("version") != 2:
            continue
        if raw.get("proposed_sections") and raw.get("doc_summary"):
            doc_summary = raw["doc_summary"]
            existing_sections = raw["proposed_sections"]
            break

    if not doc_summary or not existing_sections:
        return (
            "No existing report proposal found in this conversation. "
            "Call generate_comprehensive_report() first to create an initial proposal."
        )

    # --- Re-run planning agent with existing context + user modification request ---
    existing_titles = "\n".join(
        f"  - {s['title']}: {s.get('description', '')}" for s in existing_sections
    )

    planning_prompt = (
        f"You are modifying an existing report plan.\n\n"
        f"CURRENT SECTIONS:\n{existing_titles}\n\n"
        f"DOCUMENT SUMMARY:\n{doc_summary}\n\n"
        f"USER REQUEST:\n{user_request}\n\n"
        "Produce a new SectionPlan incorporating the user's changes. "
        "Keep existing sections unless the user explicitly asks to remove them."
    )

    plan_result = await _planning_agent.run(planning_prompt)
    plan: SectionPlan = plan_result.output

    logger.info(
        "Modified plan generated",
        section_count=len(plan.sections),
        sections=[s.title for s in plan.sections],
    )

    # --- Re-run programmatic search with the updated plan ---
    ctx.deps.phase = "planning"
    proposal = await _search_and_build_proposal(
        plan=plan,
        session_factory=ctx.deps.session_factory,
        conversation_id=ctx.deps.conversation_id,
        documents=ctx.deps.documents,
        on_tool_call=ctx.deps.on_tool_call,
        doc_summary=doc_summary,
    )

    # Same termination mechanism as generate_comprehensive_report
    ctx.deps.report_result = proposal
    logger.info("Plan modification completed", sections=len(plan.sections))
    return "Modified report proposal generated and returned to the user."


async def _search_and_build_proposal(
    plan: SectionPlan,
    session_factory: async_sessionmaker[AsyncSession],
    conversation_id: str,
    documents: list[Document],
    on_tool_call: ToolCallbackT | None,
    doc_summary: str,
) -> StructuredResult:
    """Stage 3: Execute exactly one RAG search per planned section programmatically.

    For each section in the plan, embeds the search_query, retrieves the single
    most relevant chunk from pgvector, and builds a proposal StructuredResult
    with one segment + citation per section. This is sent to the frontend as
    a sections_proposal SSE event for the user to select from.
    """
    if not settings.embeddings_enabled:
        return StructuredResult(
            content="Embeddings not available for report generation.",
            segments=[{"text": "Embeddings are required for report generation.", "citations": []}],
        )

    doc_map = _build_doc_map(documents)
    segments: list[dict[str, object]] = []
    flat_citations: list[Citation] = []
    proposed_sections: list[dict[str, str]] = []  # sent to frontend for checklist

    for i, sec in enumerate(plan.sections):
        # Emit SSE event so frontend shows "Planning: Searching <query>"
        if on_tool_call:
            await on_tool_call("searching", {"query": sec.search_query, "phase": "planning"})

        try:
            query_embedding = await async_embed_query(sec.search_query)
        except RuntimeError:
            segments.append({
                "text": f"**{i+1}. {sec.title}** — Could not search (embeddings unavailable)",
                "citations": [],
            })
            continue

        # Retrieve the single best-matching chunk for this section's search query.
        # LIMIT 1 — we only need one representative citation per section for the proposal.
        stmt = sa_text("""
            SELECT dc.content, dc.page_number, d.filename,
                   dc.embedding <=> :query_embedding AS distance
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE d.conversation_id = :conversation_id
              AND dc.embedding IS NOT NULL
            ORDER BY dc.embedding <=> :query_embedding
            LIMIT 1
        """)

        async with session_factory() as session:
            result = await session.execute(
                stmt,
                {
                    "query_embedding": str(query_embedding),
                    "conversation_id": conversation_id,
                    "top_k": 1,
                },
            )
            row = result.fetchone()

        # Build a URL-safe section ID from the title for frontend tracking
        section_id = re.sub(r"[^a-z0-9]+", "_", sec.title.lower()).strip("_")
        proposed_sections.append({
            "id": section_id,
            "title": sec.title,
            "description": sec.description,
        })

        if row:
            score = round(1 - row.distance, 3)
            # Use first 80 chars of the chunk as a preliminary quote
            quote_text = row.content[:80].strip().replace("\n", " ")
            matched_doc = doc_map.get(row.filename.lower())

            cite = Citation(
                index=len(flat_citations) + 1,
                filename=row.filename,
                page=row.page_number,
                quote=quote_text,
                document_id=matched_doc.id if matched_doc else None,
            )
            flat_citations.append(cite)

            segments.append({
                "text": f"**{i+1}. {sec.title}** — {sec.description}",
                "citations": [{
                    "index": cite.index,
                    "document_id": cite.document_id,
                    "filename": cite.filename,
                    "page": cite.page,
                    "quote": cite.quote,
                    "verified": cite.verified,
                }],
            })

            logger.info(
                "Proposal search",
                section=sec.title,
                query=sec.search_query,
                score=score,
                filename=row.filename,
                page=row.page_number,
            )
        else:
            segments.append({
                "text": f"**{i+1}. {sec.title}** — No relevant content found",
                "citations": [],
            })

    # Run citation verification on the preliminary quotes
    flat_citations = verify_citations(flat_citations, documents)
    _sync_verified_flags(segments, flat_citations)

    content = " ".join(
        str(seg["text"]) for seg in segments
    )

    return StructuredResult(
        content=content,
        segments=segments,
        citations=flat_citations,
        doc_summary=doc_summary,
        proposed_sections=proposed_sections,
    )


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
    doc_summary: str | None = None
    proposed_sections: list[dict[str, str]] | None = None


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
    session_factory: async_sessionmaker[AsyncSession],
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
        session_factory=session_factory,
        conversation_id=conversation_id,
        documents=documents,
        user_message=user_message,
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
                "The COMPLETE document text is provided below — do NOT call "
                "search_documents() or get_page() since you already have "
                "everything. Extract quotes directly from the text below.\n\n"
                f"{full_text}\n\n"
                f"Question: {user_message}"
            )
            mode = "full"
            logger.info(
                "Using full-context mode",
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

    # CRITICAL: Use iter() instead of run() for step-by-step execution control.
    #
    # Agent.run() blocks until the agent finishes ALL tool calls and produces
    # its final StructuredAnswer. If generate_comprehensive_report is called,
    # run() would let the agent continue calling search_documents/get_page
    # AFTER the report is already done — wasting tokens.
    #
    # Agent.iter() yields execution nodes one at a time. After each node
    # (model request or tool execution batch), we check if the report pipeline
    # stored its result in deps.report_result. If so, we return immediately —
    # the agent is terminated with zero additional LLM calls.
    async with agentic_agent.iter(
        user_prompt,
        deps=deps,
        message_history=message_history,
    ) as agent_run:
        node = agent_run.next_node
        while not isinstance(node, End):
            node = await agent_run.next(node)
            # Check after every step: did the report tool finish?
            if deps.report_result is not None:
                logger.info("Report tool completed — terminating agent (no further LLM calls)")
                return deps.report_result

    # Normal Q&A path: agent finished naturally, extract its structured output
    answer: StructuredAnswer = agent_run.result.output  # type: ignore[union-attr]

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


# ---------------------------------------------------------------------------
# Phase 2: Parallel report section execution
# ---------------------------------------------------------------------------
# After the user selects sections from the Phase 1 proposal, this code runs
# one section_agent per selected section IN PARALLEL (asyncio.gather).
# Each section agent has search_documents + get_page tools and the doc_summary
# from Phase 1 for context. It produces a small StructuredAnswer (~300 words)
# for its section. All sections are then combined, citations verified, and
# returned as a single StructuredResult.

_SECTION_SYSTEM_PROMPT = (
    "You are a legal document assistant writing ONE section of a Report on Title.\n\n"
    "ANSWER FORMAT:\n"
    "- Write a thorough analysis for this section only.\n"
    "- Break your answer into 3-5 segments (1-3 sentences each).\n"
    "- Keep the section to roughly one page (~300 words). Be thorough but concise.\n"
    "- Every factual claim must have a citation with a verbatim quote.\n"
    "- Use get_page() to verify exact quote wording when needed.\n"
    "- Do NOT repeat the section header — it is handled externally.\n"
    "- Be precise and concise. Lawyers value accuracy."
)

# Each section agent gets its own search + get_page tools so it can
# independently research its assigned section topic.
section_agent = Agent(
    "anthropic:claude-opus-4-6",
    deps_type=AgentDeps,
    output_type=StructuredAnswer,
    instructions=_SECTION_SYSTEM_PROMPT,
    retries=3,
)

section_agent.tool(search_documents)
section_agent.tool(get_page)


async def _run_section_agent(
    title: str,
    description: str,
    doc_summary: str,
    deps: AgentDeps,
) -> tuple[str, StructuredAnswer]:
    """Run a single section agent with doc summary for context."""
    prompt = (
        f"Section: **{title}** — {description}\n\n"
        "DOCUMENT OVERVIEW:\n"
        f"{doc_summary}\n\n"
        "Using search_documents() and get_page(), research and write this section. "
        "Search for the most relevant content, verify quotes, then produce your analysis."
    )
    result = await section_agent.run(prompt, deps=deps)
    return title, result.output


async def execute_report_sections(
    sections: list[dict[str, str]],
    doc_summary: str,
    session_factory: async_sessionmaker[AsyncSession],
    conversation_id: str,
    documents: list[Document],
    on_tool_call: ToolCallbackT | None = None,
) -> StructuredResult:
    """Execute Phase 2: run parallel agents for selected report sections.

    Each section agent gets the document summary for context and uses
    search_documents() + get_page() to research its section.
    """
    if not sections:
        return StructuredResult(content="No valid sections selected.", segments=[], citations=[])

    async def _run_one(sec: dict[str, str]) -> tuple[str, StructuredAnswer]:
        """Run a single section agent. Wraps the tool callback to prefix SSE
        events with the section title (e.g. "Rent: Searching...")."""
        sec_title = sec["title"]

        # Wrap the on_tool_call callback to add section context to SSE events
        section_callback: ToolCallbackT | None = None
        if on_tool_call is not None:
            _captured_on_tool_call = on_tool_call

            async def _prefixed_callback(
                status: str,
                details: dict[str, object],
                _title: str = sec_title,
            ) -> None:
                await _captured_on_tool_call(status, {"section": _title, **details})
            section_callback = _prefixed_callback

        section_deps = AgentDeps(
            session_factory=session_factory,
            conversation_id=conversation_id,
            documents=documents,
            on_tool_call=section_callback,
        )
        return await _run_section_agent(
            sec_title, sec.get("description", ""), doc_summary, section_deps,
        )

    # Launch all section agents in parallel. Each one independently searches
    # and reads pages to write its section. asyncio.gather runs them concurrently.
    tasks = [_run_one(sec) for sec in sections]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    doc_map = _build_doc_map(documents)
    all_segments: list[dict[str, object]] = []
    all_citations: list[Citation] = []
    content_parts: list[str] = []

    for sec, result in zip(sections, results, strict=True):
        title = sec["title"]
        if isinstance(result, BaseException):
            logger.error("Section agent failed", section=title, error=str(result))
            all_segments.append({"text": f"**{title}**", "citations": []})
            all_segments.append({
                "text": f"Error generating this section: {result}",
                "citations": [],
            })
            content_parts.append(f"**{title}**\n\nError generating this section.")
            continue

        _sec_title, answer = result
        all_segments.append({"text": f"**{title}**", "citations": []})
        content_parts.append(f"**{title}**")

        for seg in answer.segments:
            seg_citations: list[dict[str, object]] = []
            for cref in seg.citations:
                matched_doc = doc_map.get(cref.filename.lower())
                cite = Citation(
                    index=len(all_citations) + 1,
                    filename=cref.filename,
                    page=cref.page,
                    quote=cref.quote,
                    document_id=matched_doc.id if matched_doc else None,
                )
                all_citations.append(cite)
                seg_citations.append({
                    "index": cite.index,
                    "document_id": cite.document_id,
                    "filename": cite.filename,
                    "page": cite.page,
                    "quote": cite.quote,
                    "verified": cite.verified,
                })
            all_segments.append({"text": seg.text, "citations": seg_citations})
            content_parts.append(seg.text)

    all_citations = verify_citations(all_citations, documents)
    all_citations = await verify_citations_secondary(all_citations, documents)
    _sync_verified_flags(all_segments, all_citations)

    content = " ".join(content_parts)

    logger.info(
        "Report sections executed",
        sections=len(sections),
        total_segments=len(all_segments),
        total_citations=len(all_citations),
    )

    return StructuredResult(content=content, segments=all_segments, citations=all_citations)


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
