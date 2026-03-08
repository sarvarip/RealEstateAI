from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import datetime

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import StreamingResponse

from takehome.db.models import Message
from takehome.db.session import get_session
from takehome.services.conversation import get_conversation, update_conversation
from takehome.services.document import get_documents_for_conversation
from takehome.services.llm import (
    StructuredResult,
    answer_with_citations,
    execute_report_sections,
    generate_title,
)

logger = structlog.get_logger()

router = APIRouter(tags=["messages"])


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #


class CitationOut(BaseModel):
    index: int
    document_id: str | None
    filename: str
    page: int
    quote: str
    verified: bool


class SegmentOut(BaseModel):
    text: str
    citations: list[CitationOut]


class MessageOut(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    sources_cited: int
    citations: list[CitationOut]
    segments: list[SegmentOut] | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class MessageCreate(BaseModel):
    content: str
    report_sections: list[dict[str, str]] | None = None
    doc_summary: str | None = None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _message_to_out(m: Message) -> MessageOut:
    """Convert a DB Message to a MessageOut, handling both legacy and v2 formats."""
    citations: list[CitationOut] = []
    segments: list[SegmentOut] | None = None

    if m.citations_json:
        try:
            raw = json.loads(m.citations_json)
        except (json.JSONDecodeError, TypeError):
            raw = None

        if isinstance(raw, dict) and raw.get("version") == 2:
            segments = []
            for seg_data in raw.get("segments", []):
                seg_cites = [CitationOut(**c) for c in seg_data.get("citations", [])]
                citations.extend(seg_cites)
                segments.append(SegmentOut(text=seg_data["text"], citations=seg_cites))
        elif isinstance(raw, list):
            for c in raw:
                citations.append(CitationOut(**{
                    k: v for k, v in c.items()
                    if k in CitationOut.model_fields
                }))

    return MessageOut(
        id=m.id,
        conversation_id=m.conversation_id,
        role=m.role,
        content=m.content,
        sources_cited=m.sources_cited,
        citations=citations,
        segments=segments,
        created_at=m.created_at,
    )


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@router.get(
    "/api/conversations/{conversation_id}/messages",
    response_model=list[MessageOut],
)
async def list_messages(
    conversation_id: str,
    session: AsyncSession = Depends(get_session),
) -> list[MessageOut]:
    """List all messages in a conversation, ordered by creation time."""
    conversation = await get_conversation(session, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
    )
    result = await session.execute(stmt)
    messages = list(result.scalars().all())

    return [_message_to_out(m) for m in messages]


@router.post("/api/conversations/{conversation_id}/messages")
async def send_message(
    conversation_id: str,
    body: MessageCreate,
    rag_threshold: int | None = Query(None, description="Override RAG token threshold (for testing)"),
    session: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    """Send a user message and return the AI response via SSE."""
    conversation = await get_conversation(session, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    user_message = Message(
        conversation_id=conversation_id,
        role="user",
        content=body.content,
    )
    session.add(user_message)
    await session.commit()
    await session.refresh(user_message)

    logger.info("User message saved", conversation_id=conversation_id, message_id=user_message.id)

    documents = await get_documents_for_conversation(session, conversation_id)

    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .where(Message.id != user_message.id)
        .order_by(Message.created_at.asc())
    )
    result = await session.execute(stmt)
    history_messages = list(result.scalars().all())

    conversation_history: list[dict[str, str]] = [
        {"role": m.role, "content": m.content} for m in history_messages
    ]

    user_msg_count = sum(1 for m in history_messages if m.role == "user")
    is_first_message = user_msg_count == 0

    is_report_execution = bool(body.report_sections)

    async def event_stream() -> AsyncIterator[str]:
        """Generate SSE events with tool-use status and the structured LLM response."""

        yield _sse({"type": "status", "status": "thinking"})

        tool_events: asyncio.Queue[dict[str, object] | None] = asyncio.Queue()

        async def on_tool_call(status: str, details: dict[str, object]) -> None:
            await tool_events.put({"type": "status", "status": status, **details})

        async def run_agent() -> StructuredResult | None:
            try:
                if is_report_execution:
                    return await execute_report_sections(
                        sections=body.report_sections or [],
                        doc_summary=body.doc_summary or "",
                        session=session,
                        conversation_id=conversation_id,
                        documents=documents,
                        on_tool_call=on_tool_call,
                    )
                else:
                    return await answer_with_citations(
                        user_message=body.content,
                        session=session,
                        conversation_id=conversation_id,
                        conversation_history=conversation_history,
                        documents=documents,
                        rag_threshold_override=rag_threshold,
                        on_tool_call=on_tool_call,
                    )
            except Exception:
                logger.exception("Error during LLM call", conversation_id=conversation_id)
                return None
            finally:
                await tool_events.put(None)

        task = asyncio.create_task(run_agent())

        while True:
            event = await tool_events.get()
            if event is None:
                break
            yield _sse(event)

        structured = task.result()

        if structured is None:
            error_msg = "I'm sorry, an error occurred while generating a response. Please try again."
            yield _sse({"type": "content", "content": error_msg})
            yield _sse({"type": "done", "sources_cited": 0})
            return

        sources = len(structured.citations)

        citations_json = json.dumps({
            "version": 2,
            "segments": structured.segments,
        })

        yield _sse({"type": "segments", "segments": structured.segments})

        from takehome.db.session import async_session as session_factory

        async with session_factory() as save_session:
            assistant_message = Message(
                conversation_id=conversation_id,
                role="assistant",
                content=structured.content,
                sources_cited=sources,
                citations_json=citations_json,
            )
            save_session.add(assistant_message)

            # Touch the conversation's updated_at so the sidebar sorts correctly
            conv = await get_conversation(save_session, conversation_id)
            if conv is not None:
                conv.updated_at = datetime.utcnow()

            await save_session.commit()
            await save_session.refresh(assistant_message)

            if is_first_message:
                try:
                    title = await generate_title(body.content)
                    await update_conversation(save_session, conversation_id, title)
                    logger.info(
                        "Auto-generated conversation title",
                        conversation_id=conversation_id,
                        title=title,
                    )
                except Exception:
                    logger.exception("Failed to generate title", conversation_id=conversation_id)

            flat_citations = [
                {
                    "index": c.index,
                    "document_id": c.document_id,
                    "filename": c.filename,
                    "page": c.page,
                    "quote": c.quote,
                    "verified": c.verified,
                }
                for c in structured.citations
            ]

            yield _sse({
                "type": "message",
                "message": {
                    "id": assistant_message.id,
                    "conversation_id": assistant_message.conversation_id,
                    "role": assistant_message.role,
                    "content": assistant_message.content,
                    "sources_cited": assistant_message.sources_cited,
                    "citations": flat_citations,
                    "segments": structured.segments,
                    "created_at": assistant_message.created_at.isoformat(),
                },
            })

            if structured.proposed_sections and not is_report_execution:
                yield _sse({
                    "type": "sections_proposal",
                    "sections": structured.proposed_sections,
                    "doc_summary": structured.doc_summary or "",
                })

            yield _sse({
                "type": "done",
                "sources_cited": sources,
                "message_id": assistant_message.id,
            })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse(data: dict[str, object]) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"
