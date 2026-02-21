from __future__ import annotations

import os
from datetime import datetime

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import FileResponse

from takehome.db.session import get_session
from takehome.services.conversation import get_conversation
from takehome.services.document import get_document, get_documents_for_conversation, upload_document

logger = structlog.get_logger()

router = APIRouter(tags=["documents"])


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #


class DocumentOut(BaseModel):
    id: str
    conversation_id: str
    filename: str
    page_count: int
    uploaded_at: datetime

    model_config = {"from_attributes": True}


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@router.post(
    "/api/conversations/{conversation_id}/documents",
    response_model=DocumentOut,
    status_code=201,
)
async def upload_document_endpoint(
    conversation_id: str,
    file: UploadFile,
    session: AsyncSession = Depends(get_session),
    skip_embedding: bool = Query(False, description="Skip embedding (for testing without Azure OpenAI)"),
    ocr_provider: str | None = Query(None, description="Force OCR provider: anthropic, azure_di, none"),
    skip_ocr_cache: bool = Query(False, description="Bypass OCR cache and run live OCR"),
) -> DocumentOut:
    """Upload a PDF document for a conversation. Multiple documents are supported."""
    conversation = await get_conversation(session, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    try:
        document = await upload_document(
            session,
            conversation_id,
            file,
            skip_embedding=skip_embedding,
            ocr_provider_override=ocr_provider,
            skip_ocr_cache=skip_ocr_cache,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(
        "Document uploaded",
        conversation_id=conversation_id,
        document_id=document.id,
        filename=document.filename,
    )

    return DocumentOut(
        id=document.id,
        conversation_id=document.conversation_id,
        filename=document.filename,
        page_count=document.page_count,
        uploaded_at=document.uploaded_at,
    )


@router.get(
    "/api/conversations/{conversation_id}/documents",
    response_model=list[DocumentOut],
)
async def list_documents(
    conversation_id: str,
    session: AsyncSession = Depends(get_session),
) -> list[DocumentOut]:
    """List all documents in a conversation."""
    conversation = await get_conversation(session, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    documents = await get_documents_for_conversation(session, conversation_id)
    return [
        DocumentOut(
            id=d.id,
            conversation_id=d.conversation_id,
            filename=d.filename,
            page_count=d.page_count,
            uploaded_at=d.uploaded_at,
        )
        for d in documents
    ]


@router.get("/api/documents/{document_id}/content")
async def serve_document_file(
    document_id: str,
    session: AsyncSession = Depends(get_session),
) -> FileResponse:
    """Serve the raw PDF file for download/viewing."""
    document = await get_document(session, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    if not os.path.exists(document.file_path):
        raise HTTPException(status_code=404, detail="File not found on disk")

    return FileResponse(
        path=document.file_path,
        filename=document.filename,
        media_type="application/pdf",
    )
