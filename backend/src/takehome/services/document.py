from __future__ import annotations

import os
import uuid

import fitz  # PyMuPDF
import structlog
from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from takehome.config import settings
from takehome.db.models import Document, DocumentChunk
from takehome.services.chunking import chunk_document
from takehome.services.embedding import embed_texts
from takehome.services.ocr import ocr_pages

logger = structlog.get_logger()


async def _extract_pages(
    file_path: str,
    *,
    ocr_provider_override: str | None = None,
    skip_ocr_cache: bool = False,
) -> list[tuple[int, str]]:
    """Extract per-page text from a PDF, with OCR for pages containing images.

    Returns list of (page_number, text). PyMuPDF extracts the text layer first.
    For pages with images, OCR runs and its output *replaces* PyMuPDF text
    (to avoid duplicate chunks in RAG). PyMuPDF text is kept as fallback only
    if OCR fails or produces no output for a page.
    """
    pages: list[tuple[int, str]] = []
    image_page_numbers: list[int] = []

    try:
        doc = fitz.open(file_path)
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1
            text = page.get_text()  # type: ignore[union-attr]
            pages.append((page_num, text))

            if page.get_images():  # type: ignore[union-attr]
                image_page_numbers.append(page_num)
        doc.close()
    except Exception:
        logger.exception("Failed to extract text from PDF", path=file_path)
        return pages

    effective_provider = ocr_provider_override if ocr_provider_override is not None else settings.ocr_provider
    if effective_provider == "none":
        effective_provider = None

    if image_page_numbers and effective_provider:
        logger.info(
            "Pages with images detected — running OCR",
            file_path=file_path,
            image_pages=len(image_page_numbers),
            total_pages=len(pages),
            provider=effective_provider,
        )
        try:
            ocr_results = await ocr_pages(
                file_path, image_page_numbers,
                provider_override=effective_provider,
                skip_cache=skip_ocr_cache,
            )
        except Exception:
            logger.exception("OCR failed — continuing with PyMuPDF text only")
            ocr_results = {}

        if ocr_results:
            # For pages where OCR succeeded, use OCR text exclusively — it's
            # more comprehensive than PyMuPDF and avoids duplicate chunks in RAG.
            # Only fall back to PyMuPDF text if OCR didn't produce output for a page.
            pages = [
                (pn, ocr_results[pn] if pn in ocr_results and ocr_results[pn].strip() else text)
                for pn, text in pages
            ]
            logger.info("Applied OCR text", pages_replaced=len(ocr_results))
    elif image_page_numbers:
        logger.warning(
            "Pages with images detected but no OCR provider available",
            image_pages=len(image_page_numbers),
        )

    pages = [(pn, text) for pn, text in pages if text.strip()]
    return pages


    # _merge_text removed — OCR text now replaces PyMuPDF for pages with images,
    # avoiding duplicate chunks in RAG. PyMuPDF is kept as fallback if OCR fails.


async def upload_document(
    session: AsyncSession,
    conversation_id: str,
    file: UploadFile,
    *,
    skip_embedding: bool = False,
    ocr_provider_override: str | None = None,
    skip_ocr_cache: bool = False,
) -> Document:
    """Upload and process a PDF document for a conversation.

    Validates the file, saves to disk, extracts text, chunks the content,
    embeds each chunk via Azure OpenAI, and stores everything in the database.
    Multiple documents per conversation are supported.

    Testing overrides:
      skip_embedding: if True, store chunks without embeddings
      ocr_provider_override: force a specific OCR provider ("anthropic" / "azure_di" / "none")
      skip_ocr_cache: if True, bypass OCR cache and always run live OCR
    """
    if file.content_type not in ("application/pdf", "application/x-pdf"):
        filename = file.filename or ""
        if not filename.lower().endswith(".pdf"):
            raise ValueError("Only PDF files are supported.")

    content = await file.read()

    if len(content) > settings.max_upload_size:
        raise ValueError(
            f"File too large. Maximum size is {settings.max_upload_size // (1024 * 1024)}MB."
        )

    original_filename = file.filename or "document.pdf"
    unique_name = f"{uuid.uuid4().hex}_{original_filename}"
    file_path = os.path.join(settings.upload_dir, unique_name)

    os.makedirs(settings.upload_dir, exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(content)

    logger.info("Saved uploaded PDF", filename=original_filename, path=file_path, size=len(content))

    pages = await _extract_pages(
        file_path, ocr_provider_override=ocr_provider_override, skip_ocr_cache=skip_ocr_cache
    )
    page_count = len(pages)

    full_text_parts = [f"--- Page {pn} ---\n{text}" for pn, text in pages]
    extracted_text = "\n\n".join(full_text_parts) if full_text_parts else None

    logger.info(
        "Extracted text from PDF",
        filename=original_filename,
        page_count=page_count,
        text_length=len(extracted_text) if extracted_text else 0,
    )

    document = Document(
        conversation_id=conversation_id,
        filename=original_filename,
        file_path=file_path,
        extracted_text=extracted_text,
        page_count=page_count,
    )
    session.add(document)
    await session.flush()

    chunks = chunk_document(pages)
    logger.info("Chunked document", filename=original_filename, num_chunks=len(chunks))

    chunk_texts = [c.content for c in chunks]
    if skip_embedding:
        logger.info("Embedding skipped (override)", num_chunks=len(chunks))
        embeddings: list = [None] * len(chunks)
    else:
        try:
            embeddings = embed_texts(chunk_texts)
        except Exception:
            logger.exception("Embedding failed — storing chunks without vectors")
            embeddings = [None] * len(chunks)

    for chunk, embedding in zip(chunks, embeddings):
        db_chunk = DocumentChunk(
            document_id=document.id,
            chunk_index=chunk.chunk_index,
            page_number=chunk.page_number,
            content=chunk.content,
            embedding=embedding,
        )
        session.add(db_chunk)

    await session.commit()
    await session.refresh(document)

    logger.info(
        "Document processed",
        document_id=document.id,
        filename=original_filename,
        chunks=len(chunks),
    )
    return document


async def get_document(session: AsyncSession, document_id: str) -> Document | None:
    """Get a document by its ID."""
    stmt = select(Document).where(Document.id == document_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def get_documents_for_conversation(
    session: AsyncSession, conversation_id: str
) -> list[Document]:
    """Get all documents for a conversation."""
    stmt = (
        select(Document)
        .where(Document.conversation_id == conversation_id)
        .order_by(Document.uploaded_at.asc())
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_document_for_conversation(
    session: AsyncSession, conversation_id: str
) -> Document | None:
    """Get the first document for a conversation (backwards compat)."""
    docs = await get_documents_for_conversation(session, conversation_id)
    return docs[0] if docs else None


async def get_chunks_for_conversation(
    session: AsyncSession, conversation_id: str
) -> list[DocumentChunk]:
    """Load all chunks across all documents in a conversation, with document info."""
    stmt = (
        select(DocumentChunk)
        .join(Document)
        .where(Document.conversation_id == conversation_id)
        .options(selectinload(DocumentChunk.document))
        .order_by(Document.uploaded_at.asc(), DocumentChunk.chunk_index.asc())
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())
