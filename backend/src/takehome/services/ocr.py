"""OCR service: extract text from pages that contain images.

Tiered approach:
  1. Azure Document Intelligence (if keys configured) — sends the whole PDF once
  2. Anthropic Vision fallback — renders pages to PNG concurrently via asyncio

Results are cached on disk keyed by (file content hash, provider) so that
re-uploading the same PDF skips OCR entirely.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os

import fitz  # PyMuPDF
import structlog

from takehome.config import settings

logger = structlog.get_logger()

ANTHROPIC_CONCURRENCY = 10
OCR_CACHE_DIR = os.path.join(settings.upload_dir, ".ocr_cache")


def _cache_key(file_path: str, provider: str) -> str:
    """SHA-256 of file contents + provider name → deterministic cache filename."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    h.update(provider.encode())
    return h.hexdigest()


def _load_cache(file_path: str, provider: str) -> dict[int, str] | None:
    os.makedirs(OCR_CACHE_DIR, exist_ok=True)
    key = _cache_key(file_path, provider)
    cache_path = os.path.join(OCR_CACHE_DIR, f"{key}.json")
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path) as f:
            data = json.load(f)
        return {int(k): v for k, v in data.items()}
    except Exception:
        return None


def _save_cache(file_path: str, provider: str, results: dict[int, str]) -> None:
    os.makedirs(OCR_CACHE_DIR, exist_ok=True)
    key = _cache_key(file_path, provider)
    cache_path = os.path.join(OCR_CACHE_DIR, f"{key}.json")
    with open(cache_path, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f)


async def ocr_pages(
    file_path: str,
    page_numbers: list[int],
    *,
    provider_override: str | None = None,
    skip_cache: bool = False,
) -> dict[int, str]:
    """Run OCR on specific pages of a PDF and return {page_number: text}.

    page_numbers are 1-indexed (matching the rest of the codebase).
    Dispatches to Azure DI or Anthropic vision based on config (or override).
    Results are cached on disk so repeated uploads of the same PDF are instant.
    Pass skip_cache=True to force a live OCR call (useful for testing).
    """
    if not page_numbers:
        return {}

    provider = provider_override or settings.ocr_provider
    if provider is None:
        logger.warning("No OCR provider available — skipping OCR for image pages")
        return {}

    if not skip_cache:
        cached = _load_cache(file_path, provider)
        if cached is not None:
            page_set = set(page_numbers)
            hit = {k: v for k, v in cached.items() if k in page_set}
            logger.info("OCR cache hit", provider=provider, cached_pages=len(hit), requested=len(page_numbers))
            return hit

    logger.info(
        "Starting OCR (cache %s)",
        "skipped" if skip_cache else "miss",
        provider=provider,
        file_path=file_path,
        pages=len(page_numbers),
    )

    if provider == "azure_di":
        result = await asyncio.to_thread(_ocr_azure_di, file_path, page_numbers)
    else:
        result = await _ocr_anthropic_vision(file_path, page_numbers)

    if not skip_cache:
        _save_cache(file_path, provider, result)
    return result


def _ocr_azure_di(file_path: str, page_numbers: list[int]) -> dict[int, str]:
    """Use Azure Document Intelligence prebuilt-read to extract text."""
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
    from azure.core.credentials import AzureKeyCredential

    client = DocumentIntelligenceClient(
        endpoint=settings.azure_document_intelligence_endpoint,
        credential=AzureKeyCredential(settings.azure_document_intelligence_api_key),
    )

    with open(file_path, "rb") as f:
        pdf_bytes = f.read()

    poller = client.begin_analyze_document(
        "prebuilt-read",
        AnalyzeDocumentRequest(bytes_source=pdf_bytes),
    )
    result = poller.result()

    page_set = set(page_numbers)
    extracted: dict[int, str] = {}

    if result.pages:
        for page in result.pages:
            page_num = page.page_number  # 1-indexed
            if page_num not in page_set:
                continue

            lines: list[str] = []
            if page.lines:
                for line in page.lines:
                    lines.append(line.content)
            if lines:
                extracted[page_num] = "\n".join(lines)

    logger.info(
        "Azure DI OCR complete",
        pages_requested=len(page_numbers),
        pages_extracted=len(extracted),
    )
    return extracted


async def _ocr_anthropic_vision(file_path: str, page_numbers: list[int]) -> dict[int, str]:
    """Use Anthropic Claude vision to extract text, processing pages concurrently."""
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    semaphore = asyncio.Semaphore(ANTHROPIC_CONCURRENCY)

    doc = fitz.open(file_path)
    page_images: list[tuple[int, str]] = []
    for page_num in page_numbers:
        page_idx = page_num - 1
        if page_idx < 0 or page_idx >= len(doc):
            continue
        page = doc[page_idx]
        pix = page.get_pixmap(dpi=150)  # type: ignore[union-attr]
        png_bytes = pix.tobytes("png")
        b64_image = base64.b64encode(png_bytes).decode("utf-8")
        page_images.append((page_num, b64_image))
    doc.close()

    logger.info("Rendered page images", count=len(page_images))

    async def _process_page(page_num: int, b64: str) -> tuple[int, str | None]:
        async with semaphore:
            try:
                response = await client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=4096,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": b64,
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": (
                                        "Extract ALL text from this scanned document page. "
                                        "Preserve the original structure including headings, "
                                        "paragraphs, numbering, and any tabular data. "
                                        "Return only the extracted text, no commentary."
                                    ),
                                },
                            ],
                        }
                    ],
                )
                text = response.content[0].text  # type: ignore[union-attr]
                return (page_num, text if text.strip() else None)
            except anthropic.RateLimitError:
                logger.warning("Anthropic rate limit hit", page=page_num)
                await asyncio.sleep(5)
                return (page_num, None)
            except Exception:
                logger.exception("Anthropic vision OCR failed for page", page=page_num)
                return (page_num, None)

    results = await asyncio.gather(*[
        _process_page(pn, b64) for pn, b64 in page_images
    ])

    extracted: dict[int, str] = {}
    for page_num, text in results:
        if text:
            extracted[page_num] = text

    logger.info(
        "Anthropic vision OCR complete",
        pages_requested=len(page_numbers),
        pages_extracted=len(extracted),
    )
    return extracted
