from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    page_number: int
    chunk_index: int
    content: str


CHUNK_TARGET_SIZE = 1500
CHUNK_OVERLAP = 200


def chunk_document(pages: list[tuple[int, str]]) -> list[Chunk]:
    """Split extracted pages into chunks suitable for embedding and retrieval.

    Each page becomes one or more chunks. Long pages are split at paragraph
    boundaries with overlap to preserve context across chunk boundaries.

    Args:
        pages: list of (page_number, page_text) tuples (1-indexed page numbers).

    Returns:
        Ordered list of Chunk objects with global chunk_index.
    """
    chunks: list[Chunk] = []
    global_index = 0

    for page_num, text in pages:
        text = text.strip()
        if not text:
            continue

        if len(text) <= CHUNK_TARGET_SIZE:
            chunks.append(Chunk(page_number=page_num, chunk_index=global_index, content=text))
            global_index += 1
        else:
            page_chunks = _split_long_text(text)
            for sub_text in page_chunks:
                chunks.append(
                    Chunk(page_number=page_num, chunk_index=global_index, content=sub_text)
                )
                global_index += 1

    return chunks


def _split_long_text(text: str) -> list[str]:
    """Split text longer than CHUNK_TARGET_SIZE into overlapping segments.

    Tries to break at paragraph boundaries (double newline), falling back
    to sentence boundaries, then hard character splits.
    """
    paragraphs = text.split("\n\n")
    result: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = (current + "\n\n" + para).strip() if current else para.strip()

        if len(candidate) <= CHUNK_TARGET_SIZE:
            current = candidate
        else:
            if current:
                result.append(current)
            if len(para) > CHUNK_TARGET_SIZE:
                result.extend(_hard_split(para))
                current = ""
            else:
                overlap_text = current[-CHUNK_OVERLAP:] if current else ""
                current = (overlap_text + "\n\n" + para).strip() if overlap_text else para.strip()

    if current:
        result.append(current)

    return result


def _hard_split(text: str) -> list[str]:
    """Last-resort character-level split with overlap."""
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = start + CHUNK_TARGET_SIZE
        parts.append(text[start:end])
        start = end - CHUNK_OVERLAP
    return parts


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token for English text."""
    return len(text) // 4
