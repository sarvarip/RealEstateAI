# Approach

## Feature 1: Cross-Document Analysis with RAG

### Architecture Overview

The system uses a **tiered retrieval strategy**: full-context mode for small document sets, and RAG (Retrieval-Augmented Generation) for larger ones. The threshold is configurable (default: 100K tokens). This avoids unnecessary embedding overhead for small uploads while scaling to large document corpora.

### Document Processing Pipeline

1. **PDF extraction** — PyMuPDF extracts text per page. Scanned/image pages fall back to OCR (Azure Document Intelligence if available, otherwise Anthropic vision).
2. **Chunking** — Each page is split into chunks for embedding.
3. **Embedding** — Chunks are embedded via Azure OpenAI and stored in pgvector.
4. **Retrieval** — At query time, the user's question is embedded and the top-K most similar chunks are retrieved.

### Chunking Strategy

| Parameter | Value | Location |
|-----------|-------|----------|
| Target chunk size | 1,500 characters (~375 tokens) | `chunking.py:13` |
| Overlap | 200 characters | `chunking.py:14` |
| Strategy | Paragraph → sentence → hard split | `chunking.py:51-79` |

Chunks are character-based, not token-based (token count is estimated as `len(text) // 4`). The chunker tries to split at natural paragraph boundaries (`\n\n`), falls back to sentence boundaries for oversized paragraphs, and uses hard character splits with overlap as a last resort.

**Trade-off**: Legal documents often have well-defined sections (e.g., clauses, schedules, recitals). A section-aware chunker that respects these structural boundaries would produce more semantically coherent chunks. However, since section structure varies widely across document types (leases vs. deeds vs. memoranda) and is not reliably detectable from raw text extraction, the current paragraph-based approach is a pragmatic default. A production system could use document layout analysis (e.g., from Azure Document Intelligence's paragraph/section output) to improve chunk boundaries.

### Embedding

| Parameter | Value | Location |
|-----------|-------|----------|
| Model | `text-embedding-3-large` (Azure OpenAI) | `config.py:13` |
| Dimensions | 3,072 | `config.py:14` |
| Batch size | 16 | `embedding.py:43` |

The system gracefully degrades when Azure keys are not configured — embeddings are skipped and the app operates in full-context mode only.

### Vector Storage & Search

| Parameter | Value | Location |
|-----------|-------|----------|
| Database | PostgreSQL 16 + pgvector | `002_add_document_chunks.py` |
| Distance metric | Cosine distance (`<=>`) | `llm.py:175` |
| Top-K | 20 chunks | `config.py:20` |
| RAG threshold | 100,000 tokens | `config.py:19` |
| Vector index | None (sequential scan) | `002_add_document_chunks.py` |

**No HNSW/IVFFlat index** is created on the embedding column. At the current scale (legal due diligence with a handful of documents per conversation), sequential scan over pgvector is fast enough. For a production system with thousands of documents, adding an HNSW index would be essential:

```sql
CREATE INDEX ON document_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

**Why cosine distance?** Cosine similarity is standard for text embeddings because it measures directional similarity regardless of vector magnitude. The `<=>` operator in pgvector computes cosine distance (1 - cosine similarity), so lower values = more similar.

### Retrieval Flow

```
User question
    │
    ▼
Estimate total document tokens
    │
    ├─ < 100K tokens ──► Full-context mode (send all text to LLM)
    │
    └─ ≥ 100K tokens ──► RAG mode:
                            1. Embed user question
                            2. Query top-20 chunks by cosine distance
                            3. Send retrieved chunks to LLM
```

---

## Feature 2: Citation & Grounding with Structured Outputs

### Architecture Overview

The citation system uses **PydanticAI structured outputs** to produce a list of `{text, citations}` segments in a single LLM call. This replaces an earlier regex-based approach that parsed inline `[Doc: ...]` markers from free-text output — which was fragile and led to misaligned citations.

### Why Structured Outputs?

The initial implementation embedded citation markers in the LLM's free-text response (e.g., `[Doc: lease.pdf | Page 5 | "exact quote"]`) and parsed them with regex. This caused several problems:

- The LLM sometimes combined multiple quotes in a single marker (`"quote A" and "quote B"`), which required splitting into multiple Citation objects — but they all shared the same text position.
- The frontend tried to map regex matches to citations 1:1, causing chips to stack together instead of appearing inline with their corresponding text.
- A `char_start` backfill mechanism was needed for legacy messages, adding fragile complexity.

**Structured outputs eliminate all of this.** The LLM returns a JSON-validated `StructuredAnswer` with segments and citations already properly associated.

### Data Model

```python
class CitationRef(BaseModel):
    filename: str       # source document filename
    page: int           # page number
    quote: str          # verbatim 10-50 word excerpt

class AnswerSegment(BaseModel):
    text: str                          # one logical part of the answer
    citations: list[CitationRef] = []  # citations for this segment

class StructuredAnswer(BaseModel):
    segments: list[AnswerSegment]      # ordered list of answer parts
```

### LLM Choice

- **Answer generation**: Claude Opus 4.5 (`claude-opus-4-5-20251101`) with `output_type=StructuredAnswer`
- **Title generation**: Claude Haiku 3.5 (cheap, fast, good enough for titles)
- **Citation verification (secondary)**: Claude Haiku 3.5

Opus 4.5 was chosen for the main call because structured output quality directly impacts citation accuracy. The model must simultaneously answer the question, break the answer into logical segments, and produce verbatim quotes — a task that benefits from higher reasoning capability.

### Citation Verification Pipeline

After the structured output call, each citation quote is verified against the source document:

1. **Primary verification** — Normalize both the quote and the page text (lowercase, strip punctuation, collapse whitespace), then check if the normalized quote appears as a substring. Also checks adjacent pages (±1) to handle page boundary issues.

2. **Secondary verification** — For any quotes that fail primary verification, a targeted Haiku call asks "does this text contain information matching the quoted claim?" and either returns a corrected exact quote or marks the citation as unverified.

3. **Frontend display** — Verified citations show as green chips; unverified ones show as amber with a warning banner.

### Storage Format

Citations are stored in the `citations_json` column with a version marker:

```json
{
  "version": 2,
  "segments": [
    {
      "text": "The current rent is £1,750,000 per annum.",
      "citations": [{"index": 1, "filename": "...", "page": 1, "quote": "...", "verified": true}]
    }
  ]
}
```

Old messages (pre-structured-output) stored a flat list `[{index, filename, ...}]`. The backend detects the format and handles both transparently.

### Frontend Rendering

The `MessageBubble` component renders segments sequentially — each segment's text is followed by its inline citation chip(s). Clicking a chip:

1. Switches the PDF viewer to the cited document
2. Navigates to the cited page
3. Highlights the quoted text via DOM manipulation on the `react-pdf` text layer

Highlighting uses whitespace-stripped exact matching: both the PDF text layer spans and the citation quote are normalized (lowercased, punctuation removed, all whitespace stripped) before matching, which handles the fragmented way `react-pdf` renders text into individual `<span>` elements.

---

## Testing

### Integration Tests (`test_real_docs.py`)

Tests run against the live Docker stack using the real-world documents (Lease, Deed of Variation, Rent Review Memorandum).

**TestStructuredCitations** — Validates the structured-output citation pipeline with the question "What is the current rent and who are the current parties?":

- Answer mentions correct rent (£1,750,000), landlord (City of London Real Property Company Limited), and tenant (Stewarts Law LLP)
- Response includes structured segments (not just flat text)
- Every factual segment has at least one citation
- All citations reference the correct source document
- All citations include non-empty verbatim quotes
- All citations are verified against the source text

The LLM response is cached across all test methods — one API call serves all 6 assertions.

---

## Future Improvements

- **Section-aware chunking**: Use document layout analysis to respect clause/section boundaries in legal documents, producing more semantically coherent chunks.
- **HNSW index**: Add a vector index for sub-linear search at scale.
- **Streaming structured output**: PydanticAI supports streaming with `output_type` — segments could appear incrementally as the model generates them.
- **Multi-turn citation context**: Carry citation context across conversation turns so follow-up questions can reference previously cited passages.
- **Citation deduplication**: Merge citations that reference the same passage across different segments.
