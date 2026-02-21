# Approach

## How I Architected the AI System

The system has two main extensions to the baseline: **cross-document RAG** and **citation grounding with structured outputs**.

**Retrieval**: I implemented a tiered retrieval strategy. For small document sets (<100K tokens), all text is sent directly to the LLM — no embedding overhead needed. For larger corpora, the system switches to RAG: documents are chunked (1,500 chars with 200-char overlap at paragraph boundaries), embedded with `text-embedding-3-large` (3,072 dimensions), stored in pgvector, and retrieved via cosine similarity (top-20 chunks). This adaptive approach keeps things simple for the common case while scaling when needed.

**OCR**: Scanned PDFs are handled with a tiered OCR pipeline — Azure Document Intelligence when available, Anthropic vision as a fallback. This means the system works with just an Anthropic API key, no Azure required.

**Citations**: The citation system uses PydanticAI structured outputs (Claude Opus 4.5) to return answer segments paired with citations in a single call. Each segment carries its own `{filename, page, quote}` citations, so the frontend trivially renders each answer part followed by its inline citation chips. Quotes are verified against source text — first via exact substring matching (normalised for case/punctuation), then via a secondary Haiku LLM call for any misses. Unverified citations are flagged visually.

## What I Prioritised

I focused on **citation correctness** above all — in legal due diligence, a wrong or fabricated citation is worse than no citation. This led to several design choices:

- **Structured outputs over regex parsing**: The initial regex-based approach was fragile (combined quotes, misaligned chips). Structured outputs guarantee each answer segment is paired with its citations by construction.
- **Two-pass verification**: Exact text matching catches most quotes; the secondary LLM call recovers near-misses (e.g., slight paraphrasing) and flags genuine hallucinations.
- **Opus 4.5 for answer generation**: Higher reasoning capability produces better structured output quality. Haiku is used for cheap auxiliary tasks (titles, verification).

I also prioritised **graceful degradation** — the app works with only an Anthropic API key (no Azure), just without embeddings or premium OCR.

## What I'd Do Next

- **Section-aware chunking**: Legal documents have natural structure (clauses, schedules, recitals). Using document layout analysis to chunk at section boundaries would produce more semantically coherent retrieval units.
- **HNSW index**: Currently vector search uses sequential scan — fine for a few documents, but an HNSW index is needed at scale.
- **Streaming structured output**: PydanticAI supports streaming with `output_type`, so segments could appear incrementally rather than all-at-once.
- **Multi-turn citation context**: Carry cited passages across conversation turns so follow-up questions can reference earlier citations.

## Interesting Problems

**The citation alignment problem**: The trickiest issue was getting citation chips to appear inline next to their corresponding text. The initial approach embedded `[Doc: file | Page N | "quote"]` markers in the LLM's free-text output and parsed them with regex. This broke when the LLM combined multiple quotes in one marker (`"quote A" and "quote B"`), producing multiple Citation objects that all mapped to the same text position — so all chips stacked together after one paragraph. I tried increasingly complex fixes (char_start tracking, backfill for old messages, frontend grouping logic) before realising the root cause was the architecture: parsing citations from free text is fundamentally fragile. Switching to structured outputs solved it cleanly — the LLM returns segments with citations already properly associated, no parsing needed.

**PDF highlight matching**: `react-pdf` renders text into fragmented `<span>` elements that don't correspond to logical words or sentences. Matching a citation quote against these spans required stripping all whitespace from both sides before matching, then applying highlights to the correct span range via DOM manipulation.
