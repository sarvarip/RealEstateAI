# Approach

## How I Architected the AI System

The system has four extensions to the baseline: **cross-document RAG**, **citation grounding with structured outputs**, **visual document understanding** (OCR for scanned PDFs), and **structured report generation** via a multi-agent pipeline.

**Retrieval**: I implemented a tiered retrieval strategy. For small document sets (<50K tokens), all text is sent directly to the LLM — no embedding overhead needed. For larger corpora, the system switches to RAG: documents are chunked (1,500 chars with 200-char overlap at paragraph boundaries), embedded with `text-embedding-3-large` (3,072 dimensions), stored in pgvector, and retrieved via cosine similarity (top-5 chunks per search call). The agent may call `search_documents()` multiple times with different queries to build comprehensive context. This adaptive approach keeps things simple for the common case while scaling when needed.

**Visual document understanding**: Scanned PDFs are handled with a tiered OCR pipeline — Azure Document Intelligence when available, Anthropic vision (Haiku 3.5) as a fallback, with results cached on disk by content hash. This means the system works with just an Anthropic API key, no Azure required. Pages with images are detected via PyMuPDF; for those pages, OCR output replaces PyMuPDF's text layer entirely (avoiding duplicate chunks in RAG). PyMuPDF text is kept as fallback only if OCR fails.

**Agentic architecture**: The system uses PydanticAI agents (Claude Opus 4.6) with `search_documents` and `get_page` tools. The agent autonomously decides how to research a question. The agent execution loop uses `Agent.iter()` for programmatic control — allowing immediate termination after tool completion (e.g., when the report proposal is ready).

**Report generation**: A multi-agent pipeline produces structured reports (e.g., Report on Title). Phase 1 (proposal): a summary agent reads the first pages of each document, a planning agent (no tools, structured output) proposes sections with search queries, then programmatic RAG runs one search per section — bounded and deterministic. The frontend presents an interactive checklist. Phase 2 (execution): parallel section agents research and write each selected section with citations, using `search_documents` and `get_page`.

**Citations**: The citation system uses PydanticAI structured outputs to return answer segments paired with citations in a single call. Each segment carries its own `{filename, page, quote}` citations, so the frontend renders each answer part followed by its inline citation chips. Quotes are verified against source text — first via exact substring matching (normalised for case/punctuation), then via a secondary Haiku LLM call for any misses. Unverified citations are flagged visually.

## What I Prioritised

I focused on **citation correctness** above all — in legal due diligence, a wrong or fabricated citation is worse than no citation. This led to several design choices:

- **Structured outputs over regex parsing**: The initial regex-based approach was fragile (combined quotes, misaligned chips). Structured outputs guarantee each answer segment is paired with its citations by construction.
- **Two-pass verification**: Exact text matching catches most quotes; the secondary LLM call recovers near-misses (e.g., slight paraphrasing) and flags genuine hallucinations.
- **Opus 4.6 for answer generation**: Higher reasoning capability produces better structured output quality. Haiku is used for cheap auxiliary tasks (titles, verification).

I also prioritised **agentic engineering** — building the report generation pipeline revealed important lessons about controlling agent behavior:

- **`Agent.iter()` over `Agent.run()`**: `Agent.run()` doesn't allow early termination — after a tool returns, the agent keeps calling more tools, wasting tokens. `Agent.iter()` gives step-by-step control, enabling immediate termination when the report pipeline completes.
- **Remove tools to constrain agents**: Prompt instructions like "do NOT call get_page()" were consistently ignored. The only reliable way to bound an agent's behavior is to remove tools it shouldn't use. The planning agent has no tools by design.
- **Programmatic execution for bounded tasks**: If you need exactly N searches for N sections, write a loop — don't give an agent a search tool and hope it calls it N times.

Finally, **graceful degradation** — the app works with only an Anthropic API key (no Azure), just without embeddings or premium OCR.

## What I'd Do Next

- **Section-aware chunking**: Legal documents have natural structure (clauses, schedules, recitals). Using document layout analysis to chunk at section boundaries would produce more semantically coherent retrieval units.
- **HNSW index**: Currently vector search uses sequential scan — fine for a few documents, but an HNSW index is needed at scale.
- **Streaming structured output**: PydanticAI supports streaming with `output_type`, so segments could appear incrementally rather than all-at-once.
- **Multi-turn citation context**: Carry cited passages across conversation turns so follow-up questions can reference earlier citations.
- **Portfolio analysis**: Cross-conversation agents that can compare properties and extract structured data across multiple bundles simultaneously.
- **Better PDF highlighting**: The current approach uses imperative DOM manipulation (`span.style.backgroundColor`) to highlight quoted text in `react-pdf`'s text layer, which works outside React's declarative rendering model. A cleaner solution would use a custom `react-pdf` text layer renderer or a canvas-based overlay, avoiding direct DOM access.

## Interesting Problems

**The citation alignment problem**: The trickiest issue was getting citation chips to appear inline next to their corresponding text. The initial approach embedded `[Doc: file | Page N | "quote"]` markers in the LLM's free-text output and parsed them with regex. This broke when the LLM combined multiple quotes in one marker (`"quote A" and "quote B"`), producing multiple Citation objects that all mapped to the same text position — so all chips stacked together after one paragraph. I tried increasingly complex fixes (char_start tracking, backfill for old messages, frontend grouping logic) before realising the root cause was the architecture: parsing citations from free text is fundamentally fragile. Switching to structured outputs solved it cleanly — the LLM returns segments with citations already properly associated, no parsing needed.

**Agent execution control**: When building the report generation pipeline, the `generate_comprehensive_report` tool would complete and store its result, but the main agent kept calling more tools — wasting tokens. Prompt-based instructions ("do NOT call any more tools") were consistently ignored. The fix was replacing `Agent.run()` with `Agent.iter()`, which yields execution nodes one at a time. After each step, the code checks if the report result is ready and returns immediately if so — zero wasted tokens, fully programmatic control.

**PDF highlight matching**: `react-pdf` renders PDF text as an overlay of `<span role="presentation">` elements on top of the page canvas. These spans don't correspond to words or sentences — their boundaries are dictated by the PDF's internal text encoding, so a single word can be split across multiple spans. To highlight a citation quote, the system concatenates all span text with whitespace stripped, finds the quote (also whitespace-stripped) via substring match, then maps the character positions back to the original spans. Because `react-pdf` has no highlighting API, the matching spans' `style.backgroundColor` is set directly via imperative DOM access — outside React's declarative rendering flow. This is a pragmatic but suboptimal approach: it bypasses React's virtual DOM, can cause stale highlights on re-render, and wouldn't scale to complex highlighting scenarios (overlapping quotes, multi-page spans). For scanned documents where the page is an image with no text layer, the system falls back to a highlight banner showing the quoted text.
