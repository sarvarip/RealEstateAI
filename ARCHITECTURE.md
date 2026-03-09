# Architecture Deep Dive

## What Happens When You Ask a Question

This is the end-to-end flow from button click to rendered answer.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  FRONTEND (React)                                                           │
│                                                                             │
│  User types "What is the rent?" → clicks Send                               │
│       │                                                                     │
│       ▼                                                                     │
│  POST /api/conversations/{id}/messages  { content: "What is the rent?" }    │
│       │                                                                     │
│       ▼                                                                     │
│  SSE stream opens ◄──────────────────────────────────────────────┐          │
│       │  events arrive one by one:                               │          │
│       │  ┌─ status: "thinking"                                   │          │
│       │  ├─ status: "searching: current annual rent"             │          │
│       │  ├─ status: "reading page 3 of Lease.pdf"               │          │
│       │  ├─ segments: [{text, citations}, ...]                   │          │
│       │  ├─ message: {id, content, citations, segments}          │          │
│       │  └─ done: {sources_cited: 4}                             │          │
│       │                                                          │          │
│       ▼                                                          │          │
│  MessageBubble renders segments with inline CitationChips        │          │
│  (green = verified, amber = unverified)                          │          │
│                                                                  │          │
└──────────────────────────────────────────────────────────────────┼──────────┘
                                                                   │
┌──────────────────────────────────────────────────────────────────┼──────────┐
│  BACKEND (FastAPI)                                               │          │
│                                                                  │          │
│  send_message() handler                                          │          │
│       │                                                          │          │
│       ├─ Save user message to DB                                 │          │
│       ├─ Load all documents for this conversation                │          │
│       ├─ Load conversation history (prior messages)              │          │
│       │                                                          │          │
│       ▼                                                          │          │
│  ┌─ Token check: total_tokens < 50K? ────────────────────┐      │          │
│  │                                                        │      │          │
│  │  YES: Full-context mode                 NO: RAG mode   │      │          │
│  │  All doc text goes in prompt            Only metadata   │      │          │
│  │  Agent told NOT to search               Agent MUST use  │      │          │
│  │                                         search_documents│      │          │
│  └────────────────┬───────────────────────────────────────┘      │          │
│                   │                                              │          │
│                   ▼                                              │          │
│  answer_with_citations() — Agent.iter() loop                     │          │
│       │                                                          │          │
│       │  ┌──────────────────────────────────────────────┐        │          │
│       │  │  Agent (Claude Opus 4.6) decides:            │        │          │
│       │  │                                              │        │          │
│       │  │  → search_documents("annual rent")           │──── SSE status ──┘
│       │  │    ↳ pgvector cosine search, top 5 chunks    │
│       │  │                                              │
│       │  │  → get_page("Lease.pdf", 3)                  │──── SSE status
│       │  │    ↳ full page text from extracted_text      │
│       │  │                                              │
│       │  │  → produces StructuredAnswer                 │
│       │  │    { segments: [{text, citations}, ...] }    │
│       │  └──────────────────────────────────────────────┘
│       │
│       ▼
│  Citation verification (2-pass)
│       │
│       ├─ Pass 1: Exact substring match
│       │    normalize(quote) ⊂ normalize(page_text)?
│       │    Check page N, N-1, N+1
│       │
│       ├─ Pass 2: LLM fallback (Haiku 3.5)
│       │    For still-unverified citations
│       │    "Does this page contain this claim?"
│       │    → VERIFIED: "<corrected quote>" or UNVERIFIED
│       │
│       ▼
│  Save assistant message to DB (new session)
│  Stream: segments → message → done                        ──── SSE events
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────────────────────┐
│  DATABASE (PostgreSQL + pgvector)                                           │
│                                                                             │
│  conversations ──┬── messages (with citations_json)                         │
│                  └── documents ── document_chunks (with vector embeddings)   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Step by step

1. **User clicks Send** — React calls `POST /api/conversations/{id}/messages` with `{ content: "What is the rent?" }`.

2. **Backend saves the user message** to the `messages` table, loads all documents and conversation history from the DB.

3. **Token check** — estimates total tokens across all document text (`len(text) // 4`). If under 50K tokens (the common case with 3 synthetic docs), uses **full-context mode**: all document text is stuffed into the prompt and the agent is told not to use search tools. Above 50K, uses **RAG mode**: only document filenames/page counts go in the prompt, forcing the agent to use `search_documents()`.

4. **Agent execution** via `Agent.iter()` — the main agent (Claude Opus 4.6) receives the prompt and decides what tools to call. In RAG mode, it typically calls `search_documents()` 1-3 times with different queries, then `get_page()` to verify exact quote wording. Each tool call emits an SSE `status` event so the frontend can show "Searching: current annual rent" etc.

5. **Agent produces `StructuredAnswer`** — a Pydantic model with `segments: [{ text, citations: [{ filename, page, quote }] }]`. Each segment is one logical piece of the answer with its own citations. This is not parsed from free text — it's a structured output guaranteed by PydanticAI.

6. **Citation verification** runs in two passes. Pass 1: exact normalised substring match against the source page text (and adjacent pages for off-by-one errors). Pass 2: for any remaining unverified citations, Haiku 3.5 checks the quote against the source page and either corrects it or marks it unverified.

7. **Save and stream** — the assistant message (with `citations_json` in v2 format) is saved to the DB using a fresh session. The `segments`, `message`, and `done` SSE events are sent to the frontend.

8. **Frontend renders** — `MessageBubble` renders each segment's text (via Streamdown markdown renderer) followed by inline citation chips. Green = verified, amber = unverified. Clicking a chip navigates to the PDF viewer and highlights the quoted text.

9. **Title generation** — on the first message only, Haiku 3.5 generates a 3-5 word conversation title asynchronously.

---

## System Architecture

```
┌──────────┐     ┌──────────────┐     ┌──────────────────┐
│ Frontend │────▶│   Backend    │────▶│    PostgreSQL     │
│ React    │◀────│   FastAPI    │◀────│    + pgvector     │
│ Vite     │ SSE │   uvicorn    │     │                   │
│ port 5173│     │   port 8000  │     │    port 5432      │
└──────────┘     └──────┬───────┘     └──────────────────┘
                        │
                        │ HTTP
                        ▼
              ┌─────────────────────┐
              │   External APIs     │
              │                     │
              │ ┌─────────────────┐ │
              │ │ Anthropic API   │ │  Claude Opus 4.6 (agents)
              │ │                 │ │  Claude Haiku 3.5 (titles, verification, OCR)
              │ └─────────────────┘ │
              │ ┌─────────────────┐ │
              │ │ Azure OpenAI    │ │  text-embedding-3-large (3072 dims)
              │ └─────────────────┘ │
              │ ┌─────────────────┐ │
              │ │ Azure Doc Intel │ │  OCR for scanned PDFs
              │ └─────────────────┘ │
              └─────────────────────┘
```

### Services (docker-compose)

| Service    | Image                     | Port | Purpose                            |
|------------|---------------------------|------|------------------------------------|
| `frontend` | Node 20 + Vite            | 5173 | React SPA, proxies /api to backend |
| `backend`  | Python 3.12 + FastAPI     | 8000 | API, agents, document processing   |
| `db`       | pgvector/pgvector:pg16    | 5432 | PostgreSQL with vector extension   |

---

## Database Schema

```
┌─────────────────────┐
│    conversations     │
├─────────────────────┤
│ id          (PK)    │
│ title               │
│ created_at          │
│ updated_at          │
│                     │
│   ┌─── 1:N ────┐   │
│   ▼             ▼   │
│                     │
├─────────┐ ┌─────────┤
│messages │ │documents│
├─────────┤ ├─────────┤
│ id (PK) │ │ id (PK) │
│ conv_id │ │ conv_id │───── 1:N ────┐
│ role     │ │ filename│              │
│ content  │ │ filepath│              ▼
│ sources_ │ │ extract_│     ┌────────────────┐
│  cited   │ │  text   │     │document_chunks │
│ citations│ │ page_   │     ├────────────────┤
│  _json   │ │  count  │     │ id        (PK) │
│ created_ │ │ uploaded│     │ document_id(FK)│
│  at      │ │  _at    │     │ chunk_index    │
└─────────┘ └─────────┘     │ page_number    │
                              │ content        │
                              │ embedding      │
                              │  (Vector 3072) │
                              │ created_at     │
                              └────────────────┘
```

### Key design decisions

- **`extracted_text` on Document**: Stores the full text of all pages (with `--- Page N ---` markers). This enables `get_page()` tool lookups without touching the filesystem. Tradeoff: large column for big PDFs.
- **`citations_json` on Message**: Stores structured citation data as JSON. Version 2 format preserves segment-citation pairing. Alternatives: a separate `citations` table (more normalized, but loses segment ordering; citations are never queried independently).
- **`embedding` as Vector(3072)**: pgvector column with Azure OpenAI's `text-embedding-3-large` dimensions. Currently uses sequential scan (no index) — fine for small doc sets but needs an HNSW index at scale.
- **UUIDs as hex strings**: `uuid.uuid4().hex[:16]` — 16-char hex IDs. Short enough for URLs, unique enough for a single-tenant app.

---

## Agent Architecture

Five purpose-built agents, each with the minimum tools required for its task:

```
                          ┌─────────────────────────────┐
                          │    Main Agent (Opus 4.6)     │
                          │    output: StructuredAnswer   │
                          │                               │
                          │    Tools:                     │
                          │    ├─ search_documents()      │
User question ──────────▶ │    ├─ get_page()              │
                          │    └─ generate_report()  ─────┼───┐
                          │                               │   │
                          │    Execution: Agent.iter()     │   │
                          │    (early termination when     │   │
                          │     report_result is set)      │   │
                          └───────────────────────────────┘   │
                                                               │
         ┌─────────────────────────────────────────────────────┘
         │  Report pipeline (3 stages)
         ▼
  ┌──────────────┐    ┌──────────────────┐    ┌───────────────────┐
  │ Stage 1       │    │ Stage 2           │    │ Stage 3            │
  │ Summary Agent │──▶ │ Planning Agent    │──▶ │ Programmatic       │
  │ (Opus 4.6)   │    │ (Opus 4.6)       │    │ Search             │
  │               │    │                   │    │                    │
  │ Tools:        │    │ Tools: NONE       │    │ No agent — pure    │
  │ └─ get_page() │    │ (bounded: single  │    │ code. One embed +  │
  │               │    │  LLM call)        │    │ pgvector query per │
  │ Reads first   │    │                   │    │ section.           │
  │ 1-3 pages of  │    │ Output:           │    │                    │
  │ each document │    │ SectionPlan       │    │ Output:            │
  │               │    │ [{title, desc,    │    │ sections_proposal  │
  │ Output:       │    │   search_query}]  │    │ SSE event          │
  │ doc_summary   │    │                   │    │                    │
  └──────────────┘    └──────────────────┘    └───────────────────┘
                                                         │
                                                         ▼
                                                Frontend shows checklist
                                                User selects sections
                                                         │
                                                         ▼
                                          ┌──────────────────────────┐
                                          │ Phase 2: Section Agents  │
                                          │ (Opus 4.6, parallel)     │
                                          │                          │
                                          │ N agents via             │
                                          │ asyncio.gather()         │
                                          │                          │
                                          │ Each has:                │
                                          │ ├─ search_documents()    │
                                          │ └─ get_page()            │
                                          │                          │
                                          │ Each writes ~300 words   │
                                          │ for its assigned section │
                                          └──────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────┐
  │ Auxiliary agents (Haiku 3.5 — cheap, fast)                       │
  │                                                                  │
  │ Title Agent         Verification Agent                           │
  │ └─ 3-5 word title   └─ "VERIFIED: <quote>" or "UNVERIFIED"      │
  │    for sidebar          for citations that failed exact match     │
  └──────────────────────────────────────────────────────────────────┘
```

### Why Agent.iter() instead of Agent.run()

`Agent.run()` blocks until the agent finishes all tool calls and produces its final output. If `generate_comprehensive_report` runs inside `run()`, the agent continues calling `search_documents()` and `get_page()` after the report is already done — wasting tokens and time.

`Agent.iter()` yields execution step by step. After each step, our code checks `deps.report_result is not None`. If the report tool set it, we return immediately — zero additional LLM calls. This is programmatic control over agent execution, not prompt-based ("please stop") which agents ignore.

### Why the planning agent has no tools

An agent with tools will use them unpredictably — it might call `search_documents()` 10 times exploring tangents. By removing all tools, we guarantee exactly one LLM call that returns a structured `SectionPlan`. The search queries in each `PlannedSection` are executed programmatically in Stage 3 (one DB query per section, deterministic).

### AgentDeps: the dependency injection container

```python
@dataclass
class AgentDeps:
    session: AsyncSession          # DB connection for vector search
    conversation_id: str           # Scopes searches to this conversation
    documents: list[Document]      # For get_page() lookups
    user_message: str              # The original question
    on_tool_call: callback | None  # Emits SSE status events
    doc_summary: str | None        # Set by Stage 1, read by Stage 2
    phase: str | None              # Labels SSE events ("planning", etc.)
    report_result: ... | None      # Set by report tool → triggers termination
```

Every tool function receives `ctx: RunContext[AgentDeps]` and accesses these via `ctx.deps`. Each request gets its own `AgentDeps` instance, so concurrent users don't interfere. The mutable fields (`doc_summary`, `phase`, `report_result`) enable communication between tools and the outer execution loop.

---

## Retrieval Strategy

```
                    Total document tokens
                           │
                    ┌──────┴──────┐
                    │  < 50,000   │
                    │  tokens?    │
                    └──────┬──────┘
                     YES   │    NO
                  ┌────────┴────────┐
                  ▼                  ▼
         Full-Context Mode      RAG (Agentic) Mode
                  │                  │
    All doc text in prompt     Only doc metadata
    Agent told NOT to search   Agent MUST use tools
                  │                  │
                  ▼                  ▼
         Single LLM call        Multiple tool calls:
         with full context      search → get_page → ...
                  │                  │
                  └────────┬─────────┘
                           ▼
                  StructuredAnswer
                  {segments, citations}
```

### Why two modes?

**Full-context** is simpler and more reliable — the LLM sees everything, so it can't miss relevant information and citations are more accurate. But it doesn't scale: 50 documents would blow past the context window.

**RAG mode** scales to any number of documents — the agent only retrieves what it needs. But it depends on embedding quality and the agent's search strategy. It might miss relevant information if it doesn't search with the right queries.

The threshold (50K tokens, ~200K characters) automatically picks the best mode. The three synthetic docs are well under this threshold, so they use full-context. A bundle of 12+ real-world leases would trigger RAG mode.

### RAG pipeline detail

```
User query
    │
    ▼
embed_query(query)                    ← Azure OpenAI text-embedding-3-large
    │
    ▼
pgvector cosine distance search       ← SELECT ... ORDER BY embedding <=> query LIMIT 5
    │
    ▼
Top 5 chunks returned to agent        ← XML-formatted: <chunk filename="..." page="..." score="...">
    │
    ▼
Agent may call get_page()             ← Full page text from Document.extracted_text
to verify exact quote wording
```

---

## Document Ingestion Pipeline

```
PDF upload (multipart/form-data)
    │
    ▼
Validate (PDF, < 25MB)
    │
    ▼
Save to uploads/{uuid}_{filename}
    │
    ▼
PyMuPDF extracts text per page
    │
    ├─ Text pages → extracted directly
    │
    ├─ Image pages (scanned) → OCR
    │     │
    │     ├─ Azure Document Intelligence (preferred)
    │     │     └─ prebuilt-read model, whole PDF at once
    │     │
    │     └─ Anthropic Vision fallback (Haiku 3.5)
    │           └─ render page to PNG, send to Claude
    │
    │     OCR results cached: uploads/.ocr_cache/{sha256}.json
    │
    ▼
Merge text + OCR per page → "--- Page N ---\n{text}"
    │
    ▼
Chunk pages (1500 chars, 200 overlap, paragraph boundaries)
    │
    ▼
Embed each chunk → Azure OpenAI text-embedding-3-large (3072 dims)
    │
    ▼
Store Document + DocumentChunks in DB
```

---

## Citation Verification

```
LLM produces citation: {filename, page, quote}
    │
    ▼
Pass 1: Exact Substring Match
    │
    ├─ normalize(quote): lowercase, NFKD, strip punctuation, collapse whitespace
    ├─ normalize(page_text): same
    ├─ Check: norm_quote ⊂ norm_page for page N, N-1, N+1
    │
    ├─ MATCH → verified = true (correct page if off-by-one)
    └─ NO MATCH → continue to Pass 2
                │
                ▼
Pass 2: LLM Verification (Haiku 3.5)
    │
    ├─ Send quote + pages N-1, N, N+1 to Haiku
    ├─ Haiku replies: VERIFIED: "<exact quote>" or UNVERIFIED
    │
    ├─ VERIFIED → verified = true, update quote to exact text
    └─ UNVERIFIED → verified = false, shown as amber chip
```

### Why two passes?

Pass 1 is fast and free (string matching). It catches ~80% of citations. Pass 2 handles the remaining cases where the LLM slightly paraphrased the quote (e.g., changed "shall" to "will") — Haiku can recognise the semantic match and return the actual verbatim text. This costs about $0.001 per citation check.

---

## SSE Event Flow

```
Frontend ──POST──▶ Backend

Backend streams back:
    ┌─ data: {"type":"status","status":"thinking"}
    │
    ├─ data: {"type":"status","status":"searching","query":"annual rent"}
    │    └─ Frontend shows: "Searching: annual rent"
    │
    ├─ data: {"type":"status","status":"reading","filename":"Lease.pdf","page":3}
    │    └─ Frontend shows: "Reading page 3 of Lease.pdf"
    │
    ├─ data: {"type":"segments","segments":[...]}
    │    └─ Frontend renders answer with inline citation chips
    │
    ├─ data: {"type":"message","message":{...}}
    │    └─ Full message object (saved to DB)
    │
    └─ data: {"type":"done","sources_cited":4,"message_id":"abc123"}
         └─ Frontend stops thinking indicator

For reports, also:
    ├─ data: {"type":"sections_proposal","sections":[...],"doc_summary":"..."}
    │    └─ Frontend shows section checklist for user selection
```

---

## Frontend Component Tree

```
App
├── ChatSidebar
│   └── conversation list, create/delete
│
├── ChatWindow
│   ├── MessageBubble (per message)
│   │   ├── SegmentedContent (segments + inline CitationChips)
│   │   │   └── CitationChip (green ✓ or amber ⚠)
│   │   └── FlatContent (fallback for legacy messages)
│   │
│   ├── ThinkingBubble (shown during LLM processing)
│   │   └── toolStatus text ("Searching: ...")
│   │
│   ├── SectionProposal (report checklist)
│   │   └── checkboxes per section, "Generate" button
│   │
│   └── ChatInput
│       └── text input + send button + upload
│
└── DocumentViewer
    └── PDF renderer with page navigation + citation highlighting
```

State is managed via three custom hooks (`useConversations`, `useMessages`, `useDocument`) using React `useState` — no Redux or external state library.

---

## What I'd Improve in Production

| Area | Current | Production |
|------|---------|------------|
| **Vector index** | Sequential scan | HNSW index on embedding column |
| **Chunking** | Paragraph-boundary (1500 chars) | Section-aware (clause/heading detection) |
| **LLM work** | Runs in API handler (blocks connection) | Task queue (Celery/Dramatiq) |
| **Auth** | None | JWT/OAuth, multi-tenant document isolation |
| **Streaming** | Answer arrives all at once | Stream structured output incrementally |
| **Caching** | OCR only | Cache embeddings, frequent queries |
| **Monitoring** | structlog only | Prometheus metrics, LLM cost tracking |
| **DB sessions** | Fresh session for save after LLM | Worker-managed session lifecycle |
| **Document scope** | Per-conversation | Global document store with conversation links |
