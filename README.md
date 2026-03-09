# Document Q&A for Commercial Real Estate

An AI-powered document assistant for commercial real estate lawyers. Upload legal document bundles (leases, deeds, rent review memoranda) and ask questions that require cross-referencing information across multiple documents. Every answer is grounded with clickable, verified citations that navigate directly to the source page and highlight the exact quoted text.

Built with FastAPI, React, PostgreSQL + pgvector, PydanticAI (Claude Opus 4.6), and Azure/Anthropic OCR.

---

## Features

### Cross-Document Analysis (Feature 1)

Lawyers working on property acquisitions need to reconcile information spread across many documents. The system supports multi-document conversations where questions are answered by synthesising content from the full bundle.

**Tiered retrieval strategy:**

- **Full-context mode** (default): When the total document text fits within the token threshold (~50K tokens), all text is sent directly to the LLM. This is the common case for small bundles (2-5 documents) and produces the best answers since the model sees everything.
- **RAG mode**: For larger corpora, the system automatically switches to retrieval-augmented generation. Documents are chunked at paragraph boundaries (1,500 chars, 200-char overlap), embedded with Azure OpenAI `text-embedding-3-large` (3,072 dimensions), stored in pgvector, and the top-5 most relevant chunks are retrieved via cosine similarity per search call. The agent may call `search_documents()` multiple times with different queries to gather comprehensive information.

The threshold is configurable and the switch is transparent — the same question API works in both modes.

### Citation & Grounding (Feature 2)

A lawyer can't trust an AI answer they can't verify. The citation system ensures every factual claim is traceable to an exact location in the source documents.

**Agentic architecture with structured outputs:**

The system uses PydanticAI agents with `search_documents` and `get_page` tools. The agent autonomously decides how to research a question — searching for relevant chunks, reading specific pages to verify quotes, and cross-referencing across documents. Answers are returned as `StructuredAnswer` objects containing `AnswerSegment` objects, each with its own `text` and `citations` list. Each citation carries `{filename, page, quote}`. This eliminates regex parsing entirely — the segment-citation pairing is guaranteed by construction.

**Two-pass quote verification:**

1. **Primary verification** (exact match): The citation's quote is normalised (lowercased, Unicode NFKD, punctuation stripped, whitespace collapsed) and checked as a substring against the source document's page text. Adjacent pages (page-1, page+1) are also checked to handle off-by-one page references. This is essentially a programmatic CTRL-F.

2. **Secondary verification** (LLM fallback): If the exact match fails (e.g., the LLM slightly paraphrased), a targeted Haiku call receives the quote and the surrounding page text, and is asked to find the matching content. If found, it returns the corrected verbatim quote. If not found, the citation is flagged as unverified.

**Frontend citation experience:**

- Inline green citation chips appear after each answer segment, showing the document name and page number
- Unverified citations appear as amber chips with a warning icon
- Clicking a citation chip navigates the PDF viewer to the correct document and page
- For text-based PDFs: the exact quoted text is highlighted in yellow via DOM manipulation of `react-pdf`'s text layer spans (whitespace-stripped matching handles fragmented `<span>` elements). Scroll position is set via direct `scrollTop` manipulation with a self-correcting enforcer that resists layout shifts from react-pdf's layered rendering.
- For scanned PDFs (image pages): a highlight banner displays the quoted text since there's no selectable text layer to highlight
- Multi-document navigation: citation chips from different documents switch the active document automatically. Documents are listed in a dedicated sidebar section with a draggable divider separating chats and documents.

### Visual Document Understanding (Extension)

Many legal PDFs are scanned images with no text layer. The system handles these transparently:

**Tiered OCR pipeline:**

1. **Azure Document Intelligence** (`prebuilt-read`): When Azure keys are configured, the entire PDF is sent once to Azure DI, which returns structured text per page. Best accuracy, handles complex layouts.
2. **Anthropic Vision** (Claude Haiku 3.5): Fallback when Azure is unavailable. Each image page is rendered to PNG at 150 DPI via PyMuPDF, then sent to Haiku's vision API with a structured extraction prompt. Pages are processed concurrently (semaphore-limited to 10).
3. **PyMuPDF text layer**: Always runs first. For pages where OCR returns text, the OCR result replaces the PyMuPDF text entirely (avoiding duplicate content). PyMuPDF text is used only as a fallback when OCR isn't needed or unavailable.

**OCR caching:** Results are cached on disk keyed by `SHA-256(file_contents + provider)`. Re-uploading the same PDF (or restarting the server) skips OCR entirely.

**Graceful degradation:** The app works with just an Anthropic API key — no Azure required. Without Azure, you lose embeddings (RAG falls back to full-context) and premium OCR (Anthropic vision handles scanned pages instead). This is useful for development and for environments with limited API access.

### Structured Report Generation (Extension)

The system can generate structured reports (e.g., Report on Title) through a multi-agent pipeline:

**Phase 1 — Proposal (automatic):**
1. **Summary agent** reads the first 1-3 pages of each document to understand the bundle
2. **Planning agent** (no tools, structured output) proposes 5-10 sections with search queries
3. **Programmatic search** executes exactly one RAG query per section — bounded, deterministic
4. Frontend displays an interactive checklist of proposed sections with preliminary citations

**Dynamic plan modification:** After the initial proposal, the user can ask to add, remove, or change sections (e.g., "add a section about indemnity provisions"). The agent calls `modify_plan`, which reloads the persisted `doc_summary` and section plan from the database, re-runs the planning agent with the user's feedback, and returns an updated proposal. The section selection widget persists across chat switches — leaving and returning to a conversation restores the checklist.

**Phase 2 — Execution (user-confirmed):**
1. User selects which sections to expand
2. Parallel **section agents** (each with search + page-read tools) write ~300 words per section
3. All citations verified via the same two-pass pipeline
4. Combined report returned with section headers and inline citations

**Agent execution control:** The main agent's execution loop uses `Agent.iter()` (not `Agent.run()`), allowing immediate programmatic termination after the report tool completes — preventing wasted tokens from post-completion tool calls.

---

## Architecture

```
Frontend (React + Vite)         Backend (FastAPI)              Storage
┌─────────────────────┐   SSE   ┌──────────────────────┐   ┌───────────────┐
│ Chat UI             │◄───────►│ Messages Router      │   │ PostgreSQL 16 │
│ PDF Viewer          │         │  ├─ SSE streaming    │──►│  messages     │
│ Citation Chips      │         │  └─ Phase 2 dispatch │   │  documents    │
│ Text Highlighting   │         ├──────────────────────┤   │  chunks       │
│ Section Checklist   │         │ Main Agent (Opus 4.6)│   │  (pgvector)   │
└─────────────────────┘         │  ├─ search_documents │   └───────────────┘
                                │  ├─ get_page         │
                                │  ├─ generate_report  │
                                │  └─ modify_plan      │   ┌───────────────┐
                                ├──────────────────────┤──►│ Azure OpenAI  │
                                │ Report Pipeline      │   │  (embeddings) │
                                │  ├─ Summary Agent    │   └───────────────┘
                                │  ├─ Planning Agent   │
                                │  ├─ Programmatic RAG │   ┌───────────────┐
                                │  └─ Section Agents   │──►│ Anthropic API │
                                ├──────────────────────┤   └───────────────┘
                                │ Document Router      │
                                │  ├─ extract/OCR/chunk│
                                │  └─ embed & store    │
                                └──────────────────────┘
```

**Key design decisions:**

- **Agentic tool-calling**: PydanticAI `Agent` with `search_documents` and `get_page` tools. The agent autonomously decides when to search, what to read, and how to cross-reference — rather than following a fixed pipeline.
- **`Agent.iter()` for execution control**: The agent loop is driven by `Agent.iter()` instead of `Agent.run()`, allowing programmatic termination after tool completion (e.g., report generation). This prevents wasted tokens from post-completion tool calls.
- **Opus 4.6 for answer generation**: Higher reasoning quality produces better structured outputs and more accurate citations. The cost is acceptable since answers are generated once per question.
- **Haiku 3.5 for auxiliary tasks**: Title generation, secondary citation verification, and vision OCR all use the cheaper, faster Haiku model.
- **PydanticAI structured outputs**: The `StructuredAnswer` Pydantic model is passed as `output_type` to the Agent, so the LLM is constrained to return valid JSON matching the schema. No regex parsing needed.
- **Server-Sent Events (SSE)**: The backend streams events (`status` → `sections_proposal` → `content` → `message` → `done`) so the frontend can show real-time tool execution status and thinking indicators. SSE streams run to completion even when the user switches chats — UI updates are silently discarded for non-active conversations, and a "generating in background" indicator shows when switching back. This avoids aborting the backend mid-generation while keeping the UI responsive.
- **Non-blocking DB sessions**: Agent tools use short-lived sessions from an `async_sessionmaker` factory, releasing DB connections during long LLM calls. Combined with an async Azure OpenAI embedding client (`AsyncAzureOpenAI`), multiple chat sessions can run agents concurrently without blocking each other.
- **Multi-document upload**: The frontend supports selecting or drag-and-dropping multiple PDFs at once, uploading them sequentially to the existing single-file endpoint.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, TypeScript, Vite 6, Tailwind CSS, Radix UI, react-pdf, Streamdown, Framer Motion |
| Backend | Python 3.12, FastAPI, SQLAlchemy (async), Alembic, PydanticAI, structlog |
| Database | PostgreSQL 16, pgvector |
| AI Models | Claude Opus 4.6 (agents/answers), Claude Haiku 3.5 (titles, verification, vision OCR) |
| Embeddings | Azure OpenAI `text-embedding-3-large` (3,072 dimensions) |
| OCR | Azure Document Intelligence / Anthropic Vision (fallback) |
| Infrastructure | Docker Compose, uv (Python package manager) |

---

## Setup

### Prerequisites
- Docker and Docker Compose
- `just` (command runner) — install via `brew install just` or `cargo install just`

### Getting Started

1. Clone this repository

2. Run the setup command:
```bash
just setup
```

3. Add your API keys to `.env`:
```bash
ANTHROPIC_API_KEY=your_key_here

# Optional — enables embeddings (RAG) and premium OCR:
AZURE_OPENAI_ENDPOINT=https://...
AZURE_OPENAI_API_KEY=...
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://...
AZURE_DOCUMENT_INTELLIGENCE_API_KEY=...
```

4. Start everything:
```bash
just dev
```
This starts PostgreSQL, the FastAPI backend (port 8000), and the React frontend (port 5173). Database migrations run automatically.

5. Open http://localhost:5173

Local `backend/src/` and `frontend/src/` directories are mounted into the containers — edit files normally and changes hot-reload.

### Sample Documents

- `synthetic-docs/` — Programmatically generated legal documents (clean text PDFs). Good for basic testing.
- `real-docs/` — Real-world legal documents including scanned pages and title plans. Tests the full pipeline including OCR and cross-document analysis.

---

## Testing

The test suite validates the full pipeline end-to-end against the `real-docs/` bundle:

```bash
docker compose exec backend uv run pytest backend/tests/test_real_docs.py -v
```

**Test classes:**

| Class | Tests | What it validates |
|-------|-------|-------------------|
| `TestFullContext` | 6 | All 3 docs in context — answer correctness + citation presence for each question |
| `TestStructuredCitations` | 5 | Single-doc structured output — segments, factual citations, quote verification |
| `TestRAGMode` | 6 | Forced RAG (threshold=1000) — same questions, verifies RAG retrieves correct chunks |
| `TestNoAzureKeys` | 4 | Anthropic-only pipeline — OCR via vision, full-context mode, citations still work |
| `TestSecondaryVerification` | 2 | Unit test — paraphrased quote fails primary check, secondary LLM corrects it |
| `TestReportProposal` | 5 | Phase 1 report: summary + planning + programmatic search (forced RAG, threshold=1000) |
| `TestReportExecution` | 3 | Phase 2 report: select 2 sections, parallel agent generation, citations (forced RAG) |
| `TestReportModification` | 1 | Dynamic plan modification: initial proposal → user asks to add a section → verify updated plan |

Tests use per-question response caching to avoid redundant LLM calls within a test module, and retry logic for transient API errors.

---

## Project Structure

```
backend/src/takehome/
├── config.py                 # Settings (thresholds, model config, API keys)
├── db/
│   ├── models.py             # SQLAlchemy models (Conversation, Message, Document, DocumentChunk)
│   └── session.py            # Async database session
├── services/
│   ├── llm.py                # Agentic pipeline, answer generation, report generation, citation verification
│   ├── chunking.py           # Document chunking (paragraph-boundary splits with overlap)
│   ├── embedding.py          # Azure OpenAI embedding (sync batch + async single query)
│   ├── ocr.py                # Tiered OCR (Azure DI / Anthropic vision) with disk caching
│   ├── document.py           # Upload pipeline: extract → OCR → chunk → embed → store
│   └── conversation.py       # Conversation CRUD
└── web/
    ├── app.py                # FastAPI app setup
    └── routers/
        ├── messages.py       # SSE message endpoint, structured citation serialisation
        ├── documents.py      # Document upload/download endpoints
        └── conversations.py  # Conversation list/create/delete

frontend/src/
├── types.ts                  # TypeScript interfaces (Message, Citation, AnswerSegment)
├── hooks/use-messages.ts     # SSE event parsing, thinking state, segment merging, proposal persistence
├── hooks/use-document.ts     # Document management, multi-file upload
├── components/
│   ├── MessageBubble.tsx     # Segmented content rendering, citation chips, thinking animation
│   ├── DocumentViewer.tsx    # PDF viewer with text highlighting and multi-doc tabs
│   ├── ChatWindow.tsx        # Message list with auto-scroll
│   ├── ChatInput.tsx         # Message input with upload button
│   ├── ChatSidebar.tsx       # Conversation list + document list with draggable divider
│   └── DocumentUpload.tsx    # Multi-file upload dialog (drag-and-drop or file picker)
└── App.tsx                   # Root layout, state management, citation click handling
```

---

## Useful Commands

| Command | Description |
|---------|-------------|
| `just dev` | Start full stack (Postgres + backend + frontend) |
| `just stop` | Stop all services |
| `just reset` | Stop everything and clear database |
| `just check` | Run all linters and type checks |
| `just fmt` | Format all code |
| `just db-shell` | Open a psql shell |
| `just logs-backend` | Tail backend logs |
