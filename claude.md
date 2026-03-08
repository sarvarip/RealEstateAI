# Rules for AI Assistants Working on This Codebase

## Agentic Workflow Rules

- **NEVER use regex to detect user intent.** The agent has tools — let it decide autonomously which tool to call. If the agent needs to be constrained, remove tools or use `Agent.iter()` for programmatic control. Regex-based intent detection defeats the purpose of agentic architecture.
- **Use `Agent.iter()`, not `Agent.run()`**, when you need to terminate an agent after a tool completes. `Agent.run()` blocks until the agent finishes all tool calls — there is no way to interrupt it. `Agent.iter()` yields execution nodes and allows early `return` after checking state.
- **Never try to control agent behavior with prompt instructions alone.** If an agent shouldn't use a tool, remove the tool. If an agent should stop after a tool call, use `Agent.iter()`. Prompts like "do NOT call get_page()" are ignored by agents.
- **Agents without tools for bounded tasks.** If a task needs structured output with no exploration (e.g., planning), use an agent with no tools and a Pydantic `output_type`. This guarantees a single LLM call.
- **Programmatic execution for repetitive bounded tasks.** If you need exactly N searches for N items, write a loop — don't give an agent a search tool and hope it calls it N times.

## Code Standards

- **Add substantial comments to all new code.** Every function, class, and non-trivial block should have comments explaining what it does and why. The codebase owner is not deeply familiar with all implementation details — comments are essential.
- **Always use structured logging (`structlog`).** Log at key decision points: agent completion, tool calls, pipeline stages, error conditions. Include relevant context (counts, IDs, scores).
- **Use logs to diagnose issues, never guess.** When a test fails or the UI reports a problem, the first step is ALWAYS to check `docker compose logs backend --tail N`. Parse the logs before making code changes.

## Debugging & Testing

- **Check logs first.** For any test failure or UI bug, run `docker compose logs backend --tail 50` before changing code.
- **Use synthetic docs for fast iteration.** The `synthetic-docs/` folder has text-based PDFs that need no OCR. Use these for report generation tests.
- **Cache LLM responses in tests.** Use module-level `_cache_*` variables to avoid redundant LLM calls within a test class.
- **Never add `--timeout` flag** — pytest-timeout is not installed. Tests have their own HTTP client timeouts.
- **Run focused tests first** (e.g., `::TestReportProposal`) before the full suite.

## Architecture Quick Reference

- **Main agent**: `agentic_agent` (Opus 4.6) with `search_documents`, `get_page`, `generate_comprehensive_report` tools
- **Report pipeline**: summary agent → planning agent (no tools) → programmatic search → proposal. Controlled via `Agent.iter()` with early termination on `deps.report_result`.
- **Phase 2**: `execute_report_sections()` runs parallel section agents with `search_documents` + `get_page` tools
- **Citation verification**: 2-pass — exact substring match, then Haiku LLM fallback
- **Models**: Opus 4.6 for agents/answers, Haiku 3.5 for titles/verification/OCR
- **Frontend SSE events**: `status` (tool calls with phase), `sections_proposal`, `content`, `message`, `done`

## Key Files

| File | Purpose |
|------|---------|
| `backend/src/takehome/services/llm.py` | All agent definitions, tools, report pipeline, citation verification |
| `backend/src/takehome/web/routers/messages.py` | SSE streaming, Phase 2 dispatch |
| `backend/tests/test_real_docs.py` | Integration tests (32 tests, ~5 min) |
| `frontend/src/hooks/use-messages.ts` | SSE parsing, phase tracking, retry logic |
| `frontend/src/components/SectionProposal.tsx` | Report section checklist UI |
