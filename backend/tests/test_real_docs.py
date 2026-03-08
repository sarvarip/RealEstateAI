"""Integration tests for cross-document Q&A using the real-docs bundle.

These tests validate the core pipeline against the example questions
from the take-home spec. They require Docker services to be running
(docker compose up).

Run with:
    docker compose exec backend uv run pytest backend/tests/test_real_docs.py -v

The real-docs bundle contains mostly scanned PDFs. PyMuPDF alone extracts
very little text from them, but the OCR pipeline (Azure Document Intelligence
or Anthropic vision fallback) recovers the full content at upload time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

BASE_URL = "http://localhost:8000"
REAL_DOCS_DIR = Path("/app/real-docs") if Path("/app/real-docs").exists() else Path("real-docs")

REAL_DOC_FILES = [
    "Lease (06-06-2008).pdf",
    "Official Copy (NGL885533 - Deed - 31-03-2016).pdf",
    "Rent review memorandum - 8th Fl, Building 5, New Street Sq.pdf",
]

Q1 = "What is the rent as at today's date?"
Q2 = "What was the rent on 15/08/2016?"
Q3 = "What rights are granted to the tenant?"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _parse_sse_events(response: httpx.Response) -> list[dict]:
    """Parse all SSE events from a response into a list of dicts."""
    events: list[dict] = []
    for line in response.text.splitlines():
        line = line.strip()
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        try:
            events.append(json.loads(data))
        except json.JSONDecodeError:
            continue
    return events


def _ask_question_full(
    conv_id: str,
    question: str,
    rag_threshold: int | None = None,
    retries: int = 2,
    timeout: int = 300,
) -> dict:
    """Send a question and return the full message event (with segments, citations, etc.).

    Retries on transient failures (missing 'message' event typically means
    the upstream LLM had a connection error).
    """
    msg, _ = _ask_question_with_events(
        conv_id, question, rag_threshold=rag_threshold, retries=retries, timeout=timeout,
    )
    return msg


def _ask_question_with_events(
    conv_id: str,
    question: str,
    rag_threshold: int | None = None,
    retries: int = 2,
    timeout: int = 300,
) -> tuple[dict, list[dict]]:
    """Send a question and return (message_event, all_sse_events)."""
    import time

    params = {}
    if rag_threshold is not None:
        params["rag_threshold"] = rag_threshold

    last_error: Exception | None = None
    for attempt in range(1 + retries):
        if attempt > 0:
            time.sleep(5 * attempt)

        with httpx.Client(base_url=BASE_URL, timeout=timeout) as client:
            resp = client.post(
                f"/api/conversations/{conv_id}/messages",
                json={"content": question},
                params=params,
            )
            resp.raise_for_status()
            events = _parse_sse_events(resp)
            for event in events:
                if event.get("type") == "message":
                    return event["message"], events

        last_error = AssertionError(
            f"No 'message' event in SSE response (attempt {attempt + 1}). "
            f"Events: {[e.get('type') for e in events]}"
        )

    raise last_error  # type: ignore[misc]


def _all_citations(msg: dict) -> list[dict]:
    """Extract a flat list of all citations from a message's segments."""
    return [c for seg in msg.get("segments", []) for c in seg.get("citations", [])]


def _assert_has_segments_and_citations(msg: dict, label: str = "") -> None:
    """Common assertion: message has segments and at least one citation."""
    prefix = f"[{label}] " if label else ""
    segments = msg.get("segments")
    assert segments is not None and len(segments) > 0, (
        f"{prefix}Expected segments in response"
    )
    cites = _all_citations(msg)
    assert len(cites) > 0, f"{prefix}Expected at least one citation"
    for cite in cites:
        assert cite.get("quote") and len(cite["quote"].strip()) > 0, (
            f"{prefix}Citation has empty quote: {cite}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures: conversation creation
# ──────────────────────────────────────────────────────────────────────────────


def _create_conversation_with_docs() -> str:
    """Create a conversation and upload all real-docs."""
    with httpx.Client(base_url=BASE_URL, timeout=600) as client:
        resp = client.post("/api/conversations", json={})
        resp.raise_for_status()
        conv_id = resp.json()["id"]

        for filename in REAL_DOC_FILES:
            filepath = REAL_DOCS_DIR / filename
            if not filepath.exists():
                pytest.skip(f"Real doc not found: {filepath}")
            with open(filepath, "rb") as f:
                resp = client.post(
                    f"/api/conversations/{conv_id}/documents",
                    files={"file": (filename, f, "application/pdf")},
                )
                resp.raise_for_status()

        docs_resp = client.get(f"/api/conversations/{conv_id}/documents")
        docs_resp.raise_for_status()
        docs = docs_resp.json()
        assert len(docs) == 3, f"Expected 3 documents, got {len(docs)}"

    return conv_id


NO_AZURE_DOC_FILES = [
    "Official Copy (NGL885533 - Deed - 31-03-2016).pdf",
    "Rent review memorandum - 8th Fl, Building 5, New Street Sq.pdf",
]


def _create_conversation_no_azure() -> str:
    """Create a conversation with only Deed + Rent Review, no Azure services."""
    with httpx.Client(base_url=BASE_URL, timeout=300) as client:
        resp = client.post("/api/conversations", json={})
        resp.raise_for_status()
        conv_id = resp.json()["id"]

        for filename in NO_AZURE_DOC_FILES:
            filepath = REAL_DOCS_DIR / filename
            if not filepath.exists():
                pytest.skip(f"Real doc not found: {filepath}")
            with open(filepath, "rb") as f:
                resp = client.post(
                    f"/api/conversations/{conv_id}/documents",
                    files={"file": (filename, f, "application/pdf")},
                    params={
                        "skip_embedding": "true",
                        "ocr_provider": "anthropic",
                    },
                )
                resp.raise_for_status()

        docs_resp = client.get(f"/api/conversations/{conv_id}/documents")
        docs_resp.raise_for_status()
        docs = docs_resp.json()
        assert len(docs) == 2, f"Expected 2 documents, got {len(docs)}"

    return conv_id


SYNTHETIC_DOCS_DIR = Path("/app/synthetic-docs") if Path("/app/synthetic-docs").exists() else Path("synthetic-docs")

SYNTHETIC_DOC_FILES = [
    "commercial-lease-100-bishopsgate.pdf",
    "title-report-lot-7.pdf",
]


def _create_synthetic_conversation() -> str:
    """Create a conversation with synthetic (text-based) docs — fast, no OCR needed."""
    with httpx.Client(base_url=BASE_URL, timeout=120) as client:
        resp = client.post("/api/conversations", json={})
        resp.raise_for_status()
        conv_id = resp.json()["id"]

        for filename in SYNTHETIC_DOC_FILES:
            filepath = SYNTHETIC_DOCS_DIR / filename
            if not filepath.exists():
                pytest.skip(f"Synthetic doc not found: {filepath}")
            with open(filepath, "rb") as f:
                resp = client.post(
                    f"/api/conversations/{conv_id}/documents",
                    files={"file": (filename, f, "application/pdf")},
                )
                resp.raise_for_status()

        docs_resp = client.get(f"/api/conversations/{conv_id}/documents")
        docs_resp.raise_for_status()
        docs = docs_resp.json()
        assert len(docs) == 2, f"Expected 2 synthetic documents, got {len(docs)}"

    return conv_id


RENT_MEMO_FILE = "Rent review memorandum - 8th Fl, Building 5, New Street Sq.pdf"


def _create_rent_memo_conversation() -> str:
    """Create a conversation with only the Rent Review Memorandum."""
    with httpx.Client(base_url=BASE_URL, timeout=600) as client:
        resp = client.post("/api/conversations", json={})
        resp.raise_for_status()
        conv_id = resp.json()["id"]

        filepath = REAL_DOCS_DIR / RENT_MEMO_FILE
        if not filepath.exists():
            pytest.skip(f"Real doc not found: {filepath}")
        with open(filepath, "rb") as f:
            resp = client.post(
                f"/api/conversations/{conv_id}/documents",
                files={"file": (RENT_MEMO_FILE, f, "application/pdf")},
            )
            resp.raise_for_status()

    return conv_id


@pytest.fixture(scope="module")
def real_docs_conversation() -> str:
    return _create_conversation_with_docs()


@pytest.fixture(scope="module")
def rag_conversation() -> str:
    return _create_conversation_with_docs()


@pytest.fixture(scope="module")
def no_azure_conversation() -> str:
    return _create_conversation_no_azure()


@pytest.fixture(scope="module")
def synthetic_conversation() -> str:
    return _create_synthetic_conversation()


@pytest.fixture(scope="module")
def rent_memo_conversation() -> str:
    return _create_rent_memo_conversation()


# ──────────────────────────────────────────────────────────────────────────────
# Response caches — one LLM call per question per test suite
# ──────────────────────────────────────────────────────────────────────────────

_cache_full: dict[str, dict] = {}
_cache_rag: dict[str, dict] = {}
_cache_noazure: dict[str, dict] = {}
_cache_structured: dict | None = None


def _get_full(conv_id: str, question: str) -> dict:
    if question not in _cache_full:
        _cache_full[question] = _ask_question_full(conv_id, question)
    return _cache_full[question]


def _get_rag(conv_id: str, question: str) -> dict:
    if question not in _cache_rag:
        _cache_rag[question] = _ask_question_full(conv_id, question, rag_threshold=1000)
    return _cache_rag[question]


def _get_noazure(conv_id: str, question: str) -> dict:
    if question not in _cache_noazure:
        _cache_noazure[question] = _ask_question_full(
            conv_id, question, rag_threshold=9_999_999,
        )
    return _cache_noazure[question]


def _get_structured(conv_id: str) -> dict:
    global _cache_structured
    if _cache_structured is None:
        _cache_structured = _ask_question_full(
            conv_id,
            "What is the current rent and who are the current parties (landlord and tenant)?",
        )
    return _cache_structured


_cache_proposal: tuple[dict, list[dict]] | None = None
_cache_report: tuple[dict, list[dict]] | None = None


def _get_proposal(conv_id: str) -> tuple[dict, list[dict]]:
    """Phase 1: get the report proposal (summary + planning + programmatic search)."""
    global _cache_proposal
    if _cache_proposal is None:
        _cache_proposal = _ask_question_with_events(
            conv_id,
            "Please provide a comprehensive written analysis of these documents "
            "covering the key legal and commercial terms.",
            rag_threshold=1000,
            timeout=180,
        )
    return _cache_proposal


def _get_report(conv_id: str) -> tuple[dict, list[dict]]:
    """Phase 2: execute first 2 proposed sections to generate actual report content."""
    global _cache_report
    if _cache_report is not None:
        return _cache_report

    _, proposal_events = _get_proposal(conv_id)

    proposal_event = next(
        (e for e in proposal_events if e.get("type") == "sections_proposal"),
        None,
    )
    assert proposal_event is not None, (
        f"No sections_proposal SSE event found. Events: {[e.get('type') for e in proposal_events]}"
    )

    sections = proposal_event.get("sections", [])
    doc_summary = proposal_event.get("doc_summary", "")
    assert len(sections) >= 2, f"Expected at least 2 proposed sections, got {len(sections)}"

    selected = sections[:2]

    with httpx.Client(base_url=BASE_URL, timeout=300) as client:
        resp = client.post(
            f"/api/conversations/{conv_id}/messages",
            json={
                "content": "Generate report",
                "report_sections": selected,
                "doc_summary": doc_summary,
            },
            params={"rag_threshold": 1000},
        )
        resp.raise_for_status()
        events = _parse_sse_events(resp)
        for event in events:
            if event.get("type") == "message":
                _cache_report = (event["message"], events)
                return _cache_report

    raise AssertionError(
        f"No 'message' event in Phase 2 SSE. Events: {[e.get('type') for e in events]}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Full-context mode (default threshold — all 3 docs fit in context)
# ──────────────────────────────────────────────────────────────────────────────


class TestFullContext:
    """Tests using full-context mode (all document text sent to LLM)."""

    def test_q1_current_rent(self, real_docs_conversation: str) -> None:
        """Q1: answer mentions £1.75 million."""
        msg = _get_full(real_docs_conversation, Q1)
        assert any(
            t in msg["content"].lower()
            for t in ["1.75 million", "1,750,000", "£1.75m", "1.75m"]
        ), f"Expected rent of £1.75 million not found in answer:\n{msg['content']}"

    def test_q1_citations(self, real_docs_conversation: str) -> None:
        """Q1: has segments with citations referencing the Rent Review Memo."""
        msg = _get_full(real_docs_conversation, Q1)
        _assert_has_segments_and_citations(msg, "FullContext Q1")
        cites = _all_citations(msg)
        rent_cites = [c for c in cites if "rent review" in c["filename"].lower()]
        assert len(rent_cites) > 0, (
            f"Expected at least one citation from Rent Review Memo, got filenames: "
            f"{[c['filename'] for c in cites]}"
        )

    def test_q2_rent_on_date(self, real_docs_conversation: str) -> None:
        """Q2: answer mentions peppercorn."""
        msg = _get_full(real_docs_conversation, Q2)
        assert "peppercorn" in msg["content"].lower(), (
            f"Expected 'peppercorn' in answer:\n{msg['content']}"
        )

    def test_q2_citations(self, real_docs_conversation: str) -> None:
        """Q2: has citations referencing the Deed of Variation."""
        msg = _get_full(real_docs_conversation, Q2)
        _assert_has_segments_and_citations(msg, "FullContext Q2")
        cites = _all_citations(msg)
        deed_cites = [
            c for c in cites
            if "deed" in c["filename"].lower() or "official" in c["filename"].lower()
        ]
        assert len(deed_cites) > 0, (
            f"Expected at least one citation from the Deed, got filenames: "
            f"{[c['filename'] for c in cites]}"
        )

    def test_q3_tenant_rights(self, real_docs_conversation: str) -> None:
        """Q3: answer mentions rights and deed of variation."""
        msg = _get_full(real_docs_conversation, Q3)
        lower = msg["content"].lower()
        assert "right" in lower, f"Expected 'right' in answer:\n{msg['content']}"
        assert any(
            t in lower
            for t in ["deed of variation", "additional", "new right", "varied", "supplemental"]
        ), f"Expected reference to Deed of Variation in answer:\n{msg['content']}"

    def test_q3_citations(self, real_docs_conversation: str) -> None:
        """Q3: has citations (likely from Lease and/or Deed)."""
        msg = _get_full(real_docs_conversation, Q3)
        _assert_has_segments_and_citations(msg, "FullContext Q3")


# ──────────────────────────────────────────────────────────────────────────────
# Structured citations — rent + parties from Rent Review Memo only
# ──────────────────────────────────────────────────────────────────────────────


class TestStructuredCitations:
    """Validate the structured-output pipeline with a single-doc question."""

    def test_rent_and_parties_answer(self, rent_memo_conversation: str) -> None:
        msg = _get_structured(rent_memo_conversation)
        lower = msg["content"].lower()
        assert any(t in lower for t in ["1,750,000", "1.75 million", "1.75m"])
        assert "city of london" in lower or "real property" in lower
        assert "stewarts" in lower

    def test_segments_present(self, rent_memo_conversation: str) -> None:
        msg = _get_structured(rent_memo_conversation)
        segments = msg.get("segments")
        assert segments is not None and len(segments) >= 2

    def test_each_factual_segment_has_citations(self, rent_memo_conversation: str) -> None:
        msg = _get_structured(rent_memo_conversation)
        factual_kw = ["1,750,000", "1.75", "stewarts", "city of london", "real property"]
        for seg in msg.get("segments", []):
            text_lower = seg["text"].lower()
            if any(kw in text_lower for kw in factual_kw):
                assert len(seg.get("citations", [])) > 0, (
                    f"Factual segment missing citation: {seg['text'][:80]}"
                )

    def test_citations_reference_rent_memo(self, rent_memo_conversation: str) -> None:
        msg = _get_structured(rent_memo_conversation)
        for cite in _all_citations(msg):
            assert "rent review" in cite["filename"].lower()

    def test_citations_have_quotes_and_verified(self, rent_memo_conversation: str) -> None:
        msg = _get_structured(rent_memo_conversation)
        cites = _all_citations(msg)
        assert len(cites) > 0
        for cite in cites:
            assert cite.get("quote") and len(cite["quote"].strip()) > 0
            assert cite.get("verified"), f"Citation not verified: {cite}"


# ──────────────────────────────────────────────────────────────────────────────
# RAG mode (forced by setting threshold to 1000 tokens)
# ──────────────────────────────────────────────────────────────────────────────


class TestRAGMode:
    """Tests using RAG mode (forced via low threshold of 1000 tokens)."""

    def test_q1_current_rent_rag(self, rag_conversation: str) -> None:
        msg = _get_rag(rag_conversation, Q1)
        assert any(
            t in msg["content"].lower()
            for t in ["1.75 million", "1,750,000", "£1.75m", "1.75m"]
        ), f"[RAG] Expected rent of £1.75 million:\n{msg['content']}"

    def test_q1_citations_rag(self, rag_conversation: str) -> None:
        msg = _get_rag(rag_conversation, Q1)
        _assert_has_segments_and_citations(msg, "RAG Q1")

    def test_q2_rent_on_date_rag(self, rag_conversation: str) -> None:
        msg = _get_rag(rag_conversation, Q2)
        assert "peppercorn" in msg["content"].lower(), (
            f"[RAG] Expected 'peppercorn':\n{msg['content']}"
        )

    def test_q2_citations_rag(self, rag_conversation: str) -> None:
        msg = _get_rag(rag_conversation, Q2)
        _assert_has_segments_and_citations(msg, "RAG Q2")

    def test_q3_tenant_rights_rag(self, rag_conversation: str) -> None:
        msg = _get_rag(rag_conversation, Q3)
        lower = msg["content"].lower()
        assert "right" in lower
        assert any(
            t in lower
            for t in ["deed of variation", "additional", "new right", "varied", "supplemental"]
        ), f"[RAG] Expected reference to additional/new rights in answer:\n{msg['content']}"

    def test_q3_citations_rag(self, rag_conversation: str) -> None:
        msg = _get_rag(rag_conversation, Q3)
        _assert_has_segments_and_citations(msg, "RAG Q3")


# ──────────────────────────────────────────────────────────────────────────────
# No-Azure mode: Anthropic vision OCR, no embeddings, full-context only
# ──────────────────────────────────────────────────────────────────────────────


class TestNoAzureKeys:
    """Tests with only the Anthropic API key (no Azure)."""

    def test_q1_current_rent_no_azure(self, no_azure_conversation: str) -> None:
        msg = _get_noazure(no_azure_conversation, Q1)
        assert any(
            t in msg["content"].lower()
            for t in ["1.75 million", "1,750,000", "£1.75m", "1.75m"]
        ), f"[NoAzure] Expected rent of £1.75 million:\n{msg['content']}"

    def test_q1_citations_no_azure(self, no_azure_conversation: str) -> None:
        msg = _get_noazure(no_azure_conversation, Q1)
        _assert_has_segments_and_citations(msg, "NoAzure Q1")

    def test_q2_rent_on_date_no_azure(self, no_azure_conversation: str) -> None:
        msg = _get_noazure(no_azure_conversation, Q2)
        assert "peppercorn" in msg["content"].lower(), (
            f"[NoAzure] Expected 'peppercorn':\n{msg['content']}"
        )

    def test_q2_citations_no_azure(self, no_azure_conversation: str) -> None:
        """Q2 answer comes from the scanned Deed — citation quote may come from OCR text."""
        msg = _get_noazure(no_azure_conversation, Q2)
        _assert_has_segments_and_citations(msg, "NoAzure Q2")


# ──────────────────────────────────────────────────────────────────────────────
# Secondary verification — inexact quote triggers fallback LLM correction
# ──────────────────────────────────────────────────────────────────────────────


class TestSecondaryVerification:
    """Test that the secondary LLM verification recovers from inexact quotes.

    We manually construct a Citation with a paraphrased (non-verbatim) quote
    that will fail primary substring verification, then confirm the secondary
    LLM call finds and corrects it.
    """

    FAKE_PAGE_TEXT = (
        "--- Page 1 ---\n"
        "Lease particulars\n"
        "Date: 6 June 2008\n"
        "Original Parties:\n"
        "Landlord: The City of London Real Property Company Limited\n"
        "Tenant: Taylor Wessing LLP\n"
        "Property: Eighth Floor, Building 5, New Street Square, "
        "New Fetter Lane, London EC4\n"
        "Annual rent reserved prior to the review recorded in this memorandum "
        "(exclusive of any VAT): one million three hundred and sixty one thousand "
        "eight hundred and thirty three pounds (£1,361,833)\n"
        "Current landlord: The City of London Real Property Company Limited\n"
        "Current tenant: Stewarts Law LLP\n"
        "The current landlord and the current tenant record that the rent reserved "
        "under the lease has been reviewed in accordance with the lease and agreed "
        "at an annual rent of one million seven hundred and fifty thousand pounds "
        "(£1,750,000) (exclusive of any VAT) with effect from 25 March 2021.\n"
    )

    INEXACT_QUOTE = (
        "the annual rent has been agreed at one million seven hundred "
        "and fifty thousand pounds with effect from March 2021"
    )

    def _make_fake_doc(self, extracted_text: str = "") -> Any:
        from dataclasses import dataclass

        @dataclass
        class FakeDoc:
            id: str = "fake-doc-id"
            filename: str = "Rent review memorandum.pdf"
            extracted_text: str = ""
            page_count: int = 1

        return FakeDoc(extracted_text=extracted_text)

    def test_primary_fails_on_inexact_quote(self) -> None:
        """Primary (substring) verification should FAIL for a paraphrased quote."""
        from takehome.services.llm import Citation, verify_citations

        cite = Citation(
            index=1, filename="Rent review memorandum.pdf", page=1,
            quote=self.INEXACT_QUOTE, document_id="fake-doc-id", verified=False,
        )
        verify_citations([cite], [self._make_fake_doc(self.FAKE_PAGE_TEXT)])
        assert not cite.verified, "Primary should FAIL for paraphrased quote"

    def test_secondary_corrects_inexact_quote(self) -> None:
        """Secondary LLM verification should find matching text and correct the quote."""
        import asyncio

        from takehome.services.llm import Citation, verify_citations, verify_citations_secondary

        cite = Citation(
            index=1, filename="Rent review memorandum.pdf", page=1,
            quote=self.INEXACT_QUOTE, document_id="fake-doc-id", verified=False,
        )
        fake_doc = self._make_fake_doc(self.FAKE_PAGE_TEXT)
        verify_citations([cite], [fake_doc])

        corrected = asyncio.run(verify_citations_secondary([cite], [fake_doc]))
        cite = corrected[0]

        assert cite.verified, "Secondary verification should mark the citation as verified"
        assert cite.quote != self.INEXACT_QUOTE, (
            f"Expected corrected quote, still: {cite.quote}"
        )

    def test_hallucinated_quote_stays_unverified(self) -> None:
        """A completely fabricated quote should remain unverified after both passes."""
        import asyncio

        from takehome.services.llm import Citation, verify_citations, verify_citations_secondary

        hallucinated_quote = (
            "the tenant shall pay a security deposit of five hundred "
            "thousand pounds upon execution of this agreement"
        )

        cite = Citation(
            index=1, filename="Rent review memorandum.pdf", page=1,
            quote=hallucinated_quote, document_id="fake-doc-id", verified=False,
        )
        fake_doc = self._make_fake_doc(self.FAKE_PAGE_TEXT)

        verify_citations([cite], [fake_doc])
        assert not cite.verified, "Primary should FAIL for hallucinated quote"

        corrected = asyncio.run(verify_citations_secondary([cite], [fake_doc]))
        cite = corrected[0]

        assert not cite.verified, (
            "Hallucinated quote should remain UNVERIFIED after both passes — "
            "this would display as an amber warning chip in the frontend"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Report generation — agent uses generate_comprehensive_report + search tools
# Uses synthetic docs (text-based PDFs) for speed.
# ──────────────────────────────────────────────────────────────────────────────

REPORT_SECTIONS = [
    "property",
    "parties",
    "rent",
    "break",
]


class TestReportProposal:
    """Phase 1: agent calls generate_comprehensive_report and returns a proposal.

    Uses synthetic docs (text-based PDFs) for speed. Only tests the proposal
    (summary + planning + programmatic search), not the full section execution.
    """

    def test_agent_called_report_guidelines(self, synthetic_conversation: str) -> None:
        """The agent should autonomously call generate_comprehensive_report."""
        _, events = _get_proposal(synthetic_conversation)
        planning_events = [
            e for e in events
            if e.get("type") == "status" and e.get("status") == "planning"
        ]
        assert len(planning_events) >= 1, (
            f"Expected at least one 'planning' SSE event, "
            f"got events: {[e.get('status') for e in events if e.get('type') == 'status']}"
        )

    def test_agent_used_search_tools(self, synthetic_conversation: str) -> None:
        """Programmatic search should fire search events during proposal."""
        _, events = _get_proposal(synthetic_conversation)
        search_events = [
            e for e in events
            if e.get("type") == "status" and e.get("status") == "searching"
        ]
        assert len(search_events) >= 2, (
            f"Expected at least 2 search events, got {len(search_events)}"
        )

    def test_proposal_has_sections(self, synthetic_conversation: str) -> None:
        """Proposal SSE should contain at least 3 proposed sections."""
        _, events = _get_proposal(synthetic_conversation)
        proposal = next(
            (e for e in events if e.get("type") == "sections_proposal"), None
        )
        assert proposal is not None, "No sections_proposal event in SSE"
        sections = proposal.get("sections", [])
        assert len(sections) >= 3, (
            f"Expected >=3 proposed sections, got {len(sections)}"
        )

    def test_proposal_has_doc_summary(self, synthetic_conversation: str) -> None:
        """Proposal should carry the doc_summary for Phase 2."""
        _, events = _get_proposal(synthetic_conversation)
        proposal = next(
            (e for e in events if e.get("type") == "sections_proposal"), None
        )
        assert proposal is not None, "No sections_proposal event in SSE"
        doc_summary = proposal.get("doc_summary", "")
        assert len(doc_summary) > 20, (
            f"Expected a non-trivial doc_summary, got: {doc_summary!r}"
        )

    def test_proposal_section_headers(self, synthetic_conversation: str) -> None:
        """Proposal message should contain section headers."""
        msg, _ = _get_proposal(synthetic_conversation)
        all_text = " ".join(seg["text"] for seg in msg.get("segments", [])).lower()
        found = [s for s in REPORT_SECTIONS if s in all_text]
        assert len(found) >= 2, (
            f"Expected at least 2 of {REPORT_SECTIONS} in proposal, found: {found}"
        )


class TestReportExecution:
    """Phase 2: selecting sections and executing the report.

    Selects first 2 sections from the proposal, runs them in parallel,
    and validates the generated report content.
    """

    def test_report_has_segments(self, synthetic_conversation: str) -> None:
        """Phase 2 should produce multiple segments (3-5 per section, so >=4 total)."""
        msg, _ = _get_report(synthetic_conversation)
        segments = msg.get("segments", [])
        assert len(segments) >= 4, (
            f"Expected at least 4 segments for 2 sections, got {len(segments)}"
        )

    def test_report_has_citations(self, synthetic_conversation: str) -> None:
        """Phase 2 should have citations grounded in source documents."""
        msg, _ = _get_report(synthetic_conversation)
        _assert_has_segments_and_citations(msg, "Report")
        cites = _all_citations(msg)
        assert len(cites) >= 2, (
            f"Expected at least 2 citations in report, got {len(cites)}"
        )

    def test_report_mentions_key_facts(self, synthetic_conversation: str) -> None:
        """Report should include key facts from the Bishopsgate lease."""
        msg, _ = _get_report(synthetic_conversation)
        lower = msg["content"].lower()
        assert any(
            t in lower for t in ["bishopsgate", "100 bishopsgate", "850,000", "850000", "£850", "meridian"]
        ), f"Expected at least one key fact in report:\n{msg['content'][:500]}"
