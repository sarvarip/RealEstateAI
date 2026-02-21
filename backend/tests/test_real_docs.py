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

import httpx
import pytest

BASE_URL = "http://localhost:8000"
REAL_DOCS_DIR = Path("/app/real-docs") if Path("/app/real-docs").exists() else Path("real-docs")

REAL_DOC_FILES = [
    "Lease (06-06-2008).pdf",
    "Official Copy (NGL885533 - Deed - 31-03-2016).pdf",
    "Rent review memorandum - 8th Fl, Building 5, New Street Sq.pdf",
]


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


def _parse_sse_response(response: httpx.Response) -> str:
    """Extract the final assistant message content from an SSE stream."""
    for event in _parse_sse_events(response):
        if event.get("type") == "message":
            return event["message"]["content"]
        if event.get("type") == "content":
            return event.get("content", "")
    return ""


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


@pytest.fixture(scope="module")
def real_docs_conversation() -> str:
    """Shared conversation for full-context tests."""
    return _create_conversation_with_docs()


@pytest.fixture(scope="module")
def rag_conversation() -> str:
    """Separate conversation for RAG-mode tests (clean history)."""
    return _create_conversation_with_docs()


def _ask_question(conv_id: str, question: str, rag_threshold: int | None = None) -> str:
    """Send a question and return the assistant's response text."""
    params = {}
    if rag_threshold is not None:
        params["rag_threshold"] = rag_threshold

    with httpx.Client(base_url=BASE_URL, timeout=300) as client:
        resp = client.post(
            f"/api/conversations/{conv_id}/messages",
            json={"content": question},
            params=params,
        )
        resp.raise_for_status()
        return _parse_sse_response(resp)


def _ask_question_full(conv_id: str, question: str, rag_threshold: int | None = None) -> dict:
    """Send a question and return the full message event (with segments, citations, etc.)."""
    params = {}
    if rag_threshold is not None:
        params["rag_threshold"] = rag_threshold

    with httpx.Client(base_url=BASE_URL, timeout=300) as client:
        resp = client.post(
            f"/api/conversations/{conv_id}/messages",
            json={"content": question},
            params=params,
        )
        resp.raise_for_status()
        events = _parse_sse_events(resp)
        for event in events:
            if event.get("type") == "message":
                return event["message"]
        raise AssertionError(f"No 'message' event found in SSE response. Events: {[e.get('type') for e in events]}")


# ──────────────────────────────────────────────────────────────────────────────
# Full-context mode (default threshold — all 3 docs fit in context)
# ──────────────────────────────────────────────────────────────────────────────


class TestFullContext:
    """Tests using full-context mode (all document text sent to LLM)."""

    def test_q1_current_rent(self, real_docs_conversation: str) -> None:
        """Q1: What is the rent as at today's date?
        Expected: £1.75 million per the Rent Review Memorandum.
        """
        answer = _ask_question(
            real_docs_conversation,
            "What is the rent as at today's date?",
        )
        answer_lower = answer.lower()
        assert any(
            term in answer_lower
            for term in ["1.75 million", "1,750,000", "£1.75m", "1.75m", "1,750,000"]
        ), f"Expected rent of £1.75 million not found in answer:\n{answer}"

    def test_q2_rent_on_date(self, real_docs_conversation: str) -> None:
        """Q2: What was the rent on 15/08/2016?
        Expected: a peppercorn per the Deed of Variation.
        Depends on text from the scanned Deed of Variation.
        """
        answer = _ask_question(
            real_docs_conversation,
            "What was the rent on 15/08/2016?",
        )
        answer_lower = answer.lower()
        assert "peppercorn" in answer_lower, (
            f"Expected 'peppercorn' in answer:\n{answer}"
        )

    def test_q3_tenant_rights(self, real_docs_conversation: str) -> None:
        """Q3: What rights are granted to the tenant?
        Expected: mentions rights from the lease and new/additional rights
        from the Deed of Variation. Depends on scanned document content.
        """
        answer = _ask_question(
            real_docs_conversation,
            "What rights are granted to the tenant?",
        )
        answer_lower = answer.lower()
        has_rights = "right" in answer_lower
        has_variation_ref = any(
            term in answer_lower
            for term in ["deed of variation", "additional", "new right", "varied", "supplemental"]
        )
        assert has_rights, f"Expected discussion of rights in answer:\n{answer}"
        assert has_variation_ref, (
            f"Expected reference to Deed of Variation or additional/new rights in answer:\n{answer}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Structured citations — rent + parties from Rent Review Memo only
# ──────────────────────────────────────────────────────────────────────────────


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
def rent_memo_conversation() -> str:
    """Conversation with only the Rent Review Memorandum uploaded."""
    return _create_rent_memo_conversation()


_STRUCTURED_QUESTION = "What is the current rent and who are the current parties (landlord and tenant)?"

_cached_structured_response: dict | None = None


def _get_structured_response(conv_id: str) -> dict:
    """Ask the structured-citations question once and cache the result for all tests."""
    global _cached_structured_response
    if _cached_structured_response is None:
        _cached_structured_response = _ask_question_full(conv_id, _STRUCTURED_QUESTION)
    return _cached_structured_response


class TestStructuredCitations:
    """Validate that the structured-output pipeline returns correct segments and citations."""

    def test_rent_and_parties_answer(self, rent_memo_conversation: str) -> None:
        """The answer text must mention rent, landlord, and tenant."""
        msg = _get_structured_response(rent_memo_conversation)

        answer_lower = msg["content"].lower()

        assert any(
            t in answer_lower for t in ["1,750,000", "1.75 million", "1.75m"]
        ), f"Expected rent amount in answer:\n{msg['content']}"

        assert "city of london" in answer_lower or "real property" in answer_lower, (
            f"Expected landlord name in answer:\n{msg['content']}"
        )

        assert "stewarts" in answer_lower, (
            f"Expected tenant name (Stewarts Law) in answer:\n{msg['content']}"
        )

    def test_segments_present(self, rent_memo_conversation: str) -> None:
        """The response must include structured segments (not just flat content)."""
        msg = _get_structured_response(rent_memo_conversation)

        segments = msg.get("segments")
        assert segments is not None, "Expected 'segments' in message response"
        assert len(segments) >= 2, (
            f"Expected at least 2 segments (rent + parties), got {len(segments)}"
        )

    def test_each_segment_has_citations(self, rent_memo_conversation: str) -> None:
        """Every segment containing a specific factual claim should have at least one citation."""
        msg = _get_structured_response(rent_memo_conversation)

        segments = msg.get("segments", [])
        factual_keywords = ["1,750,000", "1.75", "stewarts", "city of london", "real property"]
        factual_without_cite = []
        for seg in segments:
            text_lower = seg["text"].lower()
            has_factual_claim = any(kw in text_lower for kw in factual_keywords)
            if has_factual_claim and len(seg.get("citations", [])) == 0:
                factual_without_cite.append(seg["text"])

        assert not factual_without_cite, (
            f"Factual segments missing citations:\n" +
            "\n".join(f"  - {t}" for t in factual_without_cite)
        )

    def test_citations_reference_rent_memo(self, rent_memo_conversation: str) -> None:
        """All citations should reference the Rent Review Memorandum."""
        msg = _get_structured_response(rent_memo_conversation)

        segments = msg.get("segments", [])
        for seg in segments:
            for cite in seg.get("citations", []):
                assert "rent review" in cite["filename"].lower(), (
                    f"Citation references wrong file: {cite['filename']}"
                )

    def test_citations_have_quotes(self, rent_memo_conversation: str) -> None:
        """Every citation must include a non-empty quote."""
        msg = _get_structured_response(rent_memo_conversation)

        segments = msg.get("segments", [])
        for seg in segments:
            for cite in seg.get("citations", []):
                assert cite.get("quote") and len(cite["quote"].strip()) > 0, (
                    f"Citation in segment '{seg['text'][:50]}...' has empty quote"
                )

    def test_citations_are_verified(self, rent_memo_conversation: str) -> None:
        """Citations should be verified against the source document."""
        msg = _get_structured_response(rent_memo_conversation)

        segments = msg.get("segments", [])
        all_cites = [c for seg in segments for c in seg.get("citations", [])]
        verified = [c for c in all_cites if c.get("verified")]

        assert len(verified) > 0, "Expected at least one verified citation"
        assert len(verified) == len(all_cites), (
            f"Only {len(verified)}/{len(all_cites)} citations verified"
        )


# ──────────────────────────────────────────────────────────────────────────────
# RAG mode (forced by setting threshold to 1000 tokens)
# ──────────────────────────────────────────────────────────────────────────────

RAG_THRESHOLD = 1000


class TestRAGMode:
    """Tests using RAG mode (forced via low threshold of 1000 tokens)."""

    def test_q1_current_rent_rag(self, rag_conversation: str) -> None:
        """Q1 via RAG: What is the rent as at today's date?"""
        answer = _ask_question(
            rag_conversation,
            "What is the rent as at today's date?",
            rag_threshold=RAG_THRESHOLD,
        )
        answer_lower = answer.lower()
        assert any(
            term in answer_lower
            for term in ["1.75 million", "1,750,000", "£1.75m", "1.75m", "1,750,000"]
        ), f"[RAG] Expected rent of £1.75 million not found in answer:\n{answer}"

    def test_q2_rent_on_date_rag(self, rag_conversation: str) -> None:
        """Q2 via RAG: What was the rent on 15/08/2016?"""
        answer = _ask_question(
            rag_conversation,
            "What was the rent on 15/08/2016?",
            rag_threshold=RAG_THRESHOLD,
        )
        answer_lower = answer.lower()
        assert "peppercorn" in answer_lower, (
            f"[RAG] Expected 'peppercorn' in answer:\n{answer}"
        )

    def test_q3_tenant_rights_rag(self, rag_conversation: str) -> None:
        """Q3 via RAG: What rights are granted to the tenant?"""
        answer = _ask_question(
            rag_conversation,
            "What rights are granted to the tenant?",
            rag_threshold=RAG_THRESHOLD,
        )
        answer_lower = answer.lower()
        has_rights = "right" in answer_lower
        has_variation_ref = any(
            term in answer_lower
            for term in ["deed of variation", "additional", "new right", "varied", "supplemental"]
        )
        assert has_rights, f"[RAG] Expected discussion of rights in answer:\n{answer}"
        assert has_variation_ref, (
            f"[RAG] Expected reference to additional/new rights in answer:\n{answer}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# No-Azure mode: Anthropic vision OCR, no embeddings, full-context only
# Only uploads the Deed (12 image pages) and Rent Review Memo (text-only)
# to keep the test fast. Skips the 140-page Lease.
# ──────────────────────────────────────────────────────────────────────────────

NO_AZURE_DOC_FILES = [
    "Official Copy (NGL885533 - Deed - 31-03-2016).pdf",
    "Rent review memorandum - 8th Fl, Building 5, New Street Sq.pdf",
]


def _create_conversation_no_azure() -> str:
    """Create a conversation with only Deed + Rent Review, no Azure services.

    Uses query params to skip embedding and force Anthropic vision OCR,
    simulating an environment with only an Anthropic API key.
    Only uploads 2 smaller docs (not the 140-page Lease) to stay fast.
    """
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
                        "skip_ocr_cache": "true",
                    },
                )
                resp.raise_for_status()

        docs_resp = client.get(f"/api/conversations/{conv_id}/documents")
        docs_resp.raise_for_status()
        docs = docs_resp.json()
        assert len(docs) == 2, f"Expected 2 documents, got {len(docs)}"

    return conv_id


@pytest.fixture(scope="module")
def no_azure_conversation() -> str:
    """Conversation created without any Azure services (Anthropic-only)."""
    return _create_conversation_no_azure()


class TestNoAzureKeys:
    """Tests simulating an environment with only the Anthropic API key.

    Uploads only the Deed of Variation (12 scanned pages → Anthropic vision OCR)
    and the Rent Review Memo (text-based, no OCR needed).
    All queries use full-context mode since embeddings were skipped.
    """

    FULL_CONTEXT_THRESHOLD = 9_999_999

    def test_q1_current_rent_no_azure(self, no_azure_conversation: str) -> None:
        """Q1: What is the rent as at today's date? (Anthropic-only pipeline)
        Answer comes from text-based Rent Review Memo — no OCR needed.
        """
        answer = _ask_question(
            no_azure_conversation,
            "What is the rent as at today's date?",
            rag_threshold=self.FULL_CONTEXT_THRESHOLD,
        )
        answer_lower = answer.lower()
        assert any(
            term in answer_lower
            for term in ["1.75 million", "1,750,000", "£1.75m", "1.75m", "1,750,000"]
        ), f"[NoAzure] Expected rent of £1.75 million not found in answer:\n{answer}"

    def test_q2_rent_on_date_no_azure(self, no_azure_conversation: str) -> None:
        """Q2: What was the rent on 15/08/2016? (Anthropic vision OCR)
        Answer (peppercorn) comes from the scanned Deed of Variation.
        """
        answer = _ask_question(
            no_azure_conversation,
            "What was the rent on 15/08/2016?",
            rag_threshold=self.FULL_CONTEXT_THRESHOLD,
        )
        answer_lower = answer.lower()
        assert "peppercorn" in answer_lower, (
            f"[NoAzure] Expected 'peppercorn' in answer:\n{answer}"
        )
