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


def _parse_sse_response(response: httpx.Response) -> str:
    """Extract the final assistant message content from an SSE stream."""
    content = ""
    for line in response.text.splitlines():
        line = line.strip()
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        try:
            parsed = json.loads(data)
            if parsed.get("type") == "message":
                return parsed["message"]["content"]
            if parsed.get("type") == "content":
                content += parsed.get("content", "")
        except json.JSONDecodeError:
            continue
    return content


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

    with httpx.Client(base_url=BASE_URL, timeout=120) as client:
        resp = client.post(
            f"/api/conversations/{conv_id}/messages",
            json={"content": question},
            params=params,
        )
        resp.raise_for_status()
        return _parse_sse_response(resp)


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
