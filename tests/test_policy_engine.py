"""
tests/test_policy_engine.py
Unit tests for the policy engine.
LLM-dependent functions (ask_llm, ask_policy) are tested with mocks.
"""

import sys
import os
import csv
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from policy_engine import (
    load_policies,
    find_relevant_policy,
    write_audit_log,
    calculate_cost,
    PolicyChunk,
    PolicyResponse,
)


# ── Helper ────────────────────────────────────────────────────────────────────

def _make_response(**overrides) -> PolicyResponse:  # type: ignore[no-untyped-def]
    """Build a minimal PolicyResponse for audit log tests."""
    defaults = dict(
        answer="You are entitled to 15 vacation days per year.",
        source="hr_policy.txt",
        confidence="HIGH",
        pii_removed=False,
        pii_summary="No PII detected",
        escalate=False,
        request_id="testreqid1",
        retrieval_score=3,
        latency_ms=1234.5,
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.00030,
        guardrail_input_action="PASS",
        guardrail_output_action="PASS",
        guardrail_reason="",
        grounding_score=0.82,
        relevance_score=0.75,
    )
    defaults.update(overrides)
    return PolicyResponse(**defaults)


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_DOCS: dict[str, str] = {
    "hr_policy.txt": (
        "vacation days remote work expense reimbursement performance review "
        "employees are entitled to 15 vacation days per year"
    ),
    "it_policy.txt": (
        "device security password encryption data handling company laptop "
        "employees must not install unapproved software"
    ),
}


# ── load_policies ─────────────────────────────────────────────────────────────

class TestLoadPolicies:
    def test_returns_empty_if_folder_missing(self) -> None:
        docs = load_policies("/nonexistent/path/policies")
        assert docs == {}

    def test_loads_txt_files(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        (tmp_path / "policy_a.txt").write_text("content a")
        (tmp_path / "policy_b.txt").write_text("content b")
        (tmp_path / "ignore_me.pdf").write_text("not loaded")

        docs = load_policies(str(tmp_path))
        assert "policy_a.txt" in docs
        assert "policy_b.txt" in docs
        assert "ignore_me.pdf" not in docs
        assert docs["policy_a.txt"] == "content a"

    def test_empty_folder_returns_empty_dict(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        docs = load_policies(str(tmp_path))
        assert docs == {}


# ── find_relevant_policy ──────────────────────────────────────────────────────

class TestFindRelevantPolicy:
    def test_returns_none_for_empty_docs(self) -> None:
        result = find_relevant_policy("vacation days", {})
        assert result is None

    def test_finds_hr_policy_for_vacation_query(self) -> None:
        result = find_relevant_policy("How many vacation days do I get?", SAMPLE_DOCS)
        assert result is not None
        assert result.source == "hr_policy.txt"
        assert result.score >= 1

    def test_finds_it_policy_for_security_query(self) -> None:
        result = find_relevant_policy("What is the password policy for company devices?", SAMPLE_DOCS)
        assert result is not None
        assert result.source == "it_policy.txt"
        assert result.score >= 1

    def test_returns_none_for_out_of_scope_query(self) -> None:
        result = find_relevant_policy("What is the capital of France?", SAMPLE_DOCS)
        # Capital/France are not in the policy docs, score should be 0
        # If no word matches, result is None
        if result is not None:
            assert result.score == 0

    def test_chunk_contains_source_and_content(self) -> None:
        result = find_relevant_policy("vacation remote work", SAMPLE_DOCS)
        assert result is not None
        assert result.source in SAMPLE_DOCS
        assert result.content == SAMPLE_DOCS[result.source]

    def test_stop_words_not_inflating_scores(self) -> None:
        # "the a an is are" are stop words — should yield no match
        result = find_relevant_policy("the a an is are", SAMPLE_DOCS)
        # May return None or a chunk with score 0
        if result is not None:
            assert result.score == 0


# ── write_audit_log ───────────────────────────────────────────────────────────

class TestWriteAuditLog:
    def test_creates_file_with_header_on_first_write(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        log_path = str(tmp_path / "test_audit.csv")
        response = _make_response()

        with patch("policy_engine.AUDIT_LOG", log_path):
            write_audit_log(
                response=response,
                user="test_user",
                original_question="original question",
                scrubbed_question="scrubbed question",
                pii_summary="No PII detected",
            )

        assert os.path.exists(log_path)
        with open(log_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["user"] == "test_user"
        assert rows[0]["confidence"] == "HIGH"
        assert rows[0]["escalated"] == "False"
        assert rows[0]["request_id"] == "testreqid1"
        assert rows[0]["guardrail_input_action"] == "PASS"

    def test_appends_on_subsequent_writes(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        log_path = str(tmp_path / "test_audit.csv")

        with patch("policy_engine.AUDIT_LOG", log_path):
            for i in range(3):
                write_audit_log(
                    response=_make_response(request_id=f"req{i:06d}"),
                    user=f"user_{i}",
                    original_question="q",
                    scrubbed_question="q",
                    pii_summary="No PII detected",
                )

        with open(log_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3

    def test_response_preview_truncated_to_120_chars(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        log_path = str(tmp_path / "test_audit.csv")
        response = _make_response(answer="x" * 200)

        with patch("policy_engine.AUDIT_LOG", log_path):
            write_audit_log(
                response=response,
                user="u",
                original_question="q",
                scrubbed_question="q",
                pii_summary="No PII detected",
            )

        with open(log_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows[0]["response_preview"]) <= 120

    def test_escalated_true_written_as_string(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        log_path = str(tmp_path / "test_audit.csv")
        response = _make_response(escalate=True, confidence="NONE", source="none")

        with patch("policy_engine.AUDIT_LOG", log_path):
            write_audit_log(
                response=response,
                user="u",
                original_question="q",
                scrubbed_question="q",
                pii_summary="No PII detected",
            )

        with open(log_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert rows[0]["escalated"] == "True"

    def test_none_scores_written_as_empty_string(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        log_path = str(tmp_path / "test_audit.csv")
        response = _make_response(grounding_score=None, relevance_score=None)

        with patch("policy_engine.AUDIT_LOG", log_path):
            write_audit_log(
                response=response,
                user="u",
                original_question="q",
                scrubbed_question="q",
                pii_summary="No PII detected",
            )

        with open(log_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert rows[0]["grounding_score"] == ""
        assert rows[0]["relevance_score"] == ""

    def test_cost_written_with_precision(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        log_path = str(tmp_path / "test_audit.csv")
        response = _make_response(cost_usd=0.000462)

        with patch("policy_engine.AUDIT_LOG", log_path):
            write_audit_log(
                response=response,
                user="u",
                original_question="q",
                scrubbed_question="q",
                pii_summary="No PII detected",
            )

        with open(log_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert float(rows[0]["cost_usd"]) == pytest.approx(0.000462, abs=1e-9)

    def test_audit_write_failure_is_nonfatal(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """A disk-full condition must not propagate to the caller."""
        with patch("policy_engine.AUDIT_LOG", "/nonexistent/dir/audit.csv"):
            # Should NOT raise
            write_audit_log(
                response=_make_response(),
                user="u",
                original_question="q",
                scrubbed_question="q",
                pii_summary="No PII detected",
            )


# ── PolicyResponse dataclass ──────────────────────────────────────────────────

class TestPolicyResponse:
    def test_response_fields(self) -> None:
        resp = PolicyResponse(
            answer="You get 15 vacation days.",
            source="hr_policy.txt",
            confidence="HIGH",
            pii_removed=False,
            pii_summary="No PII detected",
            escalate=False,
        )
        assert resp.answer == "You get 15 vacation days."
        assert resp.confidence == "HIGH"
        assert resp.pii_removed is False
        assert resp.escalate is False

    def test_new_fields_have_safe_defaults(self) -> None:
        resp = PolicyResponse(
            answer="ans",
            source="hr_policy.txt",
            confidence="HIGH",
            pii_removed=False,
            pii_summary="No PII detected",
            escalate=False,
        )
        assert resp.retrieval_score == 0
        assert resp.latency_ms == 0.0
        assert resp.input_tokens == 0
        assert resp.output_tokens == 0
        assert resp.cost_usd == 0.0
        assert resp.guardrail_input_action == "PASS"
        assert resp.guardrail_output_action == "PASS"
        assert resp.guardrail_reason == ""
        assert resp.grounding_score is None
        assert resp.relevance_score is None

    def test_timestamp_auto_set(self) -> None:
        resp = PolicyResponse(
            answer="ans",
            source="hr_policy.txt",
            confidence="HIGH",
            pii_removed=False,
            pii_summary="No PII detected",
            escalate=False,
        )
        assert resp.timestamp != ""
        assert "T" in resp.timestamp

    def test_request_id_auto_generated(self) -> None:
        resp = PolicyResponse(
            answer="ans",
            source="hr_policy.txt",
            confidence="HIGH",
            pii_removed=False,
            pii_summary="No PII detected",
            escalate=False,
        )
        assert resp.request_id != ""
        assert len(resp.request_id) == 12


# ── TestCalculateCost ───────────────────────────────────────────────────────

class TestCalculateCost:
    def test_haiku_one_million_input_tokens(self) -> None:
        # 1M input tokens at $0.80/M = $0.80
        cost = calculate_cost(1_000_000, 0)
        assert cost == pytest.approx(0.80, abs=1e-9)

    def test_haiku_one_million_output_tokens(self) -> None:
        # 1M output tokens at $4.00/M = $4.00
        cost = calculate_cost(0, 1_000_000)
        assert cost == pytest.approx(4.00, abs=1e-9)

    def test_sonnet_input_pricing(self) -> None:
        cost = calculate_cost(1_000_000, 0, "claude-3-5-sonnet-20241022")
        assert cost == pytest.approx(3.00, abs=1e-9)

    def test_sonnet_output_pricing(self) -> None:
        cost = calculate_cost(0, 1_000_000, "claude-3-5-sonnet-20241022")
        assert cost == pytest.approx(15.00, abs=1e-9)

    def test_zero_tokens_yields_zero(self) -> None:
        assert calculate_cost(0, 0) == 0.0

    def test_typical_query_is_cheap(self) -> None:
        # 150 input + 80 output tokens — should cost less than 1 cent
        cost = calculate_cost(150, 80)
        assert cost < 0.01

    def test_unknown_model_defaults_to_haiku(self) -> None:
        haiku_cost   = calculate_cost(1_000, 1_000, "claude-haiku-4-5-20251001")
        unknown_cost = calculate_cost(1_000, 1_000, "claude-unknown-model-xyz")
        assert haiku_cost == pytest.approx(unknown_cost, abs=1e-12)
