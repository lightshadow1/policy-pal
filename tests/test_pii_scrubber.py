"""
tests/test_pii_scrubber.py
Unit tests for the PII scrubbing layer.
"""

import sys
import os

# Allow importing from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pii_scrubber import scrub, ScrubResult


class TestScrubResult:
    def test_was_modified_no_redactions(self) -> None:
        result = ScrubResult(original="hello", scrubbed="hello", redactions=[])
        assert result.was_modified is False

    def test_was_modified_with_redactions(self) -> None:
        result = ScrubResult(
            original="my email is foo@bar.com",
            scrubbed="my email is [EMAIL REDACTED]",
            redactions=[{"type": "email", "value_length": 11}],
        )
        assert result.was_modified is True

    def test_summary_no_pii(self) -> None:
        result = ScrubResult(original="clean text", scrubbed="clean text", redactions=[])
        assert result.summary == "No PII detected"

    def test_summary_single_type(self) -> None:
        result = ScrubResult(
            original="",
            scrubbed="",
            redactions=[{"type": "email", "value_length": 15}],
        )
        assert "email" in result.summary
        assert "1" in result.summary

    def test_summary_multiple_types(self) -> None:
        result = ScrubResult(
            original="",
            scrubbed="",
            redactions=[
                {"type": "email", "value_length": 15},
                {"type": "phone", "value_length": 12},
            ],
        )
        assert "email" in result.summary
        assert "phone" in result.summary

    def test_summary_counts_same_type(self) -> None:
        result = ScrubResult(
            original="",
            scrubbed="",
            redactions=[
                {"type": "email", "value_length": 10},
                {"type": "email", "value_length": 12},
            ],
        )
        assert "2 email(s) removed" in result.summary


class TestScrubFunction:
    def test_no_pii_returns_unchanged(self) -> None:
        text = "How many vacation days do I get per year?"
        result = scrub(text)
        assert result.scrubbed == text
        assert result.was_modified is False
        assert result.redactions == []

    def test_email_redacted(self) -> None:
        result = scrub("Contact me at bob@company.com for details.")
        assert "bob@company.com" not in result.scrubbed
        assert "[EMAIL REDACTED]" in result.scrubbed
        assert result.was_modified is True
        assert any(r["type"] == "email" for r in result.redactions)

    def test_phone_redacted(self) -> None:
        result = scrub("Call me at 416-555-1234 anytime.")
        assert "416-555-1234" not in result.scrubbed
        assert "[PHONE REDACTED]" in result.scrubbed
        assert result.was_modified is True

    def test_ssn_redacted(self) -> None:
        result = scrub("My SSN is 123-45-6789.")
        assert "123-45-6789" not in result.scrubbed
        assert "[SSN REDACTED]" in result.scrubbed
        assert result.was_modified is True

    def test_employee_id_redacted(self) -> None:
        result = scrub("My employee ID is EMP-12345, how do I submit expenses?")
        assert "EMP-12345" not in result.scrubbed
        assert "[EMPLOYEE_ID REDACTED]" in result.scrubbed
        assert result.was_modified is True

    def test_canadian_postal_code_redacted(self) -> None:
        result = scrub("I live at postal code M5V 3A8.")
        assert "M5V 3A8" not in result.scrubbed
        assert "[POSTAL_CODE REDACTED]" in result.scrubbed
        assert result.was_modified is True

    def test_multiple_pii_types_redacted(self) -> None:
        text = "Hi, my email is john@company.com and my ID is EMP-99999."
        result = scrub(text)
        assert "john@company.com" not in result.scrubbed
        assert "EMP-99999" not in result.scrubbed
        assert result.was_modified is True
        types_found = {r["type"] for r in result.redactions}
        assert "email" in types_found
        assert "employee_id" in types_found

    def test_original_preserved(self) -> None:
        text = "My email is secret@example.com"
        result = scrub(text)
        assert result.original == text

    def test_value_length_logged_not_value(self) -> None:
        text = "Email: test@domain.com"
        result = scrub(text)
        for redaction in result.redactions:
            assert "value" not in redaction or redaction.get("type") is not None
            assert "value_length" in redaction
            # The actual email string should NOT be stored
            assert "test@domain.com" not in str(redaction)

    def test_empty_string(self) -> None:
        result = scrub("")
        assert result.scrubbed == ""
        assert result.was_modified is False

    def test_case_insensitive_employee_id(self) -> None:
        result = scrub("My id is emp-12345.")
        assert "emp-12345" not in result.scrubbed.lower()
        assert result.was_modified is True
