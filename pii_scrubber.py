"""
pii_scrubber.py
Enterprise compliance layer — redacts PII before it reaches the LLM.

In production this gets replaced by Amazon Comprehend.
This version uses regex patterns to demonstrate the same concept locally.
"""

import re
from dataclasses import dataclass, field

# PII patterns to detect and redact
PII_PATTERNS = {
    "email":       r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "phone":       r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "ssn":         r"\b\d{3}-\d{2}-\d{4}\b",
    "employee_id": r"\bEMP-\d{4,6}\b",
    "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    "postal_code": r"\b[A-Z]\d[A-Z]\s?\d[A-Z]\d\b",  # Canadian postal codes
}


@dataclass
class ScrubResult:
    original: str
    scrubbed: str
    redactions: list[dict] = field(default_factory=list)

    @property
    def was_modified(self) -> bool:
        return len(self.redactions) > 0

    @property
    def summary(self) -> str:
        if not self.redactions:
            return "No PII detected"
        counts = {}
        for r in self.redactions:
            counts[r["type"]] = counts.get(r["type"], 0) + 1
        return ", ".join(f"{v} {k}(s) removed" for k, v in counts.items())


def scrub(text: str) -> ScrubResult:
    """
    Detect and redact PII from user input.
    Returns both the scrubbed text and a log of what was removed.
    The original is stored for audit purposes — never sent to the LLM.
    """
    redacted = text
    redactions = []

    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, redacted, re.IGNORECASE)
        if matches:
            for match in matches:
                # Handle tuple matches from groups
                matched_str = match if isinstance(match, str) else match[0]
                if matched_str:
                    redactions.append({
                        "type": pii_type,
                        "value_length": len(matched_str),  # Log length, never the value
                    })
            redacted = re.sub(
                pattern,
                f"[{pii_type.upper()} REDACTED]",
                redacted,
                flags=re.IGNORECASE
            )

    return ScrubResult(
        original=text,
        scrubbed=redacted,
        redactions=redactions
    )


if __name__ == "__main__":
    # Quick test
    test = "Hi, my name is John, my email is john.doe@company.com and my employee ID is EMP-12345. Can I work from home?"
    result = scrub(test)
    print(f"Original:  {result.original}")
    print(f"Scrubbed:  {result.scrubbed}")
    print(f"Summary:   {result.summary}")
