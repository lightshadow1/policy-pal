"""
guardrails.py
Local analog of Amazon Bedrock Guardrails.

Implements two pipeline stages that mirror Bedrock's ApplyGuardrail API:
  check_input()  — pre-LLM: prompt attack, denied topics, word filters, length
  check_output() — post-LLM: contextual grounding, relevance, output PII masking

Actions mirror Bedrock's intervention model:
  PASS  — no issues detected; proceed normally
  WARN  — potential issue (e.g. low grounding); deliver with caution flag
  BLOCK — definitive violation; do not deliver to user

Thresholds and limits are configurable via environment variables so they can
differ between dev, staging, and production without code changes.
"""

import logging
import os
import re
from dataclasses import dataclass
from enum import Enum

from pii_scrubber import scrub

logger = logging.getLogger(__name__)

# ── Thresholds (mirrors Bedrock's configurable confidence thresholds) ─────────
GROUNDING_THRESHOLD = float(os.getenv("GUARDRAIL_GROUNDING_THRESHOLD", "0.25"))
RELEVANCE_THRESHOLD = float(os.getenv("GUARDRAIL_RELEVANCE_THRESHOLD", "0.15"))
MAX_QUESTION_CHARS  = int(os.getenv("MAX_QUESTION_CHARS", "2000"))

# ── Denied topics (mirrors Bedrock's "Denied topics" policy) ──────────────────
# Keys are topic names; values are lists of trigger phrases.
# Uses multi-word phrases to minimise false positives on legitimate HR/IT questions.
DENIED_TOPICS: dict[str, list[str]] = {
    "legal_advice": [
        "sue the company",
        "file a lawsuit",
        "legal action against",
        "wrongful termination lawsuit",
        "employment attorney",
        "file a legal claim",
    ],
    "medical_advice": [
        "medical diagnosis",
        "what medication should",
        "prescribe me",
        "is this cancer",
        "medical symptoms of",
    ],
    "investment_advice": [
        "invest my bonus",
        "buy stocks",
        "stock market advice",
        "cryptocurrency investment",
        "financial advisor recommendation",
    ],
}

# ── Word filters (mirrors Bedrock's "Word filters" policy) ────────────────────
# Exact-match blocklist applied to both input and output.
# Extend with org-specific terms (competitor names, profanity, etc.) as needed.
BLOCKED_WORDS: list[str] = []

# ── Prompt injection patterns (mirrors Bedrock "Content filters / Prompt attack") ─
# Conservative patterns that clearly indicate adversarial intent without
# false-positiving on legitimate policy questions.
INJECTION_PATTERNS: list[str] = [
    r"ignore\b.{0,40}instructions\b",
    r"forget (everything|all instructions|your rules|your system prompt)",
    r"\bact as (a |an )(?!employee|manager|admin)",
    r"\bjailbreak\b",
    r"\bDAN\b",
    r"\bdo anything now\b",
    r"disregard (your |all |the )?rules",
    r"override (your |the |all )?system",
    r"\bpretend (you are|to be) (a |an )(?!employee)",
    r"from now on[,.]? (you are|ignore|forget)",
    r"you have no restrictions",
    r"your (true |real )?purpose is",
    r"\bdeveloper mode\b",
    r"\bgod mode\b",
]

# ── Stop words for scoring ────────────────────────────────────────────────────
STOP_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "i", "my", "can", "do", "for",
    "what", "how", "when", "where", "who", "will", "be", "in", "of",
    "to", "and", "or", "but", "it", "this", "that", "with", "at",
    "by", "from", "on", "as", "if", "not", "was", "has", "have",
    "had", "its", "your", "our", "their", "we", "they", "you",
})

# ── Blocked messages ──────────────────────────────────────────────────────────
_BLOCKED_INPUT_MESSAGE = (
    "Your question could not be processed. "
    "Please ask a question related to company HR or IT policies. "
    "For other matters, contact HR (ext. 1001) or IT (ext. 2000) directly."
)

_WARNED_OUTPUT_MESSAGE = (
    "This answer has low grounding in the source policy document. "
    "Please verify directly with HR or IT before acting on this information."
)


# ── Data models ───────────────────────────────────────────────────────────────

class GuardrailAction(str, Enum):
    """Outcome of a guardrail check. Inherits from str for easy CSV serialisation."""
    PASS  = "PASS"
    WARN  = "WARN"
    BLOCK = "BLOCK"


@dataclass
class GuardrailResult:
    """
    Structured outcome of a guardrail check.
    Mirrors the assessment object returned by Bedrock's ApplyGuardrail API.
    """
    action:           GuardrailAction
    triggered_policy: str                  # e.g. "prompt_attack", "denied_topic:legal_advice"
    reason:           str                  # human-readable explanation for logs/audit
    grounding_score:  float | None = None  # 0.0–1.0; None when not applicable
    relevance_score:  float | None = None
    modified_text:    str | None   = None  # masked/redacted output text, if applicable
    blocked_message:  str | None   = None  # safe user-facing message on BLOCK/WARN


# ── Input stage ───────────────────────────────────────────────────────────────

def check_input(question: str) -> GuardrailResult:
    """
    Pre-LLM input guardrail. Mirrors Bedrock's ApplyGuardrail on the input side.

    Checks run in order (fail-fast on first violation):
      1. Input length
      2. Prompt injection / prompt attack
      3. Denied topics
      4. Word filters
    """
    # 1. Length check
    if len(question) > MAX_QUESTION_CHARS:
        logger.warning("Guardrail BLOCK: input too long (%d chars)", len(question))
        return GuardrailResult(
            action=GuardrailAction.BLOCK,
            triggered_policy="input_length",
            reason=f"Question exceeds maximum of {MAX_QUESTION_CHARS} characters.",
            blocked_message=_BLOCKED_INPUT_MESSAGE,
        )

    # 2. Prompt injection
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, question, re.IGNORECASE):
            logger.warning(
                "Guardrail BLOCK: prompt injection pattern matched — %s", pattern
            )
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                triggered_policy="prompt_attack",
                reason=f"Potential prompt injection detected (matched pattern: {pattern!r}).",
                blocked_message=_BLOCKED_INPUT_MESSAGE,
            )

    # 3. Denied topics
    q_lower = question.lower()
    for topic, phrases in DENIED_TOPICS.items():
        for phrase in phrases:
            if phrase in q_lower:
                logger.warning(
                    "Guardrail BLOCK: denied topic '%s' (phrase: %s)", topic, phrase
                )
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    triggered_policy=f"denied_topic:{topic}",
                    reason=f"Question touches denied topic '{topic}' (matched: {phrase!r}).",
                    blocked_message=_BLOCKED_INPUT_MESSAGE,
                )

    # 4. Word filters
    for word in BLOCKED_WORDS:
        if word.lower() in q_lower:
            logger.warning("Guardrail BLOCK: blocked word '%s' in input", word)
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                triggered_policy="word_filter",
                reason=f"Input contains a blocked term: {word!r}.",
                blocked_message=_BLOCKED_INPUT_MESSAGE,
            )

    return GuardrailResult(
        action=GuardrailAction.PASS,
        triggered_policy="none",
        reason="All input checks passed.",
    )


# ── Output stage ──────────────────────────────────────────────────────────────

def check_output(answer: str, source_doc: str, question: str) -> GuardrailResult:
    """
    Post-LLM output guardrail. Mirrors Bedrock's contextual grounding check
    and sensitive information filter applied to the model response.

    Checks run in order:
      1. Word filters on output
      2. Output PII masking (reuses pii_scrubber; WARN if modified)
      3. Contextual grounding score
      4. Relevance score
    """
    working_answer = answer

    # 1. Word filters on output
    for word in BLOCKED_WORDS:
        if word.lower() in working_answer.lower():
            logger.warning("Guardrail BLOCK: blocked word '%s' in output", word)
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                triggered_policy="word_filter",
                reason=f"Response contains a blocked term: {word!r}.",
                blocked_message=_BLOCKED_INPUT_MESSAGE,
            )

    # 2. Output PII masking
    pii_result = scrub(working_answer)
    if pii_result.was_modified:
        logger.info("Guardrail: masked PII in output — %s", pii_result.summary)
        working_answer = pii_result.scrubbed

    modified_text = working_answer if pii_result.was_modified else None

    # 3 & 4. Contextual grounding + relevance
    grounding = _score_grounding(working_answer, source_doc)
    relevance = _score_relevance(working_answer, question)

    logger.debug(
        "Guardrail scores — grounding=%.3f (threshold=%.2f), "
        "relevance=%.3f (threshold=%.2f)",
        grounding, GROUNDING_THRESHOLD, relevance, RELEVANCE_THRESHOLD,
    )

    if grounding < GROUNDING_THRESHOLD:
        logger.warning(
            "Guardrail WARN: grounding score %.3f < threshold %.2f",
            grounding, GROUNDING_THRESHOLD,
        )
        return GuardrailResult(
            action=GuardrailAction.WARN,
            triggered_policy="contextual_grounding",
            reason=(
                f"Grounding score {grounding:.2f} is below threshold "
                f"{GROUNDING_THRESHOLD:.2f}. Answer may not be fully "
                "supported by the policy document."
            ),
            grounding_score=grounding,
            relevance_score=relevance,
            modified_text=modified_text,
            blocked_message=_WARNED_OUTPUT_MESSAGE,
        )

    if relevance < RELEVANCE_THRESHOLD:
        logger.warning(
            "Guardrail WARN: relevance score %.3f < threshold %.2f",
            relevance, RELEVANCE_THRESHOLD,
        )
        return GuardrailResult(
            action=GuardrailAction.WARN,
            triggered_policy="relevance_check",
            reason=(
                f"Relevance score {relevance:.2f} is below threshold "
                f"{RELEVANCE_THRESHOLD:.2f}. Answer may not adequately "
                "address the question."
            ),
            grounding_score=grounding,
            relevance_score=relevance,
            modified_text=modified_text,
            blocked_message=_WARNED_OUTPUT_MESSAGE,
        )

    policy_label = "output_pii_masked" if pii_result.was_modified else "none"
    reason = (
        f"Output PII masked: {pii_result.summary}"
        if pii_result.was_modified
        else "All output checks passed."
    )

    return GuardrailResult(
        action=GuardrailAction.PASS,
        triggered_policy=policy_label,
        reason=reason,
        grounding_score=grounding,
        relevance_score=relevance,
        modified_text=modified_text,
    )


# ── Scoring helpers ───────────────────────────────────────────────────────────

def _score_grounding(answer: str, source_doc: str) -> float:
    """
    What fraction of key content words in the answer also appear in the source?

    Mirrors Bedrock's grounding score: higher = more grounded in the source.
    Only counts words of 4+ characters to filter noise and stop words.
    Returns 1.0 for empty answers (trivially grounded).
    """
    answer_words = {
        w for w in answer.lower().split()
        if len(w) >= 4 and w not in STOP_WORDS
    }
    if not answer_words:
        return 1.0

    source_words = {w for w in source_doc.lower().split() if len(w) >= 4}
    return len(answer_words & source_words) / len(answer_words)


def _score_relevance(answer: str, question: str) -> float:
    """
    What fraction of key question terms are addressed in the answer?

    Mirrors Bedrock's relevance score: higher = answer addresses the question.
    Returns 1.0 for empty questions (trivially relevant).
    """
    question_words = {
        w for w in question.lower().split()
        if len(w) >= 4 and w not in STOP_WORDS
    }
    if not question_words:
        return 1.0

    answer_words = {w for w in answer.lower().split() if len(w) >= 4}
    return len(question_words & answer_words) / len(question_words)
