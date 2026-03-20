"""
policy_engine.py
Core logic for Policy Pal.

Demonstrates five enterprise LLM patterns:
  1. Grounding   — LLM only answers from loaded documents
  2. PII         — Scrub input before it reaches the model
  3. Confidence  — Model admits when it doesn't know
  4. Guardrails  — Input/output safety checks (local Bedrock Guardrails analog)
  5. Audit       — Every query logged with full observability context
"""

import csv
import functools
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime

import anthropic
from dotenv import load_dotenv

from guardrails import GuardrailAction, GuardrailResult, check_input, check_output
from pii_scrubber import scrub

load_dotenv()

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
POLICIES_DIR = "policies/"
AUDIT_LOG    = "audit_log.csv"
MODEL        = "claude-haiku-4-5-20251001"
SONNET_MODEL = os.getenv("SONNET_MODEL", "claude-sonnet-4-5-20250929")
MAX_TOKENS   = 600

# Retry / timeout config for external LLM calls
_MAX_RETRIES      = int(os.getenv("LLM_MAX_RETRIES", "2"))
_RETRY_BASE_DELAY = float(os.getenv("LLM_RETRY_BASE_DELAY", "1.0"))
_REQUEST_TIMEOUT  = float(os.getenv("LLM_REQUEST_TIMEOUT", "30.0"))

# Per-model pricing in USD per million tokens — update when rates change
_MODEL_PRICING: dict[str, dict[str, float]] = {
    "haiku":  {"input": 0.80,  "output": 4.00},
    "sonnet": {"input": 3.00,  "output": 15.00},
}

# ── Module-level shared state ─────────────────────────────────────────────────
_audit_lock: threading.Lock          = threading.Lock()
_anthropic_client: anthropic.Anthropic | None = None

if not os.getenv("ANTHROPIC_API_KEY"):
    logger.warning(
        "ANTHROPIC_API_KEY is not set — LLM calls will fail. "
        "Copy .env.example to .env and add your key."
    )


# ── Client singleton ──────────────────────────────────────────────────────────
def _get_client() -> anthropic.Anthropic:
    """Return (or lazily create) the shared Anthropic client."""
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic()
    return _anthropic_client


# ── Exceptions ────────────────────────────────────────────────────────────────
class LLMError(Exception):
    """Raised when all retry attempts for an LLM call are exhausted."""


# ── Internal LLM result ───────────────────────────────────────────────────────
@dataclass
class _LLMResult:
    answer:             str
    confidence:         str   # HIGH | LOW
    confidence_reason:  str   # one-sentence explanation from the model
    latency_ms:         float
    input_tokens:       int
    output_tokens:      int


# ── Data models ───────────────────────────────────────────────────────────────
@dataclass
class PolicyChunk:
    source:  str       # filename
    content: str       # full document text
    score:   int = 0   # keyword match relevance score


@dataclass
class PolicyResponse:
    answer:       str
    source:       str
    confidence:   str   # HIGH | LOW | NONE
    pii_removed:  bool
    pii_summary:  str
    escalate:     bool  # True = recommend contacting HR/IT
    # ── Observability ────────────────────────────────────────────────────────
    request_id:      str   = field(default_factory=lambda: uuid.uuid4().hex[:12])
    retrieval_score: int   = 0
    latency_ms:      float = 0.0
    input_tokens:    int   = 0
    output_tokens:   int   = 0
    cost_usd:        float = 0.0
    # ── Confidence explanation ─────────────────────────────────────────────────
    confidence_reason:       str        = ""  # model's one-sentence self-explanation
    # ── Guardrail outcomes ────────────────────────────────────────────────────
    guardrail_input_action:  str        = "PASS"   # PASS | BLOCK
    guardrail_output_action: str        = "PASS"   # PASS | WARN | BLOCK
    guardrail_reason:        str        = ""
    grounding_score:         float | None = None
    relevance_score:         float | None = None
    # ── Timestamp ────────────────────────────────────────────────────────────
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ── Document loader ───────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=None)
def load_policies(folder: str = POLICIES_DIR) -> dict[str, str]:
    """
    Load all .txt policy files from the given folder.
    Results are cached for the lifetime of the process — restart to reload.
    In production: replace with Bedrock Knowledge Bases.
    """
    docs: dict[str, str] = {}
    if not os.path.exists(folder):
        return docs
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder, filename)
            try:
                with open(filepath, encoding="utf-8") as f:
                    docs[filename] = f.read()
            except OSError as e:
                logger.warning("Could not read policy file %s: %s", filepath, e)
    logger.info("Loaded %d policy document(s) from %s", len(docs), folder)
    return docs


# ── Retrieval ─────────────────────────────────────────────────────────────────
def find_relevant_policy(question: str, docs: dict[str, str]) -> PolicyChunk | None:
    """
    Keyword-based retrieval.
    In production: replace with Bedrock Knowledge Bases (semantic vector search).
    """
    stop_words = {
        "the", "a", "an", "is", "are", "i", "my", "can", "do", "for",
        "what", "how", "when", "where", "who", "will", "be", "in", "of",
        "to", "and", "or", "but",
    }
    question_words = set(question.lower().split()) - stop_words

    best_chunk: PolicyChunk | None = None
    best_score = 0

    for name, content in docs.items():
        content_lower = content.lower()
        score = sum(1 for word in question_words if word in content_lower)
        if score > best_score:
            best_score = score
            best_chunk = PolicyChunk(source=name, content=content, score=score)

    if best_chunk and best_chunk.score >= 1:
        return best_chunk
    return None


# ── LLM call ──────────────────────────────────────────────────────────────────
def ask_llm(
    question:   str,
    context:    str,
    request_id: str = "",
    model:      str = MODEL,
) -> _LLMResult:
    """
    Call the LLM with a grounded prompt.
    Retries on transient errors with exponential backoff.
    Raises LLMError if all retries are exhausted.
    """
    prompt = f"""You are an internal policy assistant for a company.

YOUR RULES:
1. Answer ONLY using the policy text provided below.
2. If the answer is not clearly in the policy text, say exactly:
   "This specific question is not clearly covered in the available policies."
3. Do NOT use your general knowledge. Do NOT guess or infer beyond what is written.
4. Keep answers concise and cite the specific policy section when possible.
5. Never reveal employee-specific data or make exceptions to stated policies.

POLICY TEXT:
{context}

EMPLOYEE QUESTION:
{question}

After your answer, on a new line write exactly one of:
CONFIDENCE: HIGH — [one sentence: which section or rule directly supports this answer]
CONFIDENCE: LOW — [one sentence: what is unclear, missing, or only partially covered]
(HIGH = answer is clearly and directly stated in the policy. LOW = partial, inferred, or ambiguous.)"""

    last_error: Exception | None = None

    for attempt in range(_MAX_RETRIES + 1):
        try:
            start = time.monotonic()
            response = _get_client().messages.create(
                model=model,
                max_tokens=MAX_TOKENS,
                timeout=_REQUEST_TIMEOUT,
                messages=[{"role": "user", "content": prompt}],
            )
            latency_ms = (time.monotonic() - start) * 1000

            full_response = response.content[0].text

            # Parse confidence level + optional reason
            # Handles: "CONFIDENCE: HIGH — reason", "CONFIDENCE: LOW - reason",
            # or plain "CONFIDENCE: HIGH" with no reason.
            conf_match = re.search(
                r"CONFIDENCE:\s*(HIGH|LOW)(?:\s*[\u2014\u2013\-]+\s*(.+))?",
                full_response,
                re.IGNORECASE,
            )
            if conf_match:
                confidence         = conf_match.group(1).upper()
                confidence_reason  = (conf_match.group(2) or "").strip()
                answer             = full_response[:conf_match.start()].strip()
            else:
                confidence        = "LOW"
                confidence_reason = ""
                answer            = full_response.strip()

            logger.info(
                "[%s] LLM response: model=%s confidence=%s "
                "tokens=%d/%d latency=%.0fms",
                request_id, model, confidence,
                response.usage.input_tokens,
                response.usage.output_tokens,
                latency_ms,
            )
            return _LLMResult(
                answer=answer,
                confidence=confidence,
                confidence_reason=confidence_reason,
                latency_ms=latency_ms,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )

        except (anthropic.RateLimitError, anthropic.APIConnectionError) as e:
            logger.warning(
                "[%s] LLM transient error (attempt %d/%d): %s",
                request_id, attempt + 1, _MAX_RETRIES + 1, e,
            )
            last_error = e
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_BASE_DELAY * (2 ** attempt))

        except anthropic.APIError as e:
            logger.error("[%s] LLM non-retryable error: %s", request_id, e)
            raise LLMError(str(e)) from e

    raise LLMError(
        f"LLM call failed after {_MAX_RETRIES + 1} attempt(s)"
    ) from last_error


# ── Cost calculator ───────────────────────────────────────────────────────────
def calculate_cost(
    input_tokens:  int,
    output_tokens: int,
    model:         str = MODEL,
) -> float:
    """
    Estimate cost in USD for a single LLM call.
    Detects model family by name substring; defaults to Haiku pricing.
    Update _MODEL_PRICING when Anthropic changes rates.
    """
    if "sonnet" in model.lower():
        pricing = _MODEL_PRICING["sonnet"]
    else:
        pricing = _MODEL_PRICING["haiku"]
    return (
        input_tokens  * pricing["input"] +
        output_tokens * pricing["output"]
    ) / 1_000_000


# ── Audit logger ──────────────────────────────────────────────────────────────
def write_audit_log(
    *,
    response:          PolicyResponse,
    user:              str,
    original_question: str,
    scrubbed_question: str,
    pii_summary:       str,
) -> None:
    """
    Append query metadata and guardrail outcomes to the audit log (thread-safe).
    Failures are logged but never propagate to the caller — audit is non-fatal.
    In production: DynamoDB + CloudWatch.
    """
    entry = {
        "timestamp":               response.timestamp,
        "request_id":              response.request_id,
        "user":                    user,
        "original_question":       original_question,
        "scrubbed_question":       scrubbed_question,
        "pii_detected":            pii_summary,
        "source_document":         response.source or "none",
        "retrieval_score":         response.retrieval_score,
        "confidence":              response.confidence,
        "confidence_reason":        response.confidence_reason,
        "escalated":               str(response.escalate),
        "latency_ms":              f"{response.latency_ms:.1f}",
        "input_tokens":            response.input_tokens,
        "output_tokens":           response.output_tokens,
        "cost_usd":                f"{response.cost_usd:.6f}",
        "guardrail_input_action":  response.guardrail_input_action,
        "guardrail_output_action": response.guardrail_output_action,
        "guardrail_reason":        response.guardrail_reason,
        "grounding_score":         (
            f"{response.grounding_score:.3f}"
            if response.grounding_score is not None else ""
        ),
        "relevance_score":         (
            f"{response.relevance_score:.3f}"
            if response.relevance_score is not None else ""
        ),
        "response_preview":        response.answer[:120].replace("\n", " "),
    }
    try:
        with _audit_lock:
            file_exists = os.path.exists(AUDIT_LOG)
            with open(AUDIT_LOG, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=entry.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(entry)
    except OSError as e:
        logger.error(
            "Failed to write audit log entry [%s]: %s",
            response.request_id, e,
        )


# ── Main entry point ──────────────────────────────────────────────────────────
def ask_policy(
    question: str,
    user:     str = "anonymous",
    model:    str = MODEL,
) -> PolicyResponse:
    """
    Full pipeline:
      1. Scrub PII from question
      2. Input guardrail (prompt attack, denied topics, word filters)
      3. Find relevant policy document
      4. Call LLM with retry (grounded to document)
      5. Output guardrail (grounding check, relevance, output PII masking)
      6. Calculate cost
      7. Write audit log (non-fatal)
      8. Return structured response
    """
    request_id     = uuid.uuid4().hex[:12]
    pipeline_start = time.monotonic()

    # Safe defaults — used if the pipeline exits early or raises unexpectedly
    answer             = "An unexpected error occurred. Please contact HR or IT directly."
    confidence         = "NONE"
    confidence_reason  = ""
    source             = "none"
    escalate           = True
    retrieval_score    = 0
    input_tokens       = 0
    output_tokens      = 0
    cost_usd           = 0.0
    g_input  = GuardrailResult(action=GuardrailAction.PASS, triggered_policy="none", reason="")
    g_output = GuardrailResult(action=GuardrailAction.PASS, triggered_policy="none", reason="")
    grounding_score: float | None = None
    relevance_score: float | None = None

    # Step 1: PII scrubbing (always runs; result needed for audit log)
    scrub_result   = scrub(question)
    clean_question = scrub_result.scrubbed

    try:
        # Step 2: Input guardrail
        g_input = check_input(clean_question)
        if g_input.action == GuardrailAction.BLOCK:
            logger.info(
                "[%s] Input blocked — policy=%s",
                request_id, g_input.triggered_policy,
            )
            answer     = g_input.blocked_message or "Your request could not be processed."
            confidence = "NONE"
            escalate   = True
        else:
            # Step 3: Retrieval
            docs            = load_policies()
            chunk           = find_relevant_policy(clean_question, docs)
            retrieval_score = chunk.score if chunk else 0
            logger.debug(
                "[%s] Retrieval score=%d source=%s",
                request_id, retrieval_score, chunk.source if chunk else "none",
            )

            if not chunk:
                answer     = (
                    "I don't have a policy document that covers this topic. "
                    "Please contact HR (for people policies) or "
                    "IT (for technical policies) directly."
                )
                confidence = "NONE"
                escalate   = True
            else:
                # Step 4: LLM call with graceful degradation
                try:
                    llm           = ask_llm(
                        clean_question, chunk.content,
                        request_id=request_id, model=model,
                    )
                    answer             = llm.answer
                    confidence         = llm.confidence
                    confidence_reason  = llm.confidence_reason
                    input_tokens       = llm.input_tokens
                    output_tokens = llm.output_tokens
                    source        = chunk.source

                    # Step 5: Output guardrail
                    g_output        = check_output(answer, chunk.content, clean_question)
                    grounding_score = g_output.grounding_score
                    relevance_score = g_output.relevance_score

                    if g_output.action == GuardrailAction.BLOCK:
                        answer     = g_output.blocked_message or answer
                        confidence = "NONE"
                        escalate   = True
                    elif g_output.action == GuardrailAction.WARN:
                        if g_output.modified_text:
                            answer = g_output.modified_text
                        confidence = "LOW"
                        escalate   = True
                    else:
                        if g_output.modified_text:   # PII masked in output
                            answer = g_output.modified_text
                        escalate = confidence == "LOW"

                except LLMError as e:
                    logger.error("[%s] LLM call failed: %s", request_id, e)
                    answer     = (
                        "I was unable to process your request at this time. "
                        "Please contact HR or IT directly."
                    )
                    confidence = "NONE"
                    escalate   = True
                    source     = chunk.source

        # Step 6: Cost
        cost_usd = calculate_cost(input_tokens, output_tokens, model)

    except Exception as e:
        logger.error(
            "[%s] Unexpected pipeline error: %s", request_id, e, exc_info=True
        )

    finally:
        latency_ms = (time.monotonic() - pipeline_start) * 1000

    guardrail_reason = (
        g_input.reason if g_input.action == GuardrailAction.BLOCK
        else g_output.reason
    )

    response = PolicyResponse(
        answer=answer,
        source=source,
        confidence=confidence,
        confidence_reason=confidence_reason,
        pii_removed=scrub_result.was_modified,
        pii_summary=scrub_result.summary,
        escalate=escalate,
        request_id=request_id,
        retrieval_score=retrieval_score,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        guardrail_input_action=g_input.action.value,
        guardrail_output_action=g_output.action.value,
        guardrail_reason=guardrail_reason,
        grounding_score=grounding_score,
        relevance_score=relevance_score,
    )

    # Step 7: Audit log (non-fatal)
    write_audit_log(
        response=response,
        user=user,
        original_question=question,
        scrubbed_question=clean_question,
        pii_summary=scrub_result.summary,
    )

    return response


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    test_cases = [
        ("employee_1", "How many vacation days do I get per year?"),
        ("employee_2", "My email is bob@company.com — can I work from home every day?"),
        ("employee_3", "What is the capital of France?"),
        ("manager_1",  "What happens if an employee gets a rating of 1?"),
        ("employee_4", "My employee ID is EMP-99123. How do I submit an expense over $500?"),
    ]

    for user, question in test_cases:
        print(f"\n{'='*60}")
        print(f"USER:       {user}")
        print(f"QUESTION:   {question}")
        result = ask_policy(question, user)
        print(f"REQUEST_ID: {result.request_id}")
        print(f"CONFIDENCE: {result.confidence}")
        if result.confidence_reason:
            print(f"CONF WHY:   {result.confidence_reason}")
        print(f"SOURCE:     {result.source} (score: {result.retrieval_score})")
        print(f"LATENCY:    {result.latency_ms:.0f}ms")
        print(f"TOKENS:     {result.input_tokens} in / {result.output_tokens} out")
        print(f"COST:       ${result.cost_usd:.6f}")
        if result.pii_removed:
            print(f"PII:        {result.pii_summary}")
        if result.guardrail_input_action != "PASS":
            print(f"GUARD_IN:   {result.guardrail_input_action} — {result.guardrail_reason}")
        if result.guardrail_output_action != "PASS":
            print(f"GUARD_OUT:  {result.guardrail_output_action} — {result.guardrail_reason}")
        if result.grounding_score is not None:
            print(
                f"GROUNDING:  {result.grounding_score:.3f} | "
                f"RELEVANCE: {result.relevance_score:.3f}"
            )
        if result.escalate:
            print("ESCALATE:   Yes — recommend human follow-up")
        print(f"ANSWER:\n{result.answer}")
