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
import hashlib
import json
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
from opentelemetry.trace import SpanKind, Status, StatusCode

from guardrails import GuardrailAction, GuardrailResult, check_input, check_output
from pii_scrubber import scrub
from telemetry import get_event_logger, get_tracer, init_tracing

load_dotenv()

logger = logging.getLogger(__name__)

# No-op unless POLICY_PAL_TRACE_FILE is set. Called at import so every entry
# point (Streamlit app, compare_models, __main__) picks it up unchanged.
init_tracing()
_tracer = get_tracer()
_event_logger = get_event_logger()

# ── Config ────────────────────────────────────────────────────────────────────
POLICIES_DIR = "policies/"
AUDIT_LOG    = "audit_log.csv"
AUDIT_RECORD = "audit_log.jsonl"
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

# ── Telemetry constants ───────────────────────────────────────────────────────
# gen_ai.provider.name is the current convention; gen_ai.system is its
# deprecated predecessor. Both are emitted for consumer compatibility.
_GENAI_PROVIDER = "anthropic"
_GENAI_AGENT    = "policy-pal"
_API_HOST       = "api.anthropic.com"

# Message content on spans is Opt-In in the GenAI conventions, and the spec warns
# it is "likely to contain sensitive information including user/PII data".
# Off unless POLICY_PAL_CAPTURE_CONTENT is set.
_CAPTURE_CONTENT = os.getenv(
    "POLICY_PAL_CAPTURE_CONTENT", ""
).strip().lower() in ("1", "true", "yes", "on")
# Where captured content goes. The GenAI conventions recommend events over span
# attributes so content can be retained and access-controlled separately, and
# the backends agree: Tempo truncates a span attribute at 2048 bytes mid-string
# with no marker, while Loki takes a 256 KB line and rejects an oversize one
# outright. "event" is the default for both reasons; "span" and "both" remain
# available for a Tempo-only setup.
_CONTENT_DESTINATION = os.getenv(
    "POLICY_PAL_CONTENT_DESTINATION", "event"
).strip().lower()
if _CONTENT_DESTINATION not in ("event", "span", "both"):
    logger.warning(
        "Unrecognised POLICY_PAL_CONTENT_DESTINATION %r; using 'event'.",
        _CONTENT_DESTINATION,
    )
    _CONTENT_DESTINATION = "event"
_CONTENT_TO_SPAN  = _CONTENT_DESTINATION in ("span", "both")
_CONTENT_TO_EVENT = _CONTENT_DESTINATION in ("event", "both")

# Two caps, because the two destinations have different ceilings. The span cap
# is set by Tempo's max_attribute_bytes; the event cap is set well inside Loki's
# 256 KB line limit with room for the envelope and both messages.
_CONTENT_MAX_CHARS       = int(os.getenv("POLICY_PAL_CONTENT_MAX_CHARS", "1800"))
_EVENT_CONTENT_MAX_CHARS = int(
    os.getenv("POLICY_PAL_EVENT_CONTENT_MAX_CHARS", "60000")
)

# gen_ai.client.inference.operation.details -- the event name from the GenAI
# conventions. Verified against the registry, not recalled.
_GENAI_EVENT_NAME = "gen_ai.client.inference.operation.details"

# How `confidence` is arrived at. It is the model's own assessment of how
# directly the retrieved policy text supports its answer -- a self-report about
# wording overlap and specificity between question and document, NOT a
# calibrated or externally validated probability. Recorded on every span that
# carries a confidence value so the caveat travels with the data.
_CONFIDENCE_METHOD = "llm_self_report"


def _user_hash(user: str) -> str:
    """
    Stable, non-reversible handle for correlating one user's requests.

    `user` is free text from a "Your name or employee ID" box, so it is PII in
    its own right -- separate from message content and not covered by the
    content-capture gate. Spans always carry the hash; the raw value is exported
    only when content capture is explicitly enabled, and is always kept in the
    audit record regardless.
    """
    return hashlib.sha256(user.encode("utf-8")).hexdigest()[:16]

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
    # ── Policy provenance (populated by parse_policy_metadata) ───────────────
    policy_generated_by: str = "manual"   # "council" | "manual"
    council_consensus:   str = "n/a"      # "unanimous" | "contested" | "unknown" | "n/a"
    compliance_flags:    int = 0          # count of compliance flags at generation time
    # ── Why the pipeline stopped (telemetry only — not written to the CSV) ────
    termination_reason: str = "completed"
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


# ── Policy metadata ───────────────────────────────────────────────────────────
def parse_policy_metadata(filepath: str) -> dict:
    """
    Read the first 10 lines of a .txt policy file and extract the Council
    header block fields used for provenance tracking.

    Expected header format (written by Council's save_policy_for_pal)::

        Generated by: Council (multi-agent boardroom)
        Topic: <topic>
        Date: <YYYYMMDD>
        Consensus: <unanimous|contested>
        Agents: Founder, HR, Finance, Legal
        Compliance Flags: <int>
        ============================================================

    Args:
        filepath: Absolute or relative path to the .txt policy file.

    Returns:
        Dict with keys:
          - ``policy_generated_by``: ``"council"`` or ``"manual"``
          - ``council_consensus``:   ``"unanimous"`` | ``"contested"`` |
                                     ``"unknown"`` | ``"n/a"``
          - ``compliance_flags``:    int (0 if absent or unreadable)
    """
    metadata: dict = {
        "policy_generated_by": "manual",
        "council_consensus":   "n/a",
        "compliance_flags":    0,
    }
    try:
        with open(filepath, encoding="utf-8") as fh:
            lines = [fh.readline() for _ in range(10)]
    except OSError as exc:
        logger.warning("parse_policy_metadata: could not read %s: %s", filepath, exc)
        return metadata

    for line in lines:
        stripped = line.strip()
        lower    = stripped.lower()

        if lower.startswith("generated by:"):
            value = stripped.split(":", 1)[1].strip().lower()
            if "council" in value:
                metadata["policy_generated_by"] = "council"

        elif lower.startswith("consensus:"):
            value = stripped.split(":", 1)[1].strip().lower()
            if value in ("unanimous", "contested"):
                metadata["council_consensus"] = value
            else:
                metadata["council_consensus"] = "unknown"

        elif lower.startswith("compliance flags:"):
            try:
                metadata["compliance_flags"] = int(stripped.split(":", 1)[1].strip())
            except (ValueError, IndexError):
                metadata["compliance_flags"] = 0

    # If file is council-generated but the consensus line was missing/unreadable
    if (
        metadata["policy_generated_by"] == "council"
        and metadata["council_consensus"] == "n/a"
    ):
        metadata["council_consensus"] = "unknown"

    return metadata


@functools.lru_cache(maxsize=None)
def load_policy_metadata(folder: str = POLICIES_DIR) -> dict[str, dict]:
    """
    Load provenance metadata for every .txt file in *folder*.

    Results are cached for the lifetime of the process — restart to reload
    after new policy files are added (same lifecycle as ``load_policies``).

    Args:
        folder: Directory to scan (default: ``POLICIES_DIR``).

    Returns:
        Mapping of ``{filename: metadata_dict}`` where each metadata dict
        contains the fields documented in :func:`parse_policy_metadata`.
    """
    metadata_map: dict[str, dict] = {}
    if not os.path.exists(folder):
        return metadata_map
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder, filename)
            try:
                metadata_map[filename] = parse_policy_metadata(filepath)
            except Exception as exc:
                logger.warning(
                    "load_policy_metadata: skipping %s: %s", filepath, exc
                )
    logger.info(
        "Loaded metadata for %d policy file(s) from %s",
        len(metadata_map), folder,
    )
    return metadata_map


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
    best_is_council = False

    for name, content in docs.items():
        content_lower = content.lower()
        score = sum(1 for word in question_words if word in content_lower)
        is_council = content_lower.startswith("generated by: council")
        # Prefer higher score; on a tie, prefer Council-generated files.
        if score > best_score or (score == best_score and is_council and not best_is_council):
            best_score = score
            best_chunk = PolicyChunk(source=name, content=content, score=score)
            best_is_council = is_council

    if best_chunk and best_chunk.score >= 1:
        return best_chunk
    return None


# ── GenAI message content ───────────────────────────────────────────────
def _messages_attr(role: str, text: str, limit: int | None = None) -> str:
    """
    Serialize one message into the shape gen_ai.{input,output}.messages expects:
    ``[{"role": ..., "parts": [{"type": "text", "content": ...}]}]``.

    OTel span attributes hold only primitives, so this is the JSON-string form
    the conventions permit when structured values are unsupported.

    Truncation is marked inline rather than silently applied -- a clipped
    transcript that does not say it was clipped is worse than no transcript.
    """
    limit = _CONTENT_MAX_CHARS if limit is None else limit
    if len(text) > limit:
        # Elide the middle, not the tail. The grounded prompt puts the policy
        # document in the middle and the employee's question at the very end,
        # so clipping the tail would drop the one part an auditor needs.
        #
        # The marker counts against the budget: without subtracting it, an input
        # just over the limit produces a LARGER attribute than leaving it alone.
        # Width is computed from `total` because both numbers are <= total.
        total  = len(text)
        marker = f"\n\u2026[elided {total} of {total} chars]\u2026\n"
        keep   = (limit - len(marker)) // 2
        if keep <= 0:
            # No room for any content. Note that text[-0:] returns the WHOLE
            # string, so this branch must not fall through to the slice below.
            text = f"\u2026[elided all {total} chars]\u2026"
        else:
            text = (
                text[:keep]
                + f"\n\u2026[elided {total - 2 * keep} of {total} chars]\u2026\n"
                + text[total - keep:]
            )
    return json.dumps(
        [{"role": role, "parts": [{"type": "text", "content": text}]}],
        ensure_ascii=False,
    )


# ── GenAI inference event ───────────────────────────────────────────────
def _emit_inference_event(
    *,
    request_id: str,
    model:      str,
    prompt:     str,
    response_text: str | None,
    attributes: dict,
) -> None:
    """
    Emit the GenAI inference event carrying message content.

    Emitted inside the active span, so the record picks up the trace and span id
    from context and joins back to the trace it describes -- content lives on the
    logs signal, correlation stays intact.

    Content is scrubbed before it gets here for the response, and upstream for
    the prompt. Failures are swallowed: telemetry must never break the request.
    """
    if not _CONTENT_TO_EVENT:
        return
    try:
        body = dict(attributes)
        # The OTLP `eventName` field does not survive the collector -> Loki hop
        # (verified by reading it back), so carry it as an attribute too. Cheap,
        # and without it the events are unfilterable by type in Loki.
        body["event.name"] = _GENAI_EVENT_NAME
        body["policy_pal.request_id"] = request_id
        body["gen_ai.input.messages"] = _messages_attr(
            "user", prompt, _EVENT_CONTENT_MAX_CHARS
        )
        if response_text is not None:
            body["gen_ai.output.messages"] = _messages_attr(
                "assistant", response_text, _EVENT_CONTENT_MAX_CHARS
            )
        _event_logger.emit(
            event_name=_GENAI_EVENT_NAME,
            body=f"{model} inference",
            attributes=body,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("[%s] Inference event emit failed: %s", request_id, exc)


# ── LLM error classification ──────────────────────────────────────────────────
def _classify_llm_error(exc: Exception) -> tuple[str, str]:
    """
    Map an exception from the provider SDK to (termination_reason, event_class).

    ``event_class`` separates infrastructure failures (our side / the network)
    from provider-side rejections (their side). A connection timeout and a rate
    limit both cause a retry, but only the latter is compliance-relevant, and
    the span structure alone can't tell them apart.
    """
    if isinstance(exc, anthropic.APITimeoutError):
        return "timeout", "infrastructure"
    if isinstance(exc, anthropic.APIConnectionError):
        return "connection_error", "infrastructure"
    if isinstance(exc, anthropic.RateLimitError):
        return "rate_limit", "provider_rejection"
    if isinstance(exc, anthropic.APIStatusError):
        return f"http_{exc.status_code}", "provider_rejection"
    return "api_error", "provider_rejection"


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
        backoff_s = 0.0

        # One CLIENT span per HTTP attempt, nested under the request span opened
        # by ask_policy. Span name per convention: "{operation} {request.model}".
        with _tracer.start_as_current_span(
            f"chat {model}",
            kind=SpanKind.CLIENT,
            # Errors are recorded manually below, with classification. Without
            # this, use_span records a second exception event on the way out and
            # overwrites the status description with the LLMError wrapper.
            record_exception=False,
            set_status_on_exception=False,
            attributes={
                "gen_ai.operation.name":       "chat",
                "gen_ai.provider.name":        _GENAI_PROVIDER,
                "gen_ai.system":               _GENAI_PROVIDER,   # deprecated alias
                "gen_ai.request.model":        model,
                "gen_ai.request.max_tokens":   MAX_TOKENS,
                "server.address":              _API_HOST,
                "policy_pal.request_id":       request_id,
                "policy_pal.llm.attempt":      attempt + 1,
                "policy_pal.llm.max_attempts": _MAX_RETRIES + 1,
            },
        ) as span:
            if _CAPTURE_CONTENT:
                # The question was PII-scrubbed before it reached this prompt,
                # so this is the redacted form as actually sent to the provider.
                # The span copy is capped to survive Tempo; the event copy below
                # carries the full text.
                span.set_attribute("policy_pal.content.captured", True)
                span.set_attribute(
                    "policy_pal.content.destination", _CONTENT_DESTINATION
                )
                if _CONTENT_TO_SPAN:
                    span.set_attribute(
                        "gen_ai.input.messages", _messages_attr("user", prompt)
                    )
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

                stop_reason = response.stop_reason or "unknown"
                span.set_attributes({
                    "gen_ai.response.id":             response.id,
                    "gen_ai.response.model":          response.model,
                    "gen_ai.response.finish_reasons": [stop_reason],
                    "gen_ai.usage.input_tokens":      response.usage.input_tokens,
                    "gen_ai.usage.output_tokens":     response.usage.output_tokens,
                    "policy_pal.llm.termination_reason": stop_reason,
                    # Anthropic's "refusal" stop reason is a provider-side
                    # decline, not an infrastructure failure.
                    "policy_pal.llm.event_class": (
                        "provider_rejection" if stop_reason == "refusal"
                        else "completed"
                    ),
                    "policy_pal.confidence":        confidence,
                    "policy_pal.confidence.method": _CONFIDENCE_METHOD,
                })
                if _CAPTURE_CONTENT:
                    # Unlike the input, the raw reply has not passed any PII
                    # filter yet -- the output guardrail runs later, in
                    # ask_policy. Scrub before it leaves the process.
                    out = scrub(full_response)
                    span.set_attribute(
                        "policy_pal.content.output_redacted", out.was_modified
                    )
                    if out.was_modified:
                        span.set_attribute(
                            "policy_pal.content.output_redactions", out.summary
                        )
                    if _CONTENT_TO_SPAN:
                        span.set_attribute(
                            "gen_ai.output.messages",
                            _messages_attr("assistant", out.scrubbed),
                        )
                    # Full content on the logs signal. Emitted inside this span,
                    # so it carries this trace and span id.
                    _emit_inference_event(
                        request_id=request_id,
                        model=model,
                        prompt=prompt,
                        response_text=out.scrubbed,
                        attributes={
                            "gen_ai.operation.name":       "chat",
                            "gen_ai.provider.name":        _GENAI_PROVIDER,
                            "gen_ai.request.model":        model,
                            "gen_ai.response.model":       response.model,
                            "gen_ai.response.id":          response.id,
                            "gen_ai.usage.input_tokens":   response.usage.input_tokens,
                            "gen_ai.usage.output_tokens":  response.usage.output_tokens,
                            "server.address":              _API_HOST,
                            "policy_pal.llm.attempt":      attempt + 1,
                            "policy_pal.content.output_redacted": out.was_modified,
                        },
                    )
                span.set_status(Status(StatusCode.OK))

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
                reason, event_class = _classify_llm_error(e)
                span.set_attributes({
                    "error.type":                        type(e).__qualname__,
                    "policy_pal.llm.termination_reason": reason,
                    "policy_pal.llm.event_class":        event_class,
                    "policy_pal.llm.will_retry":         attempt < _MAX_RETRIES,
                })
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                logger.warning(
                    "[%s] LLM transient error (attempt %d/%d): %s",
                    request_id, attempt + 1, _MAX_RETRIES + 1, e,
                )
                last_error = e
                if attempt < _MAX_RETRIES:
                    backoff_s = _RETRY_BASE_DELAY * (2 ** attempt)

            except anthropic.APIError as e:
                reason, event_class = _classify_llm_error(e)
                span.set_attributes({
                    "error.type":                        type(e).__qualname__,
                    "policy_pal.llm.termination_reason": reason,
                    "policy_pal.llm.event_class":        event_class,
                    "policy_pal.llm.will_retry":         False,
                })
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                logger.error("[%s] LLM non-retryable error: %s", request_id, e)
                raise LLMError(str(e)) from e

        # Backoff sits outside the span so span duration reflects the HTTP
        # attempt, not the wait between attempts.
        if backoff_s:
            time.sleep(backoff_s)

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
        # ── Provenance fields (Task 3) ────────────────────────────────────────
        "policy_generated_by": response.policy_generated_by,
        "council_consensus":   response.council_consensus,
        "compliance_flags":    response.compliance_flags,
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


# ── Audit record ──────────────────────────────────────────────────────────────
AUDIT_SCHEMA_VERSION = 1


def write_audit_record(
    *,
    response:          PolicyResponse,
    user:              str,
    scrubbed_question: str,
    model:             str,
) -> None:
    """
    Append the full audit record for one request as a JSON line.

    This is the system of record, distinct from the spans. Traces are for
    operations: they are sampled-by-design stores, rewritten by compaction,
    TTL'd well short of the retention a regulator expects, and -- as measured
    against this stack's Tempo -- they silently truncate long attributes
    mid-string. None of that is acceptable for evidence.

    So content here is scrubbed but NOT truncated: the 1800-character cap on
    spans exists to fit Tempo's max_attribute_bytes, and does not apply to a
    record store.

    JSONL stands in for the real target. In production: S3 with Object Lock in
    compliance mode, where retention is enforced by the storage layer rather
    than by convention, joined back to the traces on ``request_id``.

    Failures are logged, never raised -- audit must not break the request path.
    """
    record = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "timestamp":      response.timestamp,
        "request_id":     response.request_id,
        "user":           user,
        "outcome": {
            "termination_reason": response.termination_reason,
            "escalated":          response.escalate,
        },
        "question": {
            "scrubbed":     scrubbed_question,
            "pii_detected": response.pii_removed,
            "pii_summary":  response.pii_summary,
        },
        "answer": {
            "text":      response.answer,
            "delivered": response.guardrail_output_action != "BLOCK",
        },
        "retrieval": {
            "source": response.source,
            "score":  response.retrieval_score,
        },
        "confidence": {
            "value":  response.confidence,
            "reason": response.confidence_reason,
            # Not a calibrated probability -- see _CONFIDENCE_METHOD.
            "method": _CONFIDENCE_METHOD,
        },
        "guardrails": {
            "input_action":    response.guardrail_input_action,
            "output_action":   response.guardrail_output_action,
            "reason":          response.guardrail_reason,
            "grounding_score": response.grounding_score,
            "relevance_score": response.relevance_score,
        },
        "model": {
            "provider":      _GENAI_PROVIDER,
            "id":            model,
            "input_tokens":  response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost_usd":      response.cost_usd,
            "latency_ms":    round(response.latency_ms, 1),
        },
        "policy_provenance": {
            "generated_by":     response.policy_generated_by,
            "council_consensus": response.council_consensus,
            "compliance_flags": response.compliance_flags,
        },
    }
    try:
        with _audit_lock:
            with open(AUDIT_RECORD, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        logger.error(
            "Failed to write audit record [%s]: %s", response.request_id, e
        )


# ── Main entry point ──────────────────────────────────────────────────────────
def _run_pipeline(
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
    _source_meta: dict = {}   # populated when a matching policy chunk is found
    termination_reason = "completed"

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
            termination_reason = "guardrail_input_block"
        else:
            # Step 3: Retrieval
            docs            = load_policies(POLICIES_DIR)
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
                termination_reason = "no_retrieval"
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

                    # Extract provenance metadata for this source document
                    _policy_meta = load_policy_metadata(POLICIES_DIR)
                    _source_meta = _policy_meta.get(source, {})

                    # Step 5: Output guardrail
                    g_output        = check_output(answer, chunk.content, clean_question)
                    grounding_score = g_output.grounding_score
                    relevance_score = g_output.relevance_score

                    if g_output.action == GuardrailAction.BLOCK:
                        answer     = g_output.blocked_message or answer
                        confidence = "NONE"
                        escalate   = True
                        termination_reason = "guardrail_output_block"
                    elif g_output.action == GuardrailAction.WARN:
                        if g_output.modified_text:
                            answer = g_output.modified_text
                        confidence = "LOW"
                        escalate   = True
                        termination_reason = "guardrail_output_warn"
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
                    termination_reason = "llm_error"

        # Step 6: Cost
        cost_usd = calculate_cost(input_tokens, output_tokens, model)

    except Exception as e:
        logger.error(
            "[%s] Unexpected pipeline error: %s", request_id, e, exc_info=True
        )
        termination_reason = "pipeline_error"

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
        # Provenance fields derived from the source policy file's header block
        policy_generated_by=_source_meta.get("policy_generated_by", "manual"),
        council_consensus=_source_meta.get("council_consensus",   "n/a"),
        compliance_flags=_source_meta.get("compliance_flags",    0),
        termination_reason=termination_reason,
    )

    # Step 7: Audit log (non-fatal)
    write_audit_log(
        response=response,
        user=user,
        original_question=question,
        scrubbed_question=clean_question,
        pii_summary=scrub_result.summary,
    )
    write_audit_record(
        response=response,
        user=user,
        scrubbed_question=clean_question,
        model=model,
    )

    return response


# ── Request-boundary span ─────────────────────────────────────────────────────
def _annotate_request_span(span, response: PolicyResponse) -> None:
    """
    Copy pipeline outcome onto the request span.

    Fields with a GenAI convention equivalent use the convention name; the rest
    (confidence, retrieval score, guardrail actions, cost) have no equivalent in
    the spec and live under the ``policy_pal.*`` namespace rather than being
    forced into ``gen_ai.*``.
    """
    if not span.is_recording():
        return

    model_invoked = response.input_tokens > 0 or response.output_tokens > 0

    attributes: dict = {
        "policy_pal.request_id":              response.request_id,
        "policy_pal.termination_reason":      response.termination_reason,
        "policy_pal.model_invoked":           model_invoked,
        "policy_pal.escalated":               response.escalate,
        "policy_pal.confidence":              response.confidence,
        "policy_pal.confidence.method":       _CONFIDENCE_METHOD,
        "policy_pal.retrieval.score":         response.retrieval_score,
        "policy_pal.retrieval.source":        response.source,
        "policy_pal.guardrail.input.action":  response.guardrail_input_action,
        "policy_pal.guardrail.output.action": response.guardrail_output_action,
        "policy_pal.pii.detected":            response.pii_removed,
        "policy_pal.pii.summary":             response.pii_summary,
        "policy_pal.cost.usd":                response.cost_usd,
        "policy_pal.policy.generated_by":     response.policy_generated_by,
        "policy_pal.policy.council_consensus": response.council_consensus,
        "policy_pal.policy.compliance_flags": response.compliance_flags,
    }
    # Aggregate usage for the request. NOTE: also present on each child attempt
    # span — sum across all spans double-counts. Use parent-only or child-only.
    if model_invoked:
        attributes["gen_ai.usage.input_tokens"]  = response.input_tokens
        attributes["gen_ai.usage.output_tokens"] = response.output_tokens
    if response.confidence_reason:
        attributes["policy_pal.confidence_reason"] = response.confidence_reason
    if response.guardrail_reason:
        attributes["policy_pal.guardrail.reason"] = response.guardrail_reason
    if response.grounding_score is not None:
        attributes["policy_pal.grounding.score"] = response.grounding_score
    if response.relevance_score is not None:
        attributes["policy_pal.relevance.score"] = response.relevance_score

    span.set_attributes(attributes)

    # A guardrail block or an empty retrieval is a correct, intended outcome —
    # not a span error. Only genuine failures get ERROR status.
    if response.termination_reason in ("llm_error", "pipeline_error"):
        span.set_attribute("error.type", response.termination_reason)
        span.set_status(Status(StatusCode.ERROR, response.termination_reason))
    else:
        span.set_status(Status(StatusCode.OK))


def ask_policy(
    question: str,
    user:     str = "anonymous",
    model:    str = MODEL,
) -> PolicyResponse:
    """
    Answer a policy question, wrapped in an OTel request span.

    The span opens at the request boundary and always closes, including on paths
    that never reach the model (guardrail block, empty retrieval, pipeline
    error). The ``chat`` child span from :func:`ask_llm` appears only when the
    model was actually invoked — so the parent proves the request was recorded
    and the child's presence tells you whether an inference happened.
    """
    with _tracer.start_as_current_span(
        f"invoke_agent {_GENAI_AGENT}",
        kind=SpanKind.INTERNAL,
        attributes={
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.provider.name":  _GENAI_PROVIDER,
            "gen_ai.system":         _GENAI_PROVIDER,   # deprecated alias
            "gen_ai.agent.name":     _GENAI_AGENT,
            "gen_ai.request.model":  model,
            # user.hash, not user.id -- see _user_hash.
            "user.hash":             _user_hash(user),
        },
    ) as span:
        if _CAPTURE_CONTENT:
            span.set_attribute("user.id", user)
        try:
            response = _run_pipeline(question, user=user, model=model)
        except Exception as e:
            span.set_attributes({
                "error.type": type(e).__qualname__,
                "policy_pal.termination_reason": "unhandled_error",
                "policy_pal.model_invoked": False,
            })
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
        _annotate_request_span(span, response)
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
