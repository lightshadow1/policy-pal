"""
compare_models.py
Offline model comparison script — no Streamlit required.

Runs a fixed test suite against both Claude Haiku and Claude Sonnet in parallel,
prints a side-by-side report to stdout, and writes comparison_report.csv.

Usage:
    uv run python compare_models.py

The SONNET_MODEL env var overrides the default Sonnet model ID.
"""

import csv
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime

from dotenv import load_dotenv

from policy_engine import MODEL, SONNET_MODEL, ask_policy, PolicyResponse

load_dotenv()

logging.basicConfig(
    level=logging.WARNING,   # suppress INFO logs during comparison for cleaner output
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ── Models under test ─────────────────────────────────────────────────────────
MODELS: dict[str, str] = {
    "Haiku":  MODEL,
    "Sonnet": SONNET_MODEL,
}

# ── Test cases ────────────────────────────────────────────────────────────────
# Covers: normal queries, PII input, out-of-scope, low confidence, edge cases
TEST_CASES: list[tuple[str, str]] = [
    ("employee_1", "How many vacation days do I get per year?"),
    ("employee_2", "Can I work from home every day of the week?"),
    ("employee_3", "How do I submit an expense over $500?"),
    ("employee_4", "What happens during my performance review?"),
    ("employee_5", "My email is bob@company.com — am I allowed to work remotely?"),
    ("employee_6", "What is the capital of France?"),
    ("manager_1",  "What happens if an employee gets a rating of 1?"),
    ("employee_7", "Can I take vacation during my notice period?"),
]

COMPARISON_REPORT = "comparison_report.csv"


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class ModelResult:
    model_name:             str
    model_id:               str
    user:                   str
    question:               str
    answer:                 str
    confidence:             str
    grounding_score:        float | None
    relevance_score:        float | None
    latency_ms:             float
    input_tokens:           int
    output_tokens:          int
    cost_usd:               float
    guardrail_input_action: str
    guardrail_output_action: str
    guardrail_reason:       str
    escalate:               bool
    pii_removed:            bool
    request_id:             str
    retrieval_score:        int
    error:                  str = ""


# ── Execution ─────────────────────────────────────────────────────────────────

def run_model(
    question:   str,
    user:       str,
    model_name: str,
    model_id:   str,
) -> ModelResult:
    """Run a single question through the full pipeline for the given model."""
    try:
        result: PolicyResponse = ask_policy(question, user=user, model=model_id)
        return ModelResult(
            model_name=model_name,
            model_id=model_id,
            user=user,
            question=question,
            answer=result.answer,
            confidence=result.confidence,
            grounding_score=result.grounding_score,
            relevance_score=result.relevance_score,
            latency_ms=result.latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.cost_usd,
            guardrail_input_action=result.guardrail_input_action,
            guardrail_output_action=result.guardrail_output_action,
            guardrail_reason=result.guardrail_reason,
            escalate=result.escalate,
            pii_removed=result.pii_removed,
            request_id=result.request_id,
            retrieval_score=result.retrieval_score,
        )
    except Exception as e:
        return ModelResult(
            model_name=model_name,
            model_id=model_id,
            user=user,
            question=question,
            answer="",
            confidence="NONE",
            grounding_score=None,
            relevance_score=None,
            latency_ms=0.0,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            guardrail_input_action="PASS",
            guardrail_output_action="PASS",
            guardrail_reason="",
            escalate=True,
            pii_removed=False,
            request_id="",
            retrieval_score=0,
            error=str(e),
        )


def compare_all(
    test_cases:  list[tuple[str, str]],
    max_workers: int = 4,
) -> list[ModelResult]:
    """
    Execute all (test_case × model) combinations with bounded parallelism.
    Both models for the same case run in parallel.
    """
    tasks = [
        (question, user, model_name, model_id)
        for user, question in test_cases
        for model_name, model_id in MODELS.items()
    ]

    results: list[ModelResult] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(run_model, q, u, mn, mi): (q, mn)
            for q, u, mn, mi in tasks
        }
        for future in as_completed(future_map):
            q, mn = future_map[future]
            try:
                results.append(future.result())
            except Exception as e:
                print(f"  [ERROR] {mn} | {q!r}: {e}")

    return results


# ── Reporting ─────────────────────────────────────────────────────────────────

def _fmt_score(score: float | None) -> str:
    return f"{score:.2f}" if score is not None else " N/A"


def print_report(all_results: list[ModelResult]) -> None:
    """Print a formatted side-by-side comparison to stdout."""
    # Group results by question (preserve original TEST_CASES order)
    questions = [q for _, q in TEST_CASES]
    by_question: dict[str, list[ModelResult]] = {q: [] for q in questions}
    for r in all_results:
        if r.question in by_question:
            by_question[r.question].append(r)
        else:
            by_question.setdefault(r.question, []).append(r)

    model_names = list(MODELS.keys())
    sep = "=" * 72

    print(f"\n{sep}")
    print(f"  Policy Pal — Model Comparison Report")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    for name, mid in MODELS.items():
        print(f"  {name}: {mid}")
    print(f"  Cases: {len(questions)}")
    print(sep)

    for idx, question in enumerate(questions, 1):
        rows = {r.model_name: r for r in by_question.get(question, [])}
        print(f"\nCASE {idx}/{len(questions)}: {question}")
        print("-" * 72)
        print(
            f"  {'Model':<10} {'Conf':<6} {'Guard':<6} "
            f"{'Ground':>7} {'Relev':>6} {'Latency':>9} "
            f"{'Tokens':>12} {'Cost':>10}"
        )
        print(f"  {'-'*68}")

        for name in model_names:
            r = rows.get(name)
            if r is None:
                print(f"  {name:<10} (no result)")
                continue
            if r.error:
                print(f"  {name:<10} ERROR: {r.error}")
                continue

            guard = (
                f"{'BLK' if r.guardrail_input_action == 'BLOCK' else ''}"
                f"{'WRN' if r.guardrail_output_action == 'WARN' else ''}"
                or "OK"
            )
            tokens = f"{r.input_tokens}/{r.output_tokens}"
            print(
                f"  {name:<10} {r.confidence:<6} {guard:<6} "
                f"{_fmt_score(r.grounding_score):>7} "
                f"{_fmt_score(r.relevance_score):>6} "
                f"{r.latency_ms:>7.0f}ms "
                f"{tokens:>12} "
                f"${r.cost_usd:>8.5f}"
            )

        print()
        for name in model_names:
            r = rows.get(name)
            if r and not r.error:
                answer_preview = r.answer[:200].replace("\n", " ")
                print(f"  [{name}] {answer_preview}")
                if len(r.answer) > 200:
                    print(f"         ... ({len(r.answer)} chars total)")
            print()

    # Summary
    print(sep)
    print("  SUMMARY")
    print(f"  {'Model':<10} {'HIGH':>6} {'LOW':>5} {'NONE':>5} "
          f"{'Avg Grnd':>9} {'Avg Lat':>9} {'Total Cost':>12}")
    print(f"  {'-'*60}")
    for name in model_names:
        rows_for_model = [r for r in all_results if r.model_name == name and not r.error]
        if not rows_for_model:
            continue
        high  = sum(1 for r in rows_for_model if r.confidence == "HIGH")
        low   = sum(1 for r in rows_for_model if r.confidence == "LOW")
        none  = sum(1 for r in rows_for_model if r.confidence == "NONE")
        grnd  = [r.grounding_score for r in rows_for_model if r.grounding_score is not None]
        avg_g = sum(grnd) / len(grnd) if grnd else 0.0
        avg_l = sum(r.latency_ms for r in rows_for_model) / len(rows_for_model)
        total = sum(r.cost_usd for r in rows_for_model)
        print(
            f"  {name:<10} {high:>6} {low:>5} {none:>5} "
            f"{avg_g:>9.2f} {avg_l:>7.0f}ms {total:>11.5f}"
        )
    print(sep)


def write_comparison_csv(
    all_results: list[ModelResult],
    output_path: str = COMPARISON_REPORT,
) -> None:
    """Write all comparison rows to a CSV file for offline analysis."""
    if not all_results:
        return

    fieldnames = [
        "model_name", "model_id", "user", "question",
        "confidence", "grounding_score", "relevance_score",
        "latency_ms", "input_tokens", "output_tokens", "cost_usd",
        "guardrail_input_action", "guardrail_output_action", "guardrail_reason",
        "escalate", "pii_removed", "retrieval_score", "request_id",
        "answer_preview", "error",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "model_name":             r.model_name,
                "model_id":               r.model_id,
                "user":                   r.user,
                "question":               r.question,
                "confidence":             r.confidence,
                "grounding_score":        f"{r.grounding_score:.3f}" if r.grounding_score is not None else "",
                "relevance_score":        f"{r.relevance_score:.3f}" if r.relevance_score is not None else "",
                "latency_ms":             f"{r.latency_ms:.1f}",
                "input_tokens":           r.input_tokens,
                "output_tokens":          r.output_tokens,
                "cost_usd":               f"{r.cost_usd:.6f}",
                "guardrail_input_action": r.guardrail_input_action,
                "guardrail_output_action": r.guardrail_output_action,
                "guardrail_reason":       r.guardrail_reason,
                "escalate":               str(r.escalate),
                "pii_removed":            str(r.pii_removed),
                "retrieval_score":        r.retrieval_score,
                "request_id":             r.request_id,
                "answer_preview":         r.answer[:200].replace("\n", " "),
                "error":                  r.error,
            })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Running model comparison ({len(TEST_CASES)} cases × {len(MODELS)} models)...")
    print("This will make real API calls. Estimated cost: < $0.05\n")

    all_results = compare_all(TEST_CASES)
    print_report(all_results)
    write_comparison_csv(all_results)
    print(f"\nFull comparison saved to: {COMPARISON_REPORT}")
