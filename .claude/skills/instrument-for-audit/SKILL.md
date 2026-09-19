---
name: instrument-for-audit
description: Instrument an AI agent's codebase (not its existing traces) so the resulting OpenTelemetry traces can answer audit and compliance questions — which requests were handled, why each one ended the way it did, and what evidence exists for each claim. Use when adding or reviewing tracing/observability on an LLM agent specifically for audit, compliance, or evidence purposes; not a general OTel-setup or debugging-observability skill.
---

# Instrumenting an AI agent for audit evidence

Debugging instrumentation and audit instrumentation are different jobs. Debugging instrumentation
wants to know why the slow or broken request behaved that way — so it spans the expensive call,
the one with the latency and the tokens. Audit instrumentation has to answer a different question
first: *did every request get handled, and can you show what happened to each one?* That question
is decided by choices made in the first afternoon of instrumentation, and almost none of them are
obvious from the OTel docs. This skill is that afternoon's checklist.

**Origin.** Every step below traces to a specific thing that went wrong while instrumenting a real
agent (Policy Pal) and running the resulting traces through a compliance checker (also
self-built), then following the same spans through a collector into a real backend (Tempo). 21
lessons, 5 further defects an independent reviewer found in the same instrumentation afterward.
Read "What this skill is not," at the end, before treating any of this as a guarantee.

## Scope — what this skill will and will not do

**Will:**

- Turn a request-handling codebase into span structure that distinguishes decisions from
  inference attempts, including every path that never reaches the model (Steps 1–3).
- Catch attribute-naming drift against the live GenAI semantic-conventions registry, and require a
  Required / Recommended / Opt-In tier recorded against every attribute emitted (Step 4).
- Keep inferred or fabricated values out of the `gen_ai.*` namespace (Step 5).
- Build the privacy/content-capture gate from an exhaustive inventory of what leaves the process,
  not from memory of what the gate was designed to cover (Step 6).
- Specify boundary-value tests for any redaction or truncation logic, and what "correct" means for
  them (Step 7).
- Specify a round-trip verification pass against the real backend — not just the exporter's return
  value (Step 8).
- Produce, as a deliverable, an explicit written list of what the resulting trace set still cannot
  answer (Step 9), plus a short list of decisions to hand to whoever owns the compliance program
  rather than deciding them inline (see "Questions to surface").
- Ship worked, adaptable code examples for the patterns above (see "Worked examples").

**Will not:**

- **Certify compliance with any framework.** EU AI Act Article 12, SOC 2, ISO 42001, NIST AI RMF —
  this skill produces evidence-shaped traces and an honest account of their limits. Whether that
  evidence satisfies a specific legal or contractual obligation is a determination for whoever
  already reviews your controls, not for this checklist.
- **Choose your retention period, your system-of-record architecture, or your content-capture
  policy.** Those are organizational decisions this skill is built to surface, explicitly, to a
  human — never to default on your behalf. See "Questions to surface."
- **Touch application behavior.** Answers, guardrails, retrieval, business logic — untouched.
  Everything this skill drives is additive telemetry. If a step here would require changing what
  the system does rather than what it records, that's out of scope.
- **Guarantee the resulting instrumentation is defect-free**, even if every step is followed
  exactly. It's a checklist of known failure modes, derived from a real instrumentation that itself
  turned out to have five more defects an independent reviewer found after these lessons were
  already written down. See "What this skill is not."
- **Substitute for an independent review.** Following this skill is a reason to go get one, not a
  replacement for one — see Step 1's logic and the closing section.
- **Do your privacy/legal analysis for you.** It tells you to enumerate every field that leaves the
  process and gate it deliberately; it does not tell you what counts as personal data in your
  jurisdiction or your domain.

## How to use this skill

Work through the steps in order on the target codebase — this is a code-review-and-edit pass,
not something you run against an existing trace file. Steps 1–5 are what you instrument and how;
step 6 is the privacy gate; step 7 is testing what you built; step 8 is verifying it survives the
path to storage; step 9 is the deliverable you hand back, alongside a short list of decisions
(section "Questions to surface, never decide silently") that belong to whoever owns the compliance
program, not to whoever is holding the keyboard.

---

## Step 1 — Find both boundaries, separately

Every agent request has (at least) two boundaries, and audit instrumentation needs a span at each:

- **The request boundary** — where the system commits to handling a request at all. This exists
  even for requests that are refused, filtered, or answered from cache. It always exists, and it
  always closes.
- **The model boundary** — each individual attempt to call the model. This may not exist for a
  given request (see Step 2), and there may be more than one per request (retries).

Grep the codebase for the function that owns the whole request lifecycle — the one an external
caller invokes and that returns the final response — and mark it as the request-boundary span. Then
find every place that constructs a request to the model provider (SDK call, HTTP client, RPC stub)
and mark each attempt as a model-boundary span. Do this before writing any span code: if you can't
point to both boundaries in the source, you don't have enough information to instrument correctly
yet.

## Step 2 — Enumerate every path that terminates before inference

This is the step most instrumentation skips, and it's the one an auditor cares about most. Search
the request-handling path for every branch that can produce a response *without* reaching the
model boundary:

- guardrail / policy blocks (input rejected before generation)
- empty or failed retrieval (nothing to ground an answer in)
- validation failures (malformed input, length limits)
- cache hits (a previous answer is being replayed)
- rate limiting or circuit breakers
- any other early return on the request path

In the reference instrumentation, these paths were 4 of 11 requests — 36% of traffic that a
call-site-only instrumentation would have silently dropped from the record, while a compliance
checker still reported the recording checks as `met`. A policy assistant declining to answer is
more audit-relevant than one answering well: it's the strongest evidence that the system enforces
the restraint it claims to. If you only instrument the successful path, you've built a system that
can't show its own restraint.

Every path found here gets its own outcome on the request-boundary span (see Step 3) — it does not
need its own model-boundary span, because no model call happened.

## Step 3 — Structure spans as one parent per request, one child per attempt

```
invoke_agent <agent>          <- blocked . 1 span, no child
invoke_agent <agent>          <- normal . 2 spans
 \-- chat <model>
invoke_agent <agent>          <- retried once . 3 spans
 |-- chat <model>              connection_error
 \-- chat <model>              end_turn
invoke_agent <agent>          <- retries exhausted . 4 spans
 |-- chat <model>              connection_error
 |-- chat <model>              connection_error
 \-- chat <model>              connection_error
```

One parent span per request (opened at the request boundary from Step 1, always closed, regardless
of outcome). One child span per model-call attempt (opened at the model boundary, one per HTTP/RPC
attempt including retries). The nesting itself carries information a flat span list cannot: parent
count is decisions, child count is inference attempts, and children-per-parent is retry behavior —
all recoverable later without having had to name them as a metric in advance. Flattened, two
attempts at one decision are indistinguishable from two separate decisions, which is exactly the
ambiguity a compliance count must not contain.

Record *why* each child span (attempt) terminated, as an explicit attribute value — not just
success/failure. A timeout and a provider-side refusal both trigger a retry; only one of them is
compliance-relevant, and a boolean can't tell them apart. Give the request-boundary (parent) span
its own outcome/termination attribute too, covering the paths from Step 2 as first-class values —
e.g. an enum something like `{completed, guardrail_input_block, guardrail_output_block,
no_retrieval, validation_failed, cache_hit, llm_error, rate_limited, unhandled_error}` — namespaced
under your own attribute prefix (see Step 5), not invented inside `gen_ai.*`.

## Step 4 — Check attribute names against the current spec, not memory, and tier what you emit

GenAI semantic-convention attribute names move. As of this writing:

| Old name | Status | Current name |
|---|---|---|
| `gen_ai.system` | deprecated | `gen_ai.provider.name` |
| `gen_ai.prompt` | removed | `gen_ai.input.messages` |
| `gen_ai.completion` | removed | `gen_ai.output.messages` |

**Do not trust this table either — it will go stale.** Before emitting or reading any `gen_ai.*`
attribute, check it against the live registry
(https://opentelemetry.io/docs/specs/semconv/gen-ai/) or a version you've deliberately vendored and
pinned. Wrong names don't error. They silently produce spans that a checker will read as absent —
a system recording inputs correctly under the current spec can still score zero against a checker
built on stale names, and there is no exception or warning anywhere in that failure path.

Also specifically check for `gen_ai.operation.name` — it is Required on every GenAI span, and it's
the attribute that holds values like `invoke_agent` and `chat`, which is exactly what a consumer
needs to tell an agent-level span from a model-call span. A classifier that skips it and infers
kind from the *presence* of a model attribute instead will misclassify every agent span as an LLM
span, because agent spans are supposed to carry model attributes too (a model is selected at the
agent level). This is not a hypothetical: it's the single largest failure mode found in the
reference project's own checker, and it made every agent-boundary span unreachable by that tool's
"was this event recorded" check — 0% on a trace file that contained the event 11 times.

For every attribute you emit, record which requirement tier it falls in — **Required**,
**Recommended**, or **Opt-In** — per the current spec. This tier is not decoration: it's what
determines whether the absence of an attribute is a gap in your instrumentation or a deliberate,
spec-sanctioned choice. Message content (`gen_ai.input.messages` / `gen_ai.output.messages`) is
Opt-In specifically because it routinely carries personal data — treat "we don't emit it" as a
valid, tier-justified answer, not a defect to silently fix. Produce this tiering as an explicit
list or table alongside the instrumentation, not just in your head: it's the artifact that lets
someone answer "what can this trace prove" without re-deriving it from the code. See "Worked
examples" for a filled-in tiering table.

## Step 5 — Never invent a convention attribute to fill a gap

When something you need to record has no `gen_ai.*` attribute — a refusal reason, a block cause, an
internal confidence score — put it under your own namespaced prefix (`<yourapp>.*`), never inside
`gen_ai.*`. A namespaced custom attribute is honest about its provenance. A `gen_ai.*` attribute
holding a value you inferred, defaulted, or backfilled will be treated by every downstream
consumer as an observed fact from the provider, because that's what the namespace promises.

This cuts both ways: if the application never sent a parameter — a temperature, a top-p, a system
prompt override — do not emit the provider's default value for it. Emitting a default you never
actually sent fabricates a request parameter that didn't exist. Absence is information; don't
paper over it.

## Step 6 — Enumerate every field that leaves the process, then gate against that list

Before deciding what's behind a privacy/content-capture flag, build the list the other way around:
grep every call site that sets a span attribute (`set_attribute`, `set_attributes`, and equivalents
in your SDK), and for each one, write down what value it holds, where that value comes from, and
whether it can identify a person. Only after that full list exists, check it against whatever gate
or flag you have. Do not rely on your memory of what the flag was designed to cover — that's
exactly the gap that produces a defect like the one below.

**The concrete failure this step exists to catch:** content capture was gated behind an
explicit opt-in flag, and the module docstring said so. A separate field — a free-text "your name
or employee ID" input, feeding a `user.id`-shaped attribute — was set unconditionally on every
span, because whoever designed the flag was thinking about prompt content, not identity, and
identity arrived through a different code path. Merely turning tracing on (a far lower bar than
turning content capture on) shipped real names and employee IDs to the trace file and over the wire
to a collector, on every single request, silently.

**The pattern, not just the patch:** always emit a hashed or otherwise pseudonymous identifier
unconditionally (e.g. `user.hash`) so per-user correlation survives tracing being on at all. Emit
the raw identifier (`user.id`) only behind its own explicit opt-in — a separate decision from
content capture, because identity and free-text content are different risk classes with different
legitimate consumers. A durable audit-of-record store (see "Questions to surface" below) can and
should keep the real value; the trace pipeline, which fans out to more consumers and more
infrastructure, should not carry it by default. See "Worked examples" for the code shape.

## Step 7 — Test redaction and truncation at boundary values, and assert on what survives

Any redaction, truncation, or size-limiting function applied before content leaves the process
needs tests at 0, 1, exactly-the-limit, one-over-the-limit, and far-over-the-limit — not just a
mid-range "normal" input. Assert on the actual string that comes out, never on the label the
function attaches to its own output. Two specific traps, both real, both silent:

- **A slice that quietly returns everything.** A truncation budget computed as `keep = limit // 2`
  reaches 0 when the limit is 0 or 1. In Python (and several other languages with the same
  negative-index slicing convention), `text[-0:]` is `text[0:]` — the whole string, not an empty
  one. The function can label its own output `"elided N of N chars"` while returning all N of
  them unchanged. An operator who lowers the limit specifically to reduce exposure gets zero
  reduction, confidently mislabeled as total elision.
- **A limit that doesn't budget for its own bookkeeping.** If truncation appends a marker like
  `"...[elided N chars]..."`, that marker consumes space too. If the size check happens before adding
  the marker, the "truncated" output can come out *longer* than the untruncated original for
  inputs near the boundary — the exact inputs the limit exists to catch. Budget the marker's
  worst-case width out of the limit before slicing, and test inputs just over the limit
  specifically; far-over inputs will pass while the boundary case is silently broken.

The failure mode in both cases is worse than doing nothing: no redaction leaves you knowing the
data is exposed, while a redaction that silently fails leaves you believing it isn't, with a label
that says so. See "Worked examples" for a truncation function that budgets its own marker.

## Step 8 — Round-trip verify against the real backend, not just the exporter

A span your exporter accepted is not necessarily a span your backend stored intact. After wiring up
a real path — collector, backend, whatever sits downstream — read spans back out through the
backend's own query path, re-parse any structured (JSON-encoded) attribute values, and assert they
round-trip. Concrete things this step has caught:

- A tracing backend silently truncating attribute values at a byte limit mid-string, with no error
  and no truncation marker — corrupting JSON-encoded attributes into payloads that are "stored"
  (the write succeeded) but not parseable. A large fraction of records in the reference run were
  unparseable this way, and nothing in the write path, the collector, or the query path reported
  a problem *on those paths*. The backend did expose a truncation counter — it simply wasn't
  scraped, so nothing surfaced it to a human. Before concluding a failure is silent, look for the
  counter and check it's actually collected; a metric nobody scrapes is not an alert. If you emit
  structured data as a JSON string in an attribute, size it against the backend's actual
  per-attribute limit, and verify by reading it back — not by trusting the export call's return
  value. Better still, don't put it in a span attribute: the GenAI conventions recommend capturing
  message content as **events** on the logs signal precisely so it can be stored, retained and
  access-controlled separately from traces, and log backends typically allow far larger payloads
  and reject oversize records instead of truncating them into corruption.
- A collector processor (e.g. one that stamps host/resource identity) applied on a pipeline whose
  only receiver is remote OTLP — so it overwrites or adds resource attributes describing the
  *collector's* host onto spans that originated somewhere else entirely, with nothing in the
  record indicating the spans were touched. Check processor scope against pipeline origin, not
  just against processor documentation; split pipelines by origin if a processor's correctness
  depends on where the signal actually came from.
- A deploy that reports healthy without having actually taken effect — e.g. a container
  orchestrator that only recreates a container when its *definition* changes, not when a
  bind-mounted config file's *contents* change, so a corrected config is "deployed" while the old
  process keeps running unchanged. A health check and an acceptance test can both pass against the
  stale config. Any check meant to confirm instrumentation changes are live should assert against
  something that actually changes with the deploy — a build identifier or config hash that shows
  up in the emitted spans themselves — not just "something is responding and looks fine."

A test that could catch any of these three must run from outside the system whose identity or
config it's checking — a self-test that shares the property being verified (e.g., one that
generates its own test spans from the collector's own host) will pass whether or not the bug is
present, because it can never observe the mismatch.

## Step 9 — Close by stating what the trace set still cannot answer

The deliverable of this pass is not a clean bill of health. End it with an explicit, written list
of questions the resulting traces cannot support — handed to whoever will next be asked "can you
prove this?" Prompts worth running through, concretely:

- If two different people (or two different tool versions) evaluate the same trace file against
  the same question, do they get the same answer? If not, what you have is an opinion, not
  evidence.
- Can the trace set be shown to be *complete* with respect to requests actually handled — i.e., is
  there an independent count (an access log, an audit table) it can be checked against? A checker
  that only ever sees the file it's handed cannot itself detect that failures were filtered out
  before it ran, and will score an incomplete file as good or better than a complete one.
- Is there a claim resting on this trace set that depends on something the *storage layer* doesn't
  actually enforce — e.g. "this could not have been altered" where nothing prevents alteration?
  Tracing backends are generally built for investigating a request you're already looking at:
  sampled, subject to compaction, retained on a schedule shorter than most regulatory retention
  windows. A claim of immutability or of durable retention needs to point at a store that actually
  provides those properties, not at the trace pipeline.
- Every ratio or coverage number produced downstream of this instrumentation should name its unit
  at the point of display — "5 of 10 inference attempts," never "50% of spans." A denominator that
  can silently change when classification logic changes (see Step 4) is not a property of the
  system being measured, and a number that can't say what it counted is not a measurement.

## Questions to surface, never decide silently

These are deployment decisions, not instrumentation decisions. Flag each one explicitly to whoever
owns the compliance program; do not pick a default and move on.

- **Trace retention.** This is a backend setting, and default retention on most tracing backends
  is short — often far shorter than a regulatory retention requirement. Someone needs to set it
  deliberately and record what was set, not inherit whatever the backend ships with.
- **Whether a durable system of record is needed, and where it lives.** Traces are operational
  telemetry: they're built to expire, get sampled, and get rewritten by compaction. If a durable,
  tamper-evident record is required, it needs to be a separate store — joined to the trace set on
  a request id — not a longer retention setting on the trace backend itself.
- **Whether to enable content capture at all.** Never flip this on as a side effect of fixing
  something else, or because a checker scored its absence as a gap (see the aside in Step 4 about
  Opt-In tiers). State the data-minimization trade-off explicitly, in the same conversation as the
  decision, before it's made.

## Constraints to hold regardless of the target codebase

- Tracing is **off by default**, enabled only via explicit configuration (an env var or equivalent)
  — never on by default, never enabled as a side effect of another feature.
- **Local file export stays available alongside any collector endpoint**, not replaced by it. The
  file is the reproducible artifact — something that can be diffed, re-run, and handed to someone
  without needing the backend to be reachable. The backend is the human-verifiable, queryable view.
  Keep both; each is doing a job the other doesn't.
- **Never enable message-content capture without flagging the data-minimization trade-off first**,
  as its own explicit statement, not buried in a changelog line.
- **Keep any existing audit log or system of record working**, unchanged, by anything in this pass.
  Spans are additive telemetry, not a replacement for whatever record-keeping already exists.

## Worked examples

These are adaptable sketches, not a library to import — the point is the shape, in whatever
OTel SDK and language the target codebase actually uses.

### Example 1 — the two-boundary skeleton (Steps 1, 2, 3, 5, 6)

```python
def handle_request(user_id: str, question: str) -> Response:
    with tracer.start_as_current_span(
        "invoke_agent my_agent",
        attributes={
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": "my_agent",
            "user.hash": hash_user(user_id),      # unconditional -- Step 6
        },
    ) as request_span:

        if reason := guardrail.check_input(question):
            request_span.set_attribute("myapp.termination_reason", "guardrail_input_block")
            request_span.set_attribute("myapp.guardrail.reason", reason)
            return Response.blocked(reason)

        context = retrieve(question)
        if not context:
            request_span.set_attribute("myapp.termination_reason", "no_retrieval")
            return Response.escalate()

        last_reason = "llm_error"
        for attempt in range(1, MAX_ATTEMPTS + 1):
            with tracer.start_as_current_span(
                "chat my-model",
                attributes={
                    "gen_ai.operation.name": "chat",
                    "gen_ai.provider.name": "my-provider",
                    "gen_ai.request.model": MODEL_NAME,
                    "myapp.llm.attempt": attempt,
                },
            ) as attempt_span:
                try:
                    result = call_model(context, question)
                except TransientError as exc:
                    last_reason = classify(exc)          # e.g. "connection_error", "rate_limited"
                    attempt_span.set_attribute("myapp.llm.termination_reason", last_reason)
                    attempt_span.record_exception(exc)
                    continue
                attempt_span.set_attribute("myapp.llm.termination_reason", "ok")
                attempt_span.set_attribute("gen_ai.usage.input_tokens", result.input_tokens)
                attempt_span.set_attribute("gen_ai.usage.output_tokens", result.output_tokens)
                request_span.set_attribute("myapp.termination_reason", "completed")
                return Response.ok(result)

        request_span.set_attribute("myapp.termination_reason", last_reason)
        return Response.escalate()
```

Note what each failed attempt does *not* have: token counts. It never received a response, so it
has none to report — see Step 8's note on coverage ratios excluding the structurally impossible.

### Example 2 — a filled-in attribute tiering table (Step 4)

| Attribute | Tier | Emitted? | Notes |
|---|---|---|---|
| `gen_ai.operation.name` | Required | always | |
| `gen_ai.provider.name` | Required | always | |
| `gen_ai.request.model` | Recommended | always | |
| `gen_ai.usage.input_tokens` / `output_tokens` | Recommended | on successful attempts only | absent on failed attempts by construction, not a gap |
| `gen_ai.input.messages` / `output.messages` | Opt-In | only if `MYAPP_CAPTURE_CONTENT=1` | data-minimization trade-off — see "Questions to surface" |
| `user.hash` | custom (`myapp.*`-equivalent) | always | pseudonymous, survives tracing being on at all |
| `user.id` | custom, gated separately | only if `MYAPP_CAPTURE_IDENTITY=1` | independent flag from content capture — see Step 6 |
| `myapp.termination_reason` | custom | always | first-class values, not booleans — see Step 3 |

### Example 3 — the privacy-gate pattern (Step 6)

```python
attrs = {"user.hash": hash_user(user_id)}          # never gated

if os.getenv("MYAPP_CAPTURE_IDENTITY"):            # separate flag from content capture
    attrs["user.id"] = user_id

if os.getenv("MYAPP_CAPTURE_CONTENT"):
    attrs["gen_ai.input.messages"] = format_messages(truncate_for_span(question, CONTENT_MAX_CHARS))

span.set_attributes(attrs)
```

Two flags, not one. `user.id` and message content are different risk classes with different
legitimate consumers, and collapsing them into a single "tracing is on" gate is exactly how a name
field shipped unconditionally in the reference project.

### Example 4 — a boundary-safe truncation function (Step 7)

```python
def truncate_for_span(text: str, limit: int) -> str:
    """Elide the middle of text to fit within limit, budgeting the marker itself."""
    if len(text) <= limit:
        return text
    marker = f"...[elided {len(text) - limit} chars]..."
    if limit <= len(marker):
        return marker[:limit] if limit > 0 else ""
    budget = limit - len(marker)
    keep = budget // 2
    if keep <= 0:
        return marker
    return f"{text[:keep]}{marker}{text[-keep:]}"


# Assert on what survives -- not on the function's own label -- at every boundary:
assert truncate_for_span("", 100) == ""
assert truncate_for_span("hello", 100) == "hello"
assert truncate_for_span("hello world", 0) == ""
assert len(truncate_for_span("x" * 50, 10)) <= 10
assert "x" * 50 not in truncate_for_span("x" * 5000, 1800)     # the sentinel must not survive
for n in range(0, 20):                                          # boundary sweep, not one sample
    out = truncate_for_span("y" * 5000, n)
    assert len(out) <= max(n, 0)
```

The `text[-0:]` trap and the marker-not-budgeted trap (Step 7) both fail this test at `limit=0`
and at `limit` just above the marker's own width — which is exactly why the sweep matters more
than a single hand-picked test case.

## What this skill is not

This is a checklist of traps someone already walked into, not a certification or a guarantee of
correctness. The instrumentation these steps are derived from was itself found, by an independent
reviewer, to contain five further defects after the lessons above were already written down —
including the exact `user.id` leak described in Step 6. Three of those five defects were, precisely,
the failure modes this checklist already named: a redaction that reported success while doing
nothing, a gate whose name overstated its coverage, and an init function that logged success while
changing nothing. Knowing a failure mode in the abstract did not prevent shipping it days later, in
the same codebase, by the same person who wrote the checklist.

The corrective this implies is not "be more careful." It's Step 1's logic applied recursively: grade
this instrumentation, too, against something — or someone — other than the person who wrote it.
Following every step above is a reason to look harder for what's still wrong, not a reason to stop
looking.
