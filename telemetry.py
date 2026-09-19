"""
telemetry.py
Opt-in OpenTelemetry tracing for Policy Pal.

Emits OTel GenAI semantic-convention spans and writes them to a local file as
OTLP/JSON lines (one ``ExportTraceServiceRequest`` JSON object per line — the
same shape the OTel Collector's ``fileexporter`` produces).

Two independent, opt-in exporters:

``POLICY_PAL_TRACE_FILE``
    Append OTLP/JSON lines to a local file. This is what ``agentaudit report``
    consumes, and it never touches the network.

``POLICY_PAL_OTLP_ENDPOINT``
    POST spans to an OTLP/HTTP collector, e.g. ``http://your-collector-host:4318``.
    The ``/v1/traces`` path is appended if absent.

Both may be set at once; each gets its own BatchSpanProcessor. With neither set
no TracerProvider is installed and ``opentelemetry.trace.get_tracer`` hands back
the default no-op tracer, so the instrumentation in policy_engine.py costs
nothing and changes no behaviour.

Message content travels on the **logs** signal, not on spans. The GenAI
conventions say instrumentations "MAY capture user inputs sent to the model and
responses received from it as events", and recommend it over span attributes so
content can be stored, retained and access-controlled independently of traces.
The practical argument is the same direction: this stack's Tempo caps a span
attribute at 2048 bytes and truncates mid-string with no marker, corrupting
JSON-encoded content into records that still look stored, while Loki accepts a
256 KB line and *rejects* an oversize one rather than silently cutting it.

So a LoggerProvider is installed alongside the TracerProvider, sharing the same
enable/disable switches and the same two destinations (local file, OTLP). Events
are emitted inside the active span, so each carries the trace and span id and
joins back to the trace it describes.
"""

import atexit
import base64
import json
import logging
import os
import threading
from typing import Sequence

from opentelemetry import trace
from opentelemetry._logs import get_logger, set_logger_provider
from opentelemetry.exporter.otlp.proto.common._internal._log_encoder import (
    encode_logs,
)
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.sdk._logs import LoggerProvider, ReadableLogRecord
from opentelemetry.sdk._logs.export import (
    BatchLogRecordProcessor,
    LogRecordExporter,
    LogRecordExportResult,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
TRACE_FILE_ENV    = "POLICY_PAL_TRACE_FILE"
EVENT_FILE_ENV    = "POLICY_PAL_EVENT_FILE"
OTLP_ENDPOINT_ENV = "POLICY_PAL_OTLP_ENDPOINT"
ENVIRONMENT_ENV   = "POLICY_PAL_ENVIRONMENT"
SERVICE_NAME      = "policy-pal"

# Instrumentation scope — identifies these spans as hand-rolled, not from an
# upstream auto-instrumentation package.
INSTRUMENTATION_NAME    = "policy_pal.telemetry"
INSTRUMENTATION_VERSION = "0.1.0"

# ── Module-level shared state ─────────────────────────────────────────────────
_init_lock: threading.Lock = threading.Lock()
_provider: TracerProvider | None = None
_logger_provider: LoggerProvider | None = None

# OpenTelemetry allows set_tracer_provider() once per process; later calls are
# ignored with a warning. Tracked so a re-init after shutdown fails loudly
# instead of returning a provider that get_tracer() will never resolve to.
_global_provider_set: bool = False

# Byte-valued OTLP fields that must be hex, not base64, in OTLP/JSON.
_HEX_FIELDS = frozenset({"traceId", "spanId", "parentSpanId"})


# ── OTLP/JSON file exporter ───────────────────────────────────────────────────
def _b64_to_hex(value: str) -> str:
    """Convert a protobuf-JSON base64 bytes field to a lowercase hex string."""
    return base64.b64decode(value).hex()


def _hexify_ids(node: object) -> object:
    """
    Recursively rewrite traceId/spanId/parentSpanId from base64 to hex.

    The standard Protobuf JSON mapping base64-encodes ``bytes`` fields, but the
    OTLP spec explicitly overrides that: "The traceId and spanId byte arrays are
    represented as case-insensitive hex-encoded strings; they are not
    base64-encoded as is defined in the standard Protobuf JSON Mapping."
    """
    if isinstance(node, dict):
        return {
            key: (
                _b64_to_hex(value)
                if key in _HEX_FIELDS and isinstance(value, str)
                else _hexify_ids(value)
            )
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [_hexify_ids(item) for item in node]
    return node


class OTLPJSONFileSpanExporter(SpanExporter):
    """
    Serialize spans to OTLP/JSON and append them to a local file, one JSON
    object per line.

    Uses the official protobuf encoder from opentelemetry-exporter-otlp-proto-http
    so the wire shape is generated by the same code path as a real OTLP export,
    then converts protobuf -> dict -> JSON with the trace/span ID encoding fixed
    up per the OTLP spec.
    """

    def __init__(self, filepath: str) -> None:
        self._filepath = filepath
        self._lock     = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if not spans:
            return SpanExportResult.SUCCESS
        try:
            # Local import: protobuf is pulled in by the OTLP exporter package.
            from google.protobuf.json_format import MessageToDict

            request = encode_spans(spans)
            payload = _hexify_ids(MessageToDict(request))
            line    = json.dumps(payload, separators=(",", ":"))

            with self._lock:
                with open(self._filepath, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            return SpanExportResult.SUCCESS
        except Exception as exc:  # noqa: BLE001 — export must never crash the app
            logger.error("Span export failed: %s", exc, exc_info=True)
            return SpanExportResult.FAILURE

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True

    def shutdown(self) -> None:
        return None


class OTLPJSONFileLogExporter(LogRecordExporter):
    """
    The log-signal twin of :class:`OTLPJSONFileSpanExporter`.

    Same OTLP/JSON-lines shape, same hex-id fixup, a separate file. Keeping a
    local file alongside the collector matters more here than for spans: this
    file is the reproducible artifact -- diffable, re-runnable, and readable
    without the backend being up.
    """

    def __init__(self, filepath: str) -> None:
        self._filepath = filepath
        self._lock     = threading.Lock()

    def export(self, batch: Sequence[ReadableLogRecord]) -> LogRecordExportResult:
        if not batch:
            return LogRecordExportResult.SUCCESS
        try:
            from google.protobuf.json_format import MessageToDict

            request = encode_logs(batch)
            payload = _hexify_ids(MessageToDict(request))
            line    = json.dumps(payload, separators=(",", ":"))

            with self._lock:
                with open(self._filepath, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            return LogRecordExportResult.SUCCESS
        except Exception as exc:  # noqa: BLE001 -- export must never crash the app
            logger.error("Log export failed: %s", exc, exc_info=True)
            return LogRecordExportResult.FAILURE

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True

    def shutdown(self) -> None:
        return None


# ── Setup ─────────────────────────────────────────────────────────────────────
def _build_resource() -> Resource:
    """Resource attributes identifying this service to the collector."""
    attributes: dict[str, str] = {"service.name": SERVICE_NAME}
    environment = os.getenv(ENVIRONMENT_ENV, "").strip()
    if environment:
        attributes["deployment.environment.name"] = environment
    return Resource.create(attributes)


def init_tracing() -> bool:
    """
    Install a TracerProvider with whichever exporters are configured.

    Returns False (and installs nothing) when neither ``POLICY_PAL_TRACE_FILE``
    nor ``POLICY_PAL_OTLP_ENDPOINT`` is set. Safe to call from every entry point.
    """
    global _provider, _logger_provider, _global_provider_set

    trace_file    = os.getenv(TRACE_FILE_ENV, "").strip()
    otlp_endpoint = os.getenv(OTLP_ENDPOINT_ENV, "").strip()
    if not trace_file and not otlp_endpoint:
        return False

    with _init_lock:
        if _provider is not None:
            return True
        if _global_provider_set:
            # A provider was installed and later shut down. Installing another
            # would build it, report success, and leave get_tracer() resolving
            # through the dead one -- spans created against stopped exporters.
            logger.error(
                "Tracing cannot be re-initialised after shutdown in this "
                "process; OpenTelemetry allows one global TracerProvider. "
                "Restart the process to resume tracing."
            )
            return False

        provider = TracerProvider(resource=_build_resource())
        enabled: list[str] = []

        if trace_file:
            provider.add_span_processor(
                BatchSpanProcessor(OTLPJSONFileSpanExporter(trace_file))
            )
            enabled.append(f"file={trace_file}")

        if otlp_endpoint:
            # Local import so the module has no hard dependency on the network
            # exporter when only file export is in use.
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            url = otlp_endpoint.rstrip("/")
            if not url.endswith("/v1/traces"):
                url = f"{url}/v1/traces"
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=url)))
            enabled.append(f"otlp={url}")

        # Logs signal: same switches, same destinations, separate provider.
        # set_logger_provider() has no one-time-only restriction, but it is
        # installed under the same guard so the two signals cannot diverge.
        log_provider = LoggerProvider(resource=_build_resource())
        event_file = os.getenv(EVENT_FILE_ENV, "").strip()
        if not event_file and trace_file:
            # Derived rather than required: one env var fewer, and the pairing
            # stays obvious on disk (traces.jsonl / traces.events.jsonl).
            base, _, ext = trace_file.rpartition(".")
            event_file = f"{base}.events.{ext}" if base else f"{trace_file}.events"
        if event_file:
            log_provider.add_log_record_processor(
                BatchLogRecordProcessor(OTLPJSONFileLogExporter(event_file))
            )
            enabled.append(f"events={event_file}")

        if otlp_endpoint:
            from opentelemetry.exporter.otlp.proto.http._log_exporter import (
                OTLPLogExporter,
            )

            log_url = otlp_endpoint.rstrip("/")
            if not log_url.endswith("/v1/logs"):
                log_url = f"{log_url}/v1/logs"
            log_provider.add_log_record_processor(
                BatchLogRecordProcessor(OTLPLogExporter(endpoint=log_url))
            )
            enabled.append(f"otlp-logs={log_url}")

        trace.set_tracer_provider(provider)
        if trace.get_tracer_provider() is not provider:
            logger.error(
                "Global TracerProvider was already set by another component; "
                "Policy Pal spans will not be exported."
            )
            return False
        set_logger_provider(log_provider)
        _provider = provider
        _logger_provider = log_provider
        _global_provider_set = True

    atexit.register(shutdown_tracing)
    logger.info("Tracing enabled - %s", ", ".join(enabled))
    return True


def get_tracer() -> trace.Tracer:
    """
    Return the Policy Pal tracer.

    When ``init_tracing`` has not installed a provider this is the global no-op
    tracer, so callers never need to branch on whether tracing is on.
    """
    return trace.get_tracer(INSTRUMENTATION_NAME, INSTRUMENTATION_VERSION)


def get_event_logger():
    """
    Return the logger used for GenAI events.

    Like :func:`get_tracer`, this is a no-op logger when no provider is
    installed, so callers never branch on whether telemetry is enabled.
    """
    return get_logger(INSTRUMENTATION_NAME, INSTRUMENTATION_VERSION)


def shutdown_tracing() -> None:
    """Flush and tear down both providers. Idempotent."""
    global _provider, _logger_provider
    with _init_lock:
        provider, _provider = _provider, None
        log_provider, _logger_provider = _logger_provider, None
    if provider is not None:
        provider.shutdown()
    if log_provider is not None:
        log_provider.shutdown()
