"""The one event shape a retrieval run emits.

Every turn, tool call, tool result and stop reason goes out through this, so the browser has
a single thing to render and the transcript is the trace rather than a summary of one.

**`result_kind` comes from the tool's name, never from the payload.** A UI that sniffed a
result's shape to decide whether to draw a governance trace would be inferring governance
from data, which is the one thing this system does not do anywhere else: an assertion says
what it is, and so does a tool result. The map below is the whole contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

#: How to render each tool's result. Keyed on the MCP tool name.
#:
#: `composed` is the only one carrying the full lane trace, which is why `compose` exists.
#: Anything absent falls back to `json`, so a tool added to the MCP server without a decision
#: here renders as raw JSON rather than being mislabelled as something governed.
RESULT_KINDS: dict[str, str] = {
    "compose": "composed",
    "ask": "resolution",
    "search_assertions": "assertions",
    "get_provenance": "provenance",
    "graph_neighbourhood": "graph",
    "list_metrics": "metrics",
    "describe_ontology": "ontology",
}

DEFAULT_RESULT_KIND = "json"

#: Set by `RetrievalAgent` on its MCP calls so the server knows the call is part of a run that
#: records its own audit row. Carries no authority; see `RetrievalAgent._client`.
RUN_ID_HEADER = "x-lexgraph-run-id"

#: Every `kind` a client may receive. Listed so a reader knows the full set without grepping.
KINDS = frozenset({"run_started", "tool_call", "tool_result", "text", "run_finished", "run_failed"})


def result_kind_for(tool: str) -> str:
    return RESULT_KINDS.get(tool, DEFAULT_RESULT_KIND)


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class EventStream:
    """Numbers the events of one run and hands each to a sink.

    `seq` is monotonic across the whole run, not per kind. A websocket delivers in order but a
    transcript assembled from a POST body does not have to be, and a reader reconstructing the
    order from timestamps would be guessing between two events in the same millisecond.
    """

    run_id: str
    tenant_id: str
    sink: Any = None
    """Called with each event dict. None collects without publishing, which is what the POST
    route wants: the whole transcript, returned rather than streamed."""

    collected: list[dict[str, Any]] = field(default_factory=list)
    turn: int = 0
    _seq: int = 0

    def emit(self, kind: str, **fields: Any) -> dict[str, Any]:
        self._seq += 1
        event = {
            "kind": kind,
            "run_id": self.run_id,
            "tenant_id": self.tenant_id,
            "turn": self.turn,
            "seq": self._seq,
            "at": _now(),
            **fields,
        }
        self.collected.append(event)
        if self.sink is not None:
            self.sink(event)
        return event

    def tool_call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """A tool the model chose. The turn advances here rather than on the result, so a call
        whose result never arrives is still visible as a turn that was attempted."""
        self.turn += 1
        return self.emit("tool_call", tool=tool, arguments=arguments)

    def tool_result(
        self, tool: str, result: Any, *, is_error: bool = False, error: str = ""
    ) -> dict[str, Any]:
        return self.emit(
            "tool_result",
            tool=tool,
            result_kind=result_kind_for(tool),
            result=result,
            is_error=is_error,
            error=error,
        )
