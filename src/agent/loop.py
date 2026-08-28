"""A tool-calling loop over the MCP server, traced turn by turn.

This exists to be watched. The point is not that an agent can answer a question, it is that a
person can see every tool it called, every raw result it got, and every lane the system
searched, before we hand those same tools to a third party.

Three properties it has to hold:

**It reaches the tools over MCP, not in process.** The surface the agent exercises has to be
the surface a third party gets, or testing it proves nothing about what we are exposing. It
also has to be a *different process*: the MCP tool bodies are `async def` with no `await`
inside, so every graph and Athena call in them blocks the event loop. An agent awaiting its own
worker's tool call would starve the loop that has to serve it.

**It has no identity.** The caller's bearer token is forwarded verbatim to MCP and never
re-minted, so every tool call is scoped to the real end user's tenant and to the records they
are on. An agent with a service account would be a hole straight through the tenant boundary.

**It stops.** Caps are enforced here rather than trusted to the model or the SDK, and each one
reports a distinct `stop_reason` so a reader can tell "it finished" from "we cut it off". The
compose cap is cost, not safety: every call is a full lane fan-out including Athena.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from src.agent.events import RUN_ID_HEADER, EventStream
from src.agent.prompt import RETRIEVAL_TOOLS, system_prompt

logger = logging.getLogger(__name__)

#: Turns the model may take. Twelve is enough to orient, gather, check provenance and answer;
#: past that a loop is usually rephrasing itself rather than making progress.
MAX_TURNS = 12

#: Wall clock for the whole run, checked between turns. A cap on turns alone does not bound a
#: run whose every turn is a slow Athena scan.
RUN_DEADLINE_S = 120.0

#: `compose` calls per run. The cost control: one call fans out across the vector store, the
#: graph, the catalog and possibly Athena.
#:
#: The fallback, not the number in force: `routes_retrieval._agent_for` passes the tenant's
#: `max_compose_calls`. This is what a caller that supplies nothing gets, which is tests.
MAX_COMPOSE_CALLS = 3

#: Consecutive failed tool calls before giving up. A model that has misread a schema three
#: times running will not read it correctly on the fourth.
MAX_CONSECUTIVE_TOOL_ERRORS = 3

#: Which stop reasons mean we intervened rather than the model finishing.
CAPPED_REASONS = frozenset({"turn_limit", "deadline", "compose_limit", "tool_errors"})

#: Said on the answer when the run searched nothing. Not a stop reason: the model finished
#: normally and produced prose, which is exactly what makes this worth saying out loud.
NO_RETRIEVAL_WARNING = (
    "Nothing was searched. This run called neither ask nor compose, so the answer below rests on "
    "the agent's own prose and on tools that describe the system rather than query it. Treat it as "
    "unevidenced and run the question again."
)


class AgentUnavailable(RuntimeError):
    """The loop cannot run: no model configured, or the MCP server is unreachable."""


@dataclass
class RunResult:
    run_id: str
    events: list[dict[str, Any]]
    answer: str = ""
    stop_reason: str = ""
    turns: int = 0
    searched: bool = True
    """Whether `ask` or `compose` ran. False means the prose has no retrieval under it."""

    max_compose_calls: int = MAX_COMPOSE_CALLS
    """The cap that was in force, so a `compose_limit` reads as "3 of 3 searches" rather than as a
    limit the reader has to go and look up. It is a tenant setting now, so it is not a constant the
    UI can hold."""

    @property
    def was_capped(self) -> bool:
        return self.stop_reason in CAPPED_REASONS

    @property
    def warnings(self) -> list[str]:
        return [] if self.searched else [NO_RETRIEVAL_WARNING]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "events": self.events,
            "answer": self.answer,
            "stop_reason": self.stop_reason,
            "turns": self.turns,
            "was_capped": self.was_capped,
            "searched": self.searched,
            # Same key the query tools use for a caveat on the answer, so the UI has one place
            # that renders "here is why to distrust this" rather than one per producer.
            "warnings": self.warnings,
            "max_turns": MAX_TURNS,
            "max_compose_calls": self.max_compose_calls,
            "note": (
                "Every tool call and its raw result are in `events`. The prose is the agent's "
                "own and is not governed."
            ),
        }


@dataclass
class RetrievalAgent:
    """One run of the loop. Built per request, because it holds that request's token."""

    tenant_id: str
    bearer: str
    """The caller's token, forwarded verbatim. Never parsed here and never re-minted: this
    object's only claim to authority is that somebody else already verified it."""

    mcp_url: str
    model_id: str

    region: str = ""
    agent_factory: Any = None
    """Injected in tests. A factory taking (tools, system_prompt) and returning something with
    Strands' `Agent` shape, so the loop is testable without Bedrock or the SDK."""

    client_factory: Any = None
    """Injected in tests. Returns an object with Strands' `MCPClient` shape."""

    retrieval_tool: str = ""
    """Which search tool the caller picked, or empty to let the model choose.

    Enforced by withholding the other one rather than by asking the model to prefer this one. A
    tool description sits closer to the decision than the system prompt does and wins against it,
    so a preference expressed only in prose is a preference the model is free to lose."""

    max_turns: int = MAX_TURNS
    deadline_s: float = RUN_DEADLINE_S
    max_compose_calls: int = MAX_COMPOSE_CALLS

    _compose_calls: int = field(default=0, init=False)
    _searched: bool = field(default=False, init=False)
    _error_streak: int = field(default=0, init=False)
    _stop: str = field(default="", init=False)

    def run(self, question: str, *, sink: Any = None) -> RunResult:
        """Answer a question, emitting an event per turn. Synchronous by design.

        Strands' `Agent.__call__` is synchronous, and wrapping it here would hide that from the
        caller. The route runs this in a worker thread instead, which keeps the API's event loop
        free to deliver the events this produces.
        """
        run_id = f"run:{uuid4().hex[:12]}"
        stream = EventStream(run_id=run_id, tenant_id=self.tenant_id, sink=sink)
        stream.emit(
            "run_started",
            question=question,
            model_id=self.model_id,
            max_turns=self.max_turns,
        )

        started = time.monotonic()
        try:
            client = self._client(run_id)
            with client:
                tools = self._offer(client.list_tools_sync())
                agent = self._agent(tools, stream, started)
                result = agent(question)
                answer = _text_of(result)
                stop = self._stop or str(getattr(result, "stop_reason", "") or "end_turn")
        except AgentUnavailable:
            raise
        except Exception as e:  # noqa: BLE001
            # The transcript up to the failure is the useful part, so this is reported as an
            # event rather than raised past the caller who is streaming it.
            logger.warning("retrieval run %s failed: %s", run_id, e)
            stream.emit("run_failed", error=str(e), stop_reason="error")
            return RunResult(
                run_id=run_id,
                events=stream.collected,
                stop_reason="error",
                turns=stream.turn,
                max_compose_calls=self.max_compose_calls,
            )

        stream.emit(
            "run_finished",
            answer=answer,
            stop_reason=stop,
            turns=stream.turn,
            was_capped=stop in CAPPED_REASONS,
            searched=self._searched,
            warnings=[] if self._searched else [NO_RETRIEVAL_WARNING],
        )
        logger.info(
            "retrieval run %s tenant=%s turns=%d stop=%s searched=%s",
            run_id,
            self.tenant_id,
            stream.turn,
            stop,
            self._searched,
        )
        return RunResult(
            run_id=run_id,
            events=stream.collected,
            answer=answer,
            stop_reason=stop,
            turns=stream.turn,
            searched=self._searched,
            max_compose_calls=self.max_compose_calls,
        )

    def _offer(self, tools: list[Any]) -> list[Any]:
        """The tools this run may use, with the unchosen search tool removed.

        The filter is here rather than at the MCP server because the choice belongs to one run,
        not to the token: the same user asking the same question a second time may want the other
        instrument. Only the *other search tool* is dropped, never anything else -- withholding
        `get_provenance` would leave the agent quoting assertion ids it cannot check.

        An unrecognised value drops nothing. Failing open matters more than honouring a typo:
        refusing every search tool would produce a confident answer with nothing under it, which
        is the failure this whole change exists to close.
        """
        if self.retrieval_tool not in RETRIEVAL_TOOLS:
            return tools
        drop = RETRIEVAL_TOOLS - {self.retrieval_tool}
        return [t for t in tools if _tool_name(t) not in drop]

    def _client(self, run_id: str) -> Any:
        if self.client_factory is not None:
            return self.client_factory()
        if not self.model_id:
            raise AgentUnavailable(
                "no retrieval agent model is configured for this tenant, so there is nothing "
                "to drive the loop"
            )
        # Imported here so this module loads without the SDK, which keeps the event and cap
        # logic importable in a deployment that never runs an agent.
        from strands.tools.mcp import MCPClient

        headers = {"Authorization": f"Bearer {self.bearer}"} if self.bearer else {}
        # Tells the MCP server these calls belong to a run that will record its own single audit
        # row, so `compose` does not also write one per call. Not a credential and never trusted
        # as one: authority is the bearer token above, and the worst a forged value can do is
        # suppress a duplicate row that the run itself still writes.
        headers[RUN_ID_HEADER] = run_id
        return MCPClient(url=self.mcp_url, headers=headers)

    def _agent(self, tools: list[Any], stream: EventStream, started: float) -> Any:
        hooks = [_TraceHooks(self, stream, started)]
        prompt = system_prompt(self.retrieval_tool)
        if self.agent_factory is not None:
            return self.agent_factory(tools=tools, system_prompt=prompt, hooks=hooks)

        from strands import Agent
        from strands.models import BedrockModel
        from strands.types.agent import Limits

        model = BedrockModel(model_id=self.model_id, region_name=self.region or None)
        agent = Agent(
            model=model,
            tools=tools,
            system_prompt=prompt,
            hooks=hooks,
            # Silences the SDK's stdout printer. The events are the output.
            callback_handler=None,
        )
        # The SDK's own turn cap as well as ours. Ours produces the stop_reason a reader sees;
        # this one is the backstop if a hook is ever bypassed.
        return lambda prompt: agent(prompt, limits=Limits(turns=self.max_turns))

    def _should_stop(self, started: float) -> str:
        """Which cap, if any, has been reached. Empty means keep going."""
        if time.monotonic() - started > self.deadline_s:
            return "deadline"
        if self._compose_calls > self.max_compose_calls:
            return "compose_limit"
        if self._error_streak >= MAX_CONSECUTIVE_TOOL_ERRORS:
            return "tool_errors"
        return ""


class _TraceHooks:
    """Turns Strands' tool lifecycle into our events, and enforces the caps.

    A hook rather than a wrapper around each tool: the wrapper would have to be re-applied
    every time a tool is added to the MCP server, and one that was forgotten would be a tool
    call absent from the trace. Hooks see everything the agent does by construction.
    """

    def __init__(self, agent: RetrievalAgent, stream: EventStream, started: float) -> None:
        self._agent = agent
        self._stream = stream
        self._started = started

    def register_hooks(self, registry: Any, **_: Any) -> None:
        from strands.hooks import AfterToolCallEvent, BeforeToolCallEvent

        registry.add_callback(BeforeToolCallEvent, self.before_tool)
        registry.add_callback(AfterToolCallEvent, self.after_tool)

    def before_tool(self, event: Any) -> None:
        tool, args = _tool_use(event)
        if tool == "compose":
            self._agent._compose_calls += 1

        reason = self._agent._should_stop(self._started)
        if reason:
            # Cancelled rather than raised: the model is told why, the transcript records it,
            # and the run ends on the next model turn instead of losing what it already had.
            self._agent._stop = reason
            self._stream.emit("tool_call", tool=tool, arguments=args, cancelled=reason)
            event.cancel_tool = _CANCEL_MESSAGES[reason]
            return

        self._stream.tool_call(tool, args)

    def after_tool(self, event: Any) -> None:
        tool, _ = _tool_use(event)
        exception = getattr(event, "exception", None)
        result = getattr(event, "result", None)
        is_error = exception is not None or _is_error_result(result)

        if is_error:
            self._agent._error_streak += 1
        else:
            self._agent._error_streak = 0
            # Recorded here rather than on the way in, because a search that was cancelled by a
            # cap or refused outright did not search. An empty result still counts: "we looked
            # and found nothing" is a real answer, and only "we never looked" is the defect.
            if tool in RETRIEVAL_TOOLS:
                self._agent._searched = True

        self._stream.tool_result(
            tool,
            _payload(result),
            is_error=is_error,
            error=str(exception) if exception else "",
        )

    # The turn cap is Strands' own `Limits`, so it needs no hook. Named here so a reader
    # looking for it does not conclude it is missing.


_CANCEL_MESSAGES = {
    "deadline": "This run has exceeded its time budget. Answer from what you already have.",
    "compose_limit": (
        "You have used this run's budget of full searches. Answer from the evidence you "
        "already gathered rather than searching again."
    ),
    "tool_errors": (
        "Several tool calls in a row failed. Stop calling tools and report what you have, "
        "including that the calls failed."
    ),
}


def _tool_name(tool: Any) -> str:
    """A tool's name, however the SDK wrapped it. `tool_name` on Strands, `name` over MCP."""
    for attr in ("tool_name", "name"):
        value = getattr(tool, attr, None)
        if isinstance(value, str) and value:
            return value
    if isinstance(tool, dict):
        return str(tool.get("name") or tool.get("tool_name") or "")
    return ""


def _tool_use(event: Any) -> tuple[str, dict[str, Any]]:
    use = getattr(event, "tool_use", None) or {}
    return str(use.get("name", "unknown")), dict(use.get("input", {}) or {})


def _is_error_result(result: Any) -> bool:
    return isinstance(result, dict) and result.get("status") == "error"


def _payload(result: Any) -> Any:
    """The tool's own output, unwrapped from the SDK's envelope.

    Strands returns `{"toolUseId", "status", "content": [{"json"|"text": ...}]}`. The UI needs
    the tool's dict to hand to `lanesFromComposed`, so an envelope reaching the browser would
    make every consumer unwrap it.

    **A text block holding JSON is parsed back to a dict.** MCP serialises a tool's return value
    into a text block, not the `json` block this first assumed, so every composed result reached
    the browser as a *string*. Reading `.parts` off a string gives undefined, and the trace
    rendered "nothing ran" for a result that had thirty facts in it -- the worst shape of failure
    here, because it looks like an honest empty.
    """
    if not isinstance(result, dict):
        return result
    content = result.get("content")
    if not isinstance(content, list):
        return result
    for block in content:
        if isinstance(block, dict) and "json" in block:
            return block["json"]

    texts = [b["text"] for b in content if isinstance(b, dict) and "text" in b]
    if not texts:
        return result
    joined = "\n".join(texts)
    try:
        parsed = json.loads(joined)
    except (TypeError, ValueError):
        # Genuinely prose: a `ToolError` message, or a tool that returns text. Left as it is.
        return joined
    # Only a mapping is a tool result. A bare number or string that happens to be valid JSON is
    # still text as far as a reader is concerned.
    return parsed if isinstance(parsed, dict | list) else joined


def _text_of(result: Any) -> str:
    """The model's final prose, however the SDK wrapped it."""
    message = getattr(result, "message", None)
    if isinstance(message, dict):
        blocks = message.get("content") or []
        text = " ".join(b["text"] for b in blocks if isinstance(b, dict) and "text" in b)
        if text:
            return text.strip()
    return str(result).strip()
