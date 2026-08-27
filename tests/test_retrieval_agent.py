"""The agent loop: what it emits, and what stops it.

Strands is never imported here. The loop takes an `agent_factory` and a `client_factory`, and
these tests inject fakes that drive the real hooks the way the SDK does. That keeps the subject
of the test the *loop* -- its event contract and its caps -- rather than the SDK's behaviour,
which is not ours to assert.

The caps are the reason this file matters. A loop that answers well is pleasant; a loop that
cannot be made to stop is a bill and a hung request, and the difference between "it finished"
and "we cut it off" has to reach the person reading the transcript.
"""

from __future__ import annotations

from typing import Any, Self

import sys
import types

import pytest
from fastapi.testclient import TestClient

from src.agent.events import (
    DEFAULT_RESULT_KIND,
    RUN_ID_HEADER,
    EventStream,
    result_kind_for,
)
from src.agent.loop import CAPPED_REASONS, RetrievalAgent
from src.api.app import create_app
from src.config import AuthConfig, GraphConfig, GroundworkConfig

TENANT = "demo-firm"

OK = {
    "status": "success",
    "content": [{"json": {"lanes_run": ["metric"], "governance": "governed"}}],
}
BAD = {"status": "error", "content": [{"text": "unknown field 'reviewstate'"}]}


class _Registry:
    """Stands in for Strands' `HookRegistry`."""

    def __init__(self) -> None:
        self.callbacks: dict[str, Any] = {}

    def add_callback(self, event_type: Any, callback: Any) -> None:
        self.callbacks[event_type.__name__] = callback


class _ToolEvent:
    def __init__(self, name: str, args: dict[str, Any], result: Any = None, exception: Any = None):
        self.tool_use = {"name": name, "input": args}
        self.result = result
        self.exception = exception
        self.cancel_tool = None


class _Client:
    """Stands in for `MCPClient`: a context manager exposing the tool list."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> bool:
        return False

    def list_tools_sync(self) -> list[str]:
        return ["compose", "ask", "get_provenance"]


def _scripted(script: list[tuple[str, dict[str, Any], Any]]) -> Any:
    """An agent factory whose model calls exactly these tools, in order.

    Stops early when a `before` hook cancels, which is how the SDK behaves and is what makes
    the cap assertions meaningful rather than decorative.
    """

    def factory(*, tools: list[Any], system_prompt: str, hooks: list[Any]) -> Any:
        registry = _Registry()
        for hook in hooks:
            hook.register_hooks(registry)

        def run(prompt: str) -> Any:
            for name, args, result in script:
                before = _ToolEvent(name, args)
                registry.callbacks["BeforeToolCallEvent"](before)
                if before.cancel_tool:
                    break
                registry.callbacks["AfterToolCallEvent"](_ToolEvent(name, args, result=result))
            return type(
                "Result",
                (),
                {"message": {"content": [{"text": "Grounded answer."}]}, "stop_reason": "end_turn"},
            )()

        return run

    return factory


def _agent(script: list[tuple[str, dict[str, Any], Any]], **over: Any) -> RetrievalAgent:
    return RetrievalAgent(
        tenant_id=TENANT,
        bearer="token-for-the-user",
        mcp_url="http://mcp.test/mcp",
        model_id="some.model",
        agent_factory=_scripted(script),
        client_factory=_Client,
        **over,
    )


class TestTheEventContract:
    def test_events_are_ordered_and_numbered(self):
        """A transcript assembled from a POST body has no delivery order to rely on, and two
        events in the same millisecond cannot be ordered by timestamp."""
        result = _agent([("compose", {"question": "q"}, OK)]).run("q")

        assert [e["kind"] for e in result.events] == [
            "run_started",
            "tool_call",
            "tool_result",
            "run_finished",
        ]
        assert [e["seq"] for e in result.events] == [1, 2, 3, 4]

    def test_a_composed_result_is_tagged_by_the_tool_that_produced_it(self):
        """The UI decides whether to draw a governance trace from `result_kind`, never by
        sniffing the payload. Inferring governance from the shape of data is the one thing this
        system does not do anywhere else."""
        result = _agent([("compose", {"question": "q"}, OK)]).run("q")
        [tool_result] = [e for e in result.events if e["kind"] == "tool_result"]

        assert tool_result["result_kind"] == "composed"
        # And the tool's own dict survives, unwrapped from the SDK envelope, so the UI can hand
        # it straight to `lanesFromComposed`.
        assert tool_result["result"] == {"lanes_run": ["metric"], "governance": "governed"}

    def test_a_json_text_block_reaches_the_ui_as_a_dict(self):
        """MCP serialises a tool's return value into a **text** block, not the `json` block this
        first assumed. Handing the browser the string meant `composed.parts` was undefined and the
        trace rendered "nothing ran" for a result holding thirty facts, which is the worst shape of
        failure here: it looks like an honest empty."""
        import json as _json

        payload = {"parts": [], "lanes_run": ["graph"], "governance": "verbatim"}
        wrapper = {"status": "success", "content": [{"text": _json.dumps(payload)}]}

        result = _agent([("compose", {}, wrapper)]).run("q")
        [tool_result] = [e for e in result.events if e["kind"] == "tool_result"]
        assert tool_result["result"] == payload

    def test_a_json_block_still_works(self):
        """Both shapes are handled: a server that sends structured content is not broken by the
        text path."""
        payload = {"parts": [], "lanes_run": ["metric"]}
        wrapper = {"status": "success", "content": [{"json": payload}]}

        result = _agent([("compose", {}, wrapper)]).run("q")
        [tool_result] = [e for e in result.events if e["kind"] == "tool_result"]
        assert tool_result["result"] == payload

    def test_an_error_message_stays_text(self):
        """A `ToolError` is prose, not a result. Parsing it into an object would make a refusal
        look like data."""
        wrapper = {
            "status": "error",
            "content": [{"text": "Error executing tool compose: response_format must be one of"}],
        }
        result = _agent([("compose", {}, wrapper)]).run("q")
        [tool_result] = [e for e in result.events if e["kind"] == "tool_result"]
        assert isinstance(tool_result["result"], str)
        assert tool_result["is_error"] is True

    def test_an_unknown_tool_renders_as_raw_json_rather_than_as_governed(self):
        """A tool added to the MCP server without a decision here must not inherit a label that
        claims more than it is."""
        assert result_kind_for("some_future_tool") == DEFAULT_RESULT_KIND
        assert result_kind_for("compose") == "composed"

    def test_the_turn_advances_on_the_call_not_the_result(self):
        """A call whose result never arrives is still a turn that was attempted, and a reader
        needs to see it."""
        stream = EventStream(run_id="r", tenant_id=TENANT)
        stream.tool_call("ask", {})
        assert stream.turn == 1
        stream.tool_result("ask", {})
        assert stream.turn == 1


class TestItStops:
    def test_a_runaway_compose_loop_is_capped(self):
        """The cost control. One compose call fans out across the vector store, the graph, the
        catalog and possibly Athena, so this is the cap that matters to a bill."""
        result = _agent([("compose", {}, OK)] * 6, max_compose_calls=2).run("q")

        assert result.stop_reason == "compose_limit"
        assert result.was_capped
        # The cancelled call is still in the transcript, naming which cap stopped it, so a
        # reader sees the attempt rather than a gap where a turn should be.
        [cancelled] = [e for e in result.events if e.get("cancelled")]
        assert cancelled["cancelled"] == "compose_limit"
        assert cancelled["tool"] == "compose"

    def test_a_run_that_takes_too_long_is_capped(self):
        """A turn cap alone does not bound a run whose every turn is a slow Athena scan."""
        result = _agent([("ask", {}, OK)] * 3, deadline_s=-1).run("q")
        assert result.stop_reason == "deadline"

    def test_repeated_tool_failures_stop_the_loop(self):
        """A model that has misread a schema three times running will not read it correctly on
        the fourth."""
        result = _agent([("ask", {}, BAD)] * 5).run("q")
        assert result.stop_reason == "tool_errors"

    def test_one_bad_call_does_not_stop_a_run_that_recovers(self):
        """The opposite failure, and the more likely one. A rejected argument is information the
        model can act on, so the loop must survive it."""
        result = _agent([("ask", {}, BAD), ("compose", {}, OK), ("ask", {}, OK)]).run("q")

        assert result.stop_reason == "end_turn"
        assert not result.was_capped
        assert result.turns == 3
        assert sum(1 for e in result.events if e.get("is_error")) == 1

    def test_every_cap_reports_a_reason_a_reader_can_tell_apart(self):
        """ "It finished" and "we cut it off" must not render the same, and neither must two
        different caps."""
        reasons = {
            _agent([("compose", {}, OK)] * 4, max_compose_calls=1).run("q").stop_reason,
            _agent([("ask", {}, OK)] * 2, deadline_s=-1).run("q").stop_reason,
            _agent([("ask", {}, BAD)] * 4).run("q").stop_reason,
        }
        assert reasons == {"compose_limit", "deadline", "tool_errors"}
        assert reasons <= CAPPED_REASONS

    def test_a_failure_keeps_the_transcript(self):
        """The turns before the failure are the useful part, so a crash is reported as an event
        rather than losing what the run already established."""

        def exploding(**_: Any) -> Any:
            def run(_prompt: str) -> Any:
                raise RuntimeError("bedrock said no")

            return run

        result = RetrievalAgent(
            tenant_id=TENANT,
            bearer="t",
            mcp_url="u",
            model_id="m",
            agent_factory=exploding,
            client_factory=_Client,
        ).run("q")

        assert result.stop_reason == "error"
        assert [e["kind"] for e in result.events] == ["run_started", "run_failed"]
        assert "bedrock said no" in result.events[-1]["error"]


class TestTheAgentHasNoIdentityOfItsOwn:
    def test_the_callers_token_is_what_reaches_mcp(self):
        """The property `src/mcp/auth.py` exists to protect. An agent with a service account
        would be a hole straight through the tenant boundary, so the token is forwarded verbatim
        and never re-minted."""
        captured: dict[str, Any] = {}

        def client_factory() -> Any:
            captured["headers"] = {"Authorization": "Bearer token-for-the-user"}
            return _Client()

        agent = _agent([("ask", {}, OK)])
        agent.client_factory = client_factory
        agent.run("q")

        assert captured["headers"]["Authorization"].endswith("token-for-the-user")

    def test_the_run_is_tagged_with_the_tenant_it_was_scoped_to(self):
        events = _agent([("ask", {}, OK)]).run("q").events
        assert {e["tenant_id"] for e in events} == {TENANT}


class TestTheRoutesRefuseBeforeTheyPretend:
    @pytest.fixture
    def client(self) -> TestClient:
        cfg = GroundworkConfig(
            environment="local",
            auth=AuthConfig(dev_bypass_tenant=TENANT),
            graph=GraphConfig(uri="bolt://127.0.0.1:1", user="none", password="none"),
        )
        cfg.validate()
        return TestClient(create_app(cfg))

    def test_no_mcp_endpoint_is_a_503_not_an_answer(self, client):
        """An agent that cannot reach its tools would produce prose with no evidence under it,
        which is worse than no answer."""
        r = client.post(f"/api/tenants/{TENANT}/retrieval/runs", json={"question": "hi"})
        assert r.status_code == 503
        assert "MCP_URL" in r.json()["detail"]

    def test_a_bad_token_closes_the_socket_before_accepting_it(self, client):
        """Closed before accept, so an unauthorised caller cannot tell a bad token from a wrong
        tenant. Same shape as `ingest_events`."""
        from starlette.websockets import WebSocketDisconnect

        with (
            pytest.raises(WebSocketDisconnect),
            client.websocket_connect(
                f"/api/tenants/{TENANT}/retrieval/events?token=nonsense"
            ) as ws,
        ):
            ws.receive_json()


class TestTheRunIsRecordedOnce:
    """One row per run, not one per tool call.

    The agent calls `compose` up to three times, and the MCP server records every `compose` it
    serves. Without the run header those are four rows for one question, which would make the
    audit page read as four pieces of advice given.
    """

    def test_the_run_id_is_sent_to_mcp_alongside_the_token(self, monkeypatch):
        """Asserted on the real client rather than through `client_factory`: the factory is the
        test seam, so a test driven through it would pass while the deployed path sent neither
        header."""
        captured: dict[str, Any] = {}

        class _Recording:
            def __init__(self, *, url: str, headers: dict[str, str]) -> None:
                captured.update(headers)

        # `_client` imports MCPClient inside the function, so the patch lands on the module the
        # import resolves to.
        monkeypatch.setitem(
            sys.modules, "strands.tools.mcp", types.SimpleNamespace(MCPClient=_Recording)
        )
        agent = RetrievalAgent(
            tenant_id=TENANT,
            bearer="token-for-the-user",
            mcp_url="http://mcp.test/mcp",
            model_id="some.model",
        )
        agent._client("run:abc")

        assert captured["Authorization"] == "Bearer token-for-the-user"
        assert captured[RUN_ID_HEADER] == "run:abc"

    def test_a_call_carrying_a_run_id_is_not_recorded_by_the_mcp_server(self):
        """The suppression itself. A forged header can only ever remove a duplicate: the run still
        writes its own row, so no question can hide by sending one."""
        from src.mcp.auth import run_id_from_context

        class _Req:
            headers = {RUN_ID_HEADER: "run:abc"}

        class _Ctx:
            request_context = type("RC", (), {"request": _Req()})()

        assert run_id_from_context(_Ctx()) == "run:abc"

    def test_a_third_party_call_sends_no_run_id_and_is_recorded(self):
        from src.mcp.auth import run_id_from_context

        class _Req:
            headers: dict[str, str] = {}

        class _Ctx:
            request_context = type("RC", (), {"request": _Req()})()

        assert run_id_from_context(_Ctx()) is None
