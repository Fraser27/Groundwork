"""The Retrieval agent: a tool-calling loop whose whole transcript is the answer.

Two ways in, one loop:

    WS   /tenants/{t}/retrieval/events?token=...   turns as they happen
    POST /tenants/{t}/retrieval/runs               the whole transcript at the end

**The run executes inside the websocket handler**, not a background task. Two reasons. A
task started by the POST and watched over the socket has a race nobody can close from the
client side: the run can finish before the browser has subscribed. And a background task
dies with the container, so a run would vanish mid-answer with no way to say why. Running it
in the handler means the run's lifetime is the socket's lifetime, which for an interactive
test surface is exactly right: close the tab and the work stops.

The POST route is the same loop without the socket. It is the seam tests drive, and the
fallback where a websocket is blocked.

Both are read-only. The agent's prose is ungoverned by construction and is never written back
to the graph.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal

import anyio
from fastapi import (
    APIRouter,
    Body,
    Header,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from pydantic import BaseModel, Field

from src.agent.loop import AgentUnavailable, RetrievalAgent
from src.agent.prompt import RETRIEVAL_TOOLS
from src.api.deps import Services, ServicesDep, TenantDep, get_services
from src.api.events import get_event_hub
from src.auth import AuthError, bearer_from_header
from src.graph.scope import ScopeViolation
from src.query_audit import event_for_run

logger = logging.getLogger(__name__)

router = APIRouter(tags=["retrieval"])


class RunRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    tool: Literal["auto", "ask", "compose"] = "auto"
    """Which search tool to allow. `auto` leaves the choice to the model.

    A choice here is enforced by withholding the other one from the agent's tool list, so it
    binds rather than suggests."""


def _tool_choice(tool: str) -> str:
    """The value `RetrievalAgent` wants: a tool name, or empty for the model's own choice.

    Anything unrecognised normalises to the model's choice. The websocket takes this from a client
    message rather than a validated body, and the safe reading of a value nobody understands is
    "no constraint" -- a constraint built from a typo could withhold both search tools.
    """
    return tool if tool in RETRIEVAL_TOOLS else ""


def _agent_for(
    services: Services, tenant_id: str, bearer: str, tool: str = "auto"
) -> RetrievalAgent:
    """Build the loop, or refuse with the reason.

    503 rather than a stub answer: an agent that cannot reach its tools would produce prose
    with no evidence under it, which is worse than no answer at all.
    """
    mcp_url = services.config.mcp_url
    if not mcp_url:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "no MCP endpoint is configured (MCP_URL unset), so the agent has no tools to call",
        )
    settings = services.settings_for(tenant_id)
    model_id = settings.retrieval_agent_model
    if not model_id:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "no retrieval agent model is configured for this tenant",
        )
    return RetrievalAgent(
        tenant_id=tenant_id,
        bearer=bearer,
        mcp_url=mcp_url,
        model_id=model_id,
        region=services.config.models.region,
        retrieval_tool=_tool_choice(tool),
        # Read per run, not per process. The right number depends on which model drives the loop,
        # and that is a tenant setting too, so a constant here made the cheaper model's setting a
        # trap: it spends calls rediscovering `compose`'s arguments and an administrator had no way
        # to pay for that.
        max_compose_calls=settings.max_compose_calls,
    )


@router.post("/tenants/{tenant}/retrieval/runs")
async def run_retrieval(
    services: ServicesDep,
    principal: TenantDep,
    body: Annotated[RunRequest, Body()],
    authorization: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    """Run the loop and return the whole transcript.

    Synchronous, so the caller waits. That is the honest shape for a POST: the alternative is a
    background task whose result nothing reliably collects, since a task dies with the container.
    """
    ctx, _ = principal
    agent = _agent_for(
        services, ctx.tenant_id, bearer_from_header(authorization) or "", body.tool
    )
    try:
        result = await anyio.to_thread.run_sync(agent.run, body.question)
    except AgentUnavailable as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(e)) from e
    _record_run(services, ctx, body.question, result)
    return result.to_dict()


@router.websocket("/tenants/{tenant}/retrieval/events")
async def retrieval_events(websocket: WebSocket, tenant: str, token: str = "") -> None:
    """Run the loop, streaming each turn as it happens.

    The token arrives as a query parameter for the same reason as `ingest_events`: a browser
    cannot set headers on a WebSocket handshake. Same trade-off, accepted for the same reason,
    with one difference worth stating -- this token is also **forwarded to MCP** as the agent's
    only authority, so the tool calls are scoped to this user rather than to the service.

    Unlike ingest events there is no poll behind this. A client on one task will not see a run
    driven on another, so this needs `appDesiredCount: 1`; the POST route is the fallback.
    """
    services = get_services()
    hub = get_event_hub()

    try:
        ctx, _ = services.authenticator.authenticate(token)
        services.authenticator.assert_tenant_matches(ctx, tenant)
    except (AuthError, ScopeViolation):
        # Closed before accept, so an unauthorised caller never gets a socket and cannot tell a
        # bad token from a wrong tenant.
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    try:
        agent = _agent_for(services, ctx.tenant_id, token)
    except HTTPException as e:
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR, reason=str(e.detail)[:120])
        return

    await websocket.accept()
    sub = hub.subscribe(ctx.tenant_id)

    try:
        first = await websocket.receive_json()
        question = str(first.get("question", "")).strip()
        if not question:
            await websocket.send_json({"kind": "run_failed", "error": "no question"})
            return
        # Set after the handshake rather than passed to `_agent_for` above, because the choice
        # arrives with the question. The earlier build is the configuration check, which has to
        # happen before `accept` so a misconfigured tenant gets a close reason instead of a socket.
        agent.retrieval_tool = _tool_choice(str(first.get("tool", "")))

        # The run is driven in a worker thread and its events are relayed here. Strands' API is
        # synchronous, so running it on this loop would block the send that delivers its events.
        async with anyio.create_task_group() as tg:
            tg.start_soon(_relay, websocket, sub)
            result = await anyio.to_thread.run_sync(
                lambda: agent.run(
                    question,
                    sink=lambda event: hub.publish(ctx.tenant_id, event),
                )
            )
            # The streamed run is the one people actually use, so this is the path that matters
            # for the audit. Recorded here rather than inside `agent.run` because the loop is
            # deliberately free of service dependencies.
            _record_run(services, ctx, question, result)
            # Let the relay drain what the run just published before the group is torn down.
            await anyio.sleep(0.1)
            tg.cancel_scope.cancel()
    except WebSocketDisconnect:
        # The reader left, so the run has no audience. Nothing to clean up beyond the socket:
        # the loop is read-only and wrote nothing.
        pass
    except Exception as e:  # noqa: BLE001
        logger.warning("retrieval socket for %s failed: %s", ctx.tenant_id, e)
        with anyio.CancelScope(shield=True):
            try:
                await websocket.send_json({"kind": "run_failed", "error": str(e)})
            except Exception as send_error:  # noqa: BLE001
                # The socket is already broken, which is why the send failed. Logged rather
                # than swallowed so a reader of the logs sees both halves of the failure.
                logger.debug("could not report the failure to the client: %s", send_error)
    finally:
        hub.unsubscribe(ctx.tenant_id, sub)


def _record_run(services: Services, ctx: Any, question: str, result: Any) -> None:
    """One row per run, carrying the tools it called.

    A failed run is recorded too. "The agent was asked this and could not answer" is part of the
    record, and a log holding only successes overstates how well the surface works.
    """
    services.record_question(event_for_run(ctx.tenant_id, ctx.user_id, question, result))


async def _relay(websocket: WebSocket, sub: Any) -> None:
    """Forward hub events to the socket until cancelled."""
    while True:
        event = await sub.queue.get()
        await websocket.send_json(event)
