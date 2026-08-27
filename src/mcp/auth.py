"""Authenticating an MCP tool call.

The property this module exists to hold: **an agent has no identity of its own.** Every
tool call carries the bearer token of the person driving the agent, is verified by the
same `Authenticator` the REST API uses, and produces the same `AuthContext` — so tenant
scoping and matter walls apply identically whether a lawyer clicks a button or an agent
calls a tool. An agent that could see more than its user is a privilege breach wearing a
helpful face, and for legal work that is not a tradeable property.

The token is read from the transport request, never from a tool argument. Tool arguments
are model-controlled, and a caller that can name its own identity has none.
"""

from __future__ import annotations

from mcp.server.fastmcp import Context

from src.agent.events import RUN_ID_HEADER
from src.api.deps import Services
from src.auth import Grants, bearer_from_header
from src.graph.scope import AuthContext


def run_id_from_context(ctx: Context) -> str | None:
    """The retrieval run this call belongs to, if it says it belongs to one.

    Only ever used to avoid writing a second audit row for a call a run already records. Never an
    identity and never a grant: unlike the bearer token this is unverified, and the most a forged
    value can achieve is suppressing a duplicate.
    """
    request = ctx.request_context.request
    if request is None:
        return None
    return request.headers.get(RUN_ID_HEADER) or None


def bearer_from_context(ctx: Context) -> str | None:
    request = ctx.request_context.request
    if request is None:
        # No HTTP request behind this call (stdio transport, or a synthetic
        # invocation). There is no header to read and nothing to fall back to, so this
        # can only succeed under the local dev bypass.
        return None
    return bearer_from_header(request.headers.get("authorization"))


def principal_from_context(services: Services, ctx: Context) -> tuple[AuthContext, Grants]:
    """Verify the caller's token and build the scope every read is filtered by.

    Delegates to the REST API's `Authenticator` rather than re-deriving anything. That
    includes the local dev bypass, which `Authenticator._dev_context` re-checks the
    environment for and `GroundworkConfig.validate` refuses to start with outside local —
    two independent gates, neither of which this transport may weaken.

    `include_suggestions` is not exposed. The REST API takes it as a query parameter for a
    research surface a person is looking at; an agent relaying a PREDICTED guess to a client
    as a finding has no such surface, so the flag stays off here by construction.
    """
    return services.authenticator.authenticate(bearer_from_context(ctx))
