"""MCP tools over the same governed layer the REST API serves.

Mounted as an ASGI app at `src.mcp.server:app`, which is what `docker-compose.yml`'s
`mcp` profile and `cdk/lib/mcp-stack.ts` both point at. Same image as the API, entered at
a different module — an MCP tool that answered a question differently from the matching
endpoint would mean two governance implementations, and the tools would be the one nobody
audits.

Two properties this module has to hold, both non-negotiable:

**No standalone agent identity.** Every tool call is authorized as the user whose bearer
token it carries, through `src/mcp/auth.py` and the API's own `Authenticator`. Same tenant
filter, same ethical walls. There is no service account into the graph.

**Every tool states its own trust semantics.** An agent choosing between `ask` and
`search_assertions` has to be able to tell that one is governed and the other is raw, from
the tool description alone — it will not read this file. The descriptions are therefore
longer than a docstring would normally be; they are the interface.

`include_suggestions` is deliberately absent from every tool. PREDICTED assertions are
guesses, and letting an agent opt into them is how a guess reaches a client as a finding.
"""

from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from src.api.deps import Services, build_services, get_services, set_services
from src.auth import AuthError
from src.config import LexGraphConfig
from src.documents.review import AssertionNotFound
from src.graph.assertions import EpistemicClass, ReviewState
from src.graph.scope import AuthContext, ScopeViolation
from src.mcp.auth import principal_from_context
from src.query.resolver import QueryBlocked, Resolution, Tier

logger = logging.getLogger(__name__)

INSTRUCTIONS = """
# LexGraph — governed semantic layer for legal work

Every fact this server returns carries provenance: a document page and the verbatim
quote, or the proof tree of an inference. When a client asks "how do you know that?",
call `get_provenance` and you will have a real answer.

Start with `ask`. It routes through four tiers and tells you which one answered:

  1. governed metric — SQL compiled from an approved definition, no AI in the path
  2. graph traversal — verified relationships only
  3. hybrid          — passages, then verified relationships out from them
  4. AI-written SQL  — ungoverned, and switchable off per firm

Tiers 1-3 are governed. Tier 4 is a starting point, not an answer. Always relay the tier,
and never present a tier-4 result as established.

Authorization is the user's, not yours. You see exactly what the person whose token you
are carrying sees — same firm, same ethical screens. If a question returns nothing, that may
be because the answer is screened off from your user, so do not conclude that a fact does
not exist.

Where a screen applies, you are told which matter and who to contact. Relay that. A lawyer
reading "no conflicts found" when the truth is "none you can see" is the harm the screen
exists to prevent, and you are the one holding the only warning they will get.
"""

#: What an agent is told for an id that does not exist, belongs to another firm, or sits on
#: a matter nobody assigned it to. An in-tenant ethical screen is deliberately *not*
#: flattened into this: the agent is told the matter is screened and who to contact, so the
#: lawyer driving it learns their view is incomplete rather than concluding nothing is there.
_UNREACHABLE = "no assertion with that id is available to you"

#: Premise recursion cap. A proof tree is a DAG and the store is append-only, so a cycle
#: should be impossible — but "should be impossible" is not a termination condition for a
#: recursive read serving a caller that can retry.
_MAX_PREMISE_DEPTH = 6

#: Row cap. An agent asking for everything still gets a bounded response.
_MAX_ROWS = 200


def _principal(ctx: Context) -> tuple[Services, AuthContext]:
    """The services and the caller's scope, or a refusal.

    Grants are discarded: every tool here is read-only, and role checks belong on the write
    surfaces (approve, reject, governance) which are the REST API's. If a write tool is ever
    added it must take `Grants` and check `can_review`, exactly as `routes_review` does.
    """
    services = get_services()
    try:
        auth_ctx, _ = principal_from_context(services, ctx)
    except AuthError as e:
        raise ToolError(str(e)) from e
    return services, auth_ctx


async def ask(ctx: Context, question: str, execute: bool = False) -> dict[str, Any]:
    """Answer a question through the governed resolver, and report how it was answered.

    TRUST: varies by tier, and the result says which. `tier_name` is GOVERNED_METRIC (SQL
    compiled from a definition a human approved — no AI wrote it, and the same question
    always returns the same number), GRAPH_TRAVERSAL or HYBRID (verified facts above the
    firm's confidence floor, each one traceable), or LLM_SQL (an AI model wrote the SQL; it
    passed the query firewall but nobody approved it — treat it as a draft). `governed` is
    false only for LLM_SQL. `assertions_used` lists the facts the answer rests on; pass any
    of them to `get_provenance` for the document page and quote behind it.

    Scoped to the calling user: their firm, and only the matters they may see. An empty
    answer may mean "not permitted for you" rather than "does not exist".

    Args:
        question: A question in ordinary language.
        execute: Run the query. False returns the SQL for review without running it, which
            is the point of a governed metric — a person can read what it will do first.
    """
    services, auth_ctx = _principal(ctx)
    settings = services.settings_for(auth_ctx.tenant_id)

    try:
        resolution = services.build_resolver().resolve(
            auth_ctx, question, settings, execute=execute
        )
    except QueryBlocked as e:
        # A deliberate refusal of a well-formed question, not a failure. The message names
        # the remedy so the agent can tell its user what to ask an administrator for.
        raise ToolError(str(e)) from e
    except ScopeViolation as e:
        raise ToolError(str(e)) from e

    logger.info(
        "mcp ask tenant=%s tier=%s assertions=%d",
        auth_ctx.tenant_id,
        resolution.tier.name,
        len(resolution.assertions_used),
    )
    return _resolution_out(resolution)


def _resolution_out(resolution: Resolution) -> dict[str, Any]:
    out = resolution.to_dict()
    # Names rather than the ints `to_dict` emits: the REST caller renders these, an agent
    # reads them.
    out["tiers_attempted"] = [Tier(t).name for t in resolution.tiers_attempted]
    return out


async def list_metrics(ctx: Context) -> dict[str, Any]:
    """List the governed metrics — the measures a human has defined and approved.

    TRUST: the highest available. Each compiles to deterministic SQL with no model in the
    path, so a question a metric covers has one correct answer rather than a plausible one.
    Prefer phrasing a question so `ask` matches a metric here over letting it fall through
    to AI-written SQL.

    `time_grains` is a hard restriction, not a hint: a metric declaring month/quarter/year
    cannot be served daily, and asking for a grain outside the list fails rather than
    quietly answering at the wrong one. `aggregation` says whether the number may be summed
    — `non_additive` measures (ratios, averages, distinct counts) are wrong if you add them
    up yourself.
    """
    services, _ = _principal(ctx)
    matcher = services.metric_matcher
    if matcher is None:
        return {
            "metrics": [],
            "note": "no metric pack is loaded, so no governed metric can answer",
        }

    return {
        "metrics": [
            {
                "metric_id": m.metric_id,
                "name": m.name,
                "definition": m.definition,
                "synonyms": list(m.synonyms),
                "type": m.type,
                "expression": m.expression,
                "source_table": m.source_table,
                "aggregation": m.aggregation,
                "time_grains": list(m.time_grains),
                "time_grain_column": m.time_grain_column,
                "filterable_columns": [p.column for p in m.parameters],
                "unit": m.unit,
                "owner": m.owner,
            }
            for m in matcher.metrics
        ]
    }


async def describe_ontology(ctx: Context) -> dict[str, Any]:
    """Describe the vocabulary: entity types, relationships, and inference rules.

    TRUST: this is the schema, not data. It says what the graph *can* record, not what it
    does.

    The split between governing and descriptive predicates is the part worth reading.
    Governing predicates are a closed set: they are what conflict checks, privilege walls
    and limitation-period calculations read, so a write using an unapproved synonym is
    rejected outright. That is why a conflict check can be relied on — it cannot silently
    miss rows recorded under a different name. Descriptive predicates are open, because a
    wrong subject-matter tag costs retrieval precision rather than a malpractice claim.
    """
    services, _ = _principal(ctx)
    onto = services.ontology

    def _pred(p: Any) -> dict[str, Any]:
        return {
            "id": p.id,
            "label": p.label,
            "description": p.description,
            "domain": list(p.domain),
            "range": list(p.range),
            "symmetric": p.symmetric,
        }

    return {
        "domain": onto.domain,
        "version": onto.version,
        "entity_types": [
            {"id": e.id, "label": e.label, "description": e.description}
            for e in onto.entities.values()
        ],
        "governing_predicates": [_pred(p) for p in onto.predicates.values() if p.governing],
        "descriptive_predicates": [
            _pred(p) for p in onto.predicates.values() if not p.governing
        ],
        "rules": [
            {
                "id": r.id,
                "version": r.version,
                "description": r.description,
                "when": list(r.when),
                "then": r.then,
                "min_premise_class": r.min_premise_class,
                "method": r.method,
            }
            for r in onto.rules
        ],
    }


async def search_assertions(
    ctx: Context,
    review_state: str | None = None,
    epistemic_class: str | None = None,
    matter_id: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """List raw individual facts, filtered. An inventory, not an answer.

    TRUST: mixed, and stated per row — which is the point of this tool. `ask` returns only
    facts that clear the firm's confidence floor and were signed off where sign-off was
    required. This returns everything the user may see, including claims still waiting for a
    human and claims below the floor. Read `review_state` and `below_floor` on every row
    before repeating it: a PENDING row is a model's unverified proposal, and relaying it as
    a finding is the exact failure the review queue exists to prevent.

    `epistemic_class` is how the system came to believe each fact:
      DECLARED        a system of record said so
      EXTRACTED_DET   a quote was mechanically confirmed to appear in the document
      EXTRACTED_MODEL a model's interpretation — needs a human before it counts
      INFERRED        derived by a rule; `get_provenance` returns the proof tree
      PREDICTED       a statistical guess, never an answer

    Use `ask` for questions. Use this to audit, or to see what is awaiting review.

    `withheld_matters` names the matters screened off from this user, each with the reason
    recorded and a contact. `withheld_matter_count` is how many. Non-zero means the view is
    incomplete, so "nothing found" is not the same as "nothing exists" — relay the names and
    the contact to your user rather than reporting a clean result.

    Args:
        review_state: AUTO_ASSERTED | PENDING | APPROVED | REJECTED. Omit for all.
        epistemic_class: One of the classes above. Omit for all.
        matter_id: Restrict to one matter. Omit for every matter the user may see.
        limit: Maximum rows, capped at 200.
    """
    services, auth_ctx = _principal(ctx)
    _validate_enum(review_state, ReviewState, "review_state")
    _validate_enum(epistemic_class, EpistemicClass, "epistemic_class")

    floor = services.settings_for(auth_ctx.tenant_id).min_confidence_floor
    # `visible` applies the tenant filter and the walls, so a screened `matter_id` narrows
    # to nothing. `withheld_matters` below is what stops that reading as "holds no facts".
    records = services.review_queue.visible(auth_ctx)

    if review_state:
        records = [r for r in records if r.assertion.review_state.value == review_state]
    if epistemic_class:
        records = [r for r in records if r.assertion.epistemic_class.value == epistemic_class]
    if matter_id:
        records = [r for r in records if r.assertion.matter_id == matter_id]

    records.sort(key=lambda r: r.assertion.confidence)
    return {
        "assertions": [
            _assertion_out(r.assertion, floor) for r in records[: min(limit, _MAX_ROWS)]
        ],
        "total": len(records),
        "confidence_floor": floor,
        # Mirrors the REST matters endpoint. Named, not counted: an agent that can only say
        # "one matter was withheld" cannot tell the lawyer which client to ask about, and
        # "no conflicts" read off a filtered list is the harm the wall exists to prevent.
        "withheld_matters": auth_ctx.withheld_matters(),
        "withheld_matter_count": len(auth_ctx.matter_denylist),
    }


def _validate_enum(value: str | None, enum: type, field: str) -> None:
    if value is None:
        return
    allowed = [m.value for m in enum]
    if value not in allowed:
        raise ToolError(f"{field} must be one of {allowed}, got {value!r}")


def _assertion_out(a: Any, floor: float) -> dict[str, Any]:
    return {
        "assertion_id": a.assertion_id,
        "subject_id": a.subject_id,
        "predicate": a.predicate,
        "object_id": a.object_id,
        "epistemic_class": a.epistemic_class.value,
        "method": a.method,
        "confidence": a.confidence,
        "review_state": a.review_state.value,
        "matter_id": a.matter_id,
        "recorded_at": a.recorded_at,
        "superseded_at": a.superseded_at,
        "is_current": a.is_current,
        "source": a.source_locator.to_dict(),
        "below_floor": a.confidence < floor,
    }


async def get_provenance(ctx: Context, assertion_id: str) -> dict[str, Any]:
    """Show why the system believes one fact — the document page and quote, or the proof tree.

    TRUST: this is the audit trail itself. For an extraction it returns the filename, the
    page and the verbatim quote, which is what a person would use to check it by hand: open
    that file at that page and search for that sentence. For an inference it returns the
    premise tree, recursively, so a derived fact unwinds into the facts it rests on.

    Call this before repeating any fact to a client. `explanation` is written for a
    non-engineer and can be relayed as-is.

    An id that does not exist or belongs to another firm produces the same answer, which is
    deliberate — a firm learns nothing about another firm. An id on a matter your user is
    screened from says so, naming the matter and a contact: a screen inside a firm is
    documented and acknowledged, and relaying it is what stops a lawyer reading a filtered
    result as a clean one.
    """
    services, auth_ctx = _principal(ctx)
    floor = services.settings_for(auth_ctx.tenant_id).min_confidence_floor

    record = _fetch(services, auth_ctx, assertion_id)
    a = record.assertion

    return {
        "assertion": _assertion_out(a, floor),
        "explanation": _explain(a),
        "rule_id": a.rule_id,
        "rule_version": a.rule_version,
        "premises": _premise_tree(services, auth_ctx, a.premises, floor, depth=0),
    }


def _fetch(services: Services, auth_ctx: AuthContext, assertion_id: str) -> Any:
    try:
        return services.review_queue.fetch(auth_ctx, assertion_id)
    except ScopeViolation as e:
        if e.is_screen:
            logger.info(
                "mcp provenance screened tenant=%s matter=%s", auth_ctx.tenant_id, e.matter_id
            )
            raise ToolError(str(e)) from e
        raise ToolError(_UNREACHABLE) from e
    except AssertionNotFound as e:
        logger.info("mcp provenance miss tenant=%s id=%s", auth_ctx.tenant_id, assertion_id)
        raise ToolError(_UNREACHABLE) from e


def _premise_tree(
    services: Services,
    auth_ctx: AuthContext,
    premise_ids: tuple[str, ...],
    floor: float,
    *,
    depth: int,
) -> list[dict[str, Any]]:
    if not premise_ids or depth >= _MAX_PREMISE_DEPTH:
        return []

    out: list[dict[str, Any]] = []
    for pid in premise_ids:
        try:
            record = services.review_queue.fetch(auth_ctx, pid)
        except ScopeViolation as e:
            # A premise the caller may not see. Reported as present-but-withheld rather
            # than dropped: a partial proof tree that looks complete is worse than one
            # that admits a gap. A screen also says which matter and who to ask, so the
            # gap is actionable rather than a dead end.
            gap: dict[str, Any] = {"assertion_id": pid, "visible": False}
            if e.is_screen:
                gap |= {"screened": True, "matter_id": e.matter_id, "contact": e.contact}
                gap["note"] = str(e)
            out.append(gap)
            continue
        except AssertionNotFound:
            out.append({"assertion_id": pid, "visible": False})
            continue
        out.append(
            {
                **_assertion_out(record.assertion, floor),
                "visible": True,
                "explanation": _explain(record.assertion),
                "premises": _premise_tree(
                    services, auth_ctx, record.assertion.premises, floor, depth=depth + 1
                ),
            }
        )
    return out


def _explain(a: Any) -> str:
    """Plain-language account of where a fact came from. Mirrors the REST provenance
    endpoint deliberately — one fact must not be explained two different ways."""
    cls = a.epistemic_class
    if cls is EpistemicClass.DECLARED:
        return (
            f"Recorded directly from a system of record ({a.method}). Not inferred or "
            "interpreted."
        )
    if cls is EpistemicClass.EXTRACTED_DET:
        return (
            f"A quote was confirmed to appear in the document by exact text match "
            f"({a.method}). That establishes only that the text is present, not what it "
            "means."
        )
    if cls is EpistemicClass.EXTRACTED_MODEL:
        return (
            f"Proposed by an AI model ({a.method}) reading the document. Requires human "
            "approval before it can influence an answer."
        )
    if cls is EpistemicClass.INFERRED:
        return (
            f"Derived by rule {a.rule_id} from {len(a.premises)} supporting fact(s). Its "
            "confidence cannot exceed that of its weakest premise."
        )
    return (
        "A statistical suggestion based on the shape of the graph, not on any document. "
        "Never used to answer questions."
    )


async def graph_neighbourhood(ctx: Context, node_id: str, depth: int = 2) -> dict[str, Any]:
    """Show the verified relationships around one entity, walking out a few hops.

    TRUST: filtered. Only edges the user may see, above the firm's confidence floor and
    signed off where sign-off was required — so a narrower view than `search_assertions`,
    and a defensible one. Every edge carries its `assertion_id`, so any of them can be
    taken to `get_provenance`.

    An empty result means "nothing verified and visible to you", which is not the same as
    "no such relationships". Call `search_assertions` and read `withheld_matters` to find out
    whether an ethical screen is narrowing what you can see here.

    Args:
        node_id: Entity id, e.g. `party:acme-corporation` or `document:d1`. `ask` and
            `search_assertions` return these as `subject_id` and `object_id`.
        depth: Hops to follow, 1-3. Deeper traversals fan out and pull in weakly related
            matters.
    """
    services, auth_ctx = _principal(ctx)
    if not 1 <= depth <= 3:
        raise ToolError("depth must be between 1 and 3")
    if services.graph_reader is None:
        raise ToolError("the graph is not reachable right now")

    settings = services.settings_for(auth_ctx.tenant_id)
    edges = services.graph_reader.expand(
        auth_ctx,
        [node_id],
        depth=depth,
        min_confidence=settings.min_confidence_floor,
    )

    node_ids = {node_id} | {e["subject_id"] for e in edges} | {e["object_id"] for e in edges}
    return {
        "nodes": [{"id": n} for n in sorted(node_ids)],
        "edges": edges,
        "confidence_floor": settings.min_confidence_floor,
    }


TOOLS = (
    ask,
    list_metrics,
    describe_ontology,
    search_assertions,
    get_provenance,
    graph_neighbourhood,
)


def build_server() -> FastMCP:
    """A fresh `FastMCP` with the tools registered.

    A factory rather than a module singleton because a `FastMCP`'s session manager may only
    be run once, so a second `create_app()` — which tests do — needs its own instance.
    """
    server: FastMCP = FastMCP(
        name="lexgraph",
        instructions=INSTRUCTIONS,
        host="0.0.0.0",
        stateless_http=True,
    )
    for fn in TOOLS:
        server.add_tool(fn)
    return server


def create_app(config: LexGraphConfig | None = None):
    """Build the ASGI app, sharing the same service container the REST API uses.

    Services are built here rather than imported from `src.api.app` so that importing this
    module does not start a FastAPI app. The container is the same shape either way, which
    is what keeps a tool and its matching endpoint answering identically.
    """
    services = build_services(config)
    set_services(services)
    if services.authenticator.dev_mode:
        logger.warning(
            "DEV AUTH BYPASS ACTIVE — unauthenticated MCP tool calls will be served as "
            "tenant %r",
            services.config.auth.dev_bypass_tenant,
        )
    return build_server().streamable_http_app()


app = create_app()

__all__ = ["TOOLS", "app", "build_server", "create_app"]
