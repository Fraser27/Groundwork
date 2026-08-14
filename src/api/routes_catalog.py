"""Read surfaces: dashboard, matters, ontology, graph neighbourhood.

The matters endpoint is the one with a subtlety worth reading: screened matters are
named, with a reason and a contact, in a list of their own. Never in `matters` — a
caller that iterates the visible list must not be able to reach a screened matter by
accident, which is why this is two lists rather than one with a flag.
"""

from __future__ import annotations

from collections import Counter
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, status

from src.api.deps import ServicesDep, TenantDep
from src.graph.assertions import EpistemicClass, ReviewState
from src.ontology.loader import load_ontology

router = APIRouter(tags=["catalog"])


@router.get("/tenants/{tenant}/dashboard")
async def dashboard(services: ServicesDep, principal: TenantDep) -> dict[str, Any]:
    ctx, grants = principal
    settings = services.settings_for(ctx.tenant_id)
    records = services.review_queue.visible(ctx)

    by_class = Counter(r.assertion.epistemic_class.value for r in records if r.is_current)
    by_state = Counter(r.assertion.review_state.value for r in records if r.is_current)

    return {
        "tenant_id": ctx.tenant_id,
        "assertions_by_class": {c.value: by_class.get(c.value, 0) for c in EpistemicClass},
        "assertions_by_review_state": {
            s.value: by_state.get(s.value, 0) for s in ReviewState
        },
        "pending_review": by_state.get(ReviewState.PENDING.value, 0),
        "retracted": sum(1 for r in records if not r.is_current),
        "confidence_floor": settings.min_confidence_floor,
        "ontology_domain": settings.ontology_domain,
        "can_review": grants.can_review,
        "kill_switch_active": settings.block_ungoverned_queries,
    }


@router.get("/tenants/{tenant}/matters")
async def list_matters(services: ServicesDep, principal: TenantDep) -> dict[str, Any]:
    """Matters the caller may see, and — named — those they are screened from.

    Naming them is the point. Reading "no conflicts found" when the truth is "none that
    you can see" is how an ethical wall causes the harm it exists to prevent, and a bare
    count does not tell a lawyer which client to go and ask about.
    """
    ctx, _ = principal
    records = services.review_queue.visible(ctx)

    matter_ids = {r.assertion.matter_id for r in records if r.assertion.matter_id}
    matters = [
        {
            "matter_id": mid,
            "assertion_count": sum(1 for r in records if r.assertion.matter_id == mid),
        }
        for mid in sorted(matter_ids)
    ]

    # A separate list, never a flag on a row in `matters`: no caller can then treat a
    # screened matter as readable by forgetting to check the flag. `visible()` has
    # already dropped them, so these come from the grant.
    withheld = ctx.withheld_matters()

    return {"matters": matters, "withheld": withheld, "withheld_count": len(withheld)}


@router.get("/ontology/{domain}")
async def get_ontology(domain: str) -> dict[str, Any]:
    """The vocabulary, including the help text the UI shows for each term."""
    try:
        onto = load_ontology(domain)
    except FileNotFoundError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e)) from e

    def _pred(p: Any) -> dict[str, Any]:
        return {
            "id": p.id,
            "label": p.label,
            "description": p.description,
            "help": p.help,
            "governing": p.governing,
            "domain": list(p.domain),
            "range": list(p.range),
            "symmetric": p.symmetric,
        }

    # Split rather than flat, and named to match the UI's `Ontology` type. The split
    # is the point: "which predicates are closed?" is the question an administrator
    # actually asks, and a flat list with a boolean makes them work it out.
    return {
        "domain": onto.domain,
        "version": onto.version,
        "entity_types": [
            {
                "id": e.id,
                "label": e.label,
                "description": e.description,
                "help": e.help,
            }
            for e in onto.entities.values()
        ],
        "governing_predicates": [
            _pred(p) for p in onto.predicates.values() if p.governing
        ],
        "descriptive_predicates": [
            _pred(p) for p in onto.predicates.values() if not p.governing
        ],
        "rules": [
            {
                "id": r.id,
                "version": r.version,
                "description": r.description,
                "help": r.help,
                "when": list(r.when),
                "then": r.then,
                "min_premise_class": r.min_premise_class,
                "method": r.method,
            }
            for r in onto.rules
        ],
    }


@router.get("/tenants/{tenant}/graph/neighbourhood")
async def neighbourhood(
    services: ServicesDep,
    principal: TenantDep,
    node_id: Annotated[str, Query()],
    depth: Annotated[int, Query(ge=1, le=3)] = 2,
) -> dict[str, Any]:
    """Edges around a node, for the graph explorer.

    Built from the assertion store rather than Cypher while the graph writer is
    unfinished, so the shape is the same once it is swapped for a scoped read.
    """
    ctx, _ = principal
    settings = services.settings_for(ctx.tenant_id)
    records = [r for r in services.review_queue.visible(ctx) if r.is_current]

    frontier = {node_id}
    seen_nodes: set[str] = set()
    edges: list[dict[str, Any]] = []

    for _ in range(depth):
        next_frontier: set[str] = set()
        for r in records:
            a = r.assertion
            if a.subject_id in frontier or a.object_id in frontier:
                edges.append(
                    {
                        "assertion_id": a.assertion_id,
                        "source": a.subject_id,
                        "target": a.object_id,
                        "predicate": a.predicate,
                        "epistemic_class": a.epistemic_class.value,
                        "confidence": a.confidence,
                        "review_state": a.review_state.value,
                        "below_floor": a.confidence < settings.min_confidence_floor,
                        # Drawn more prominently: these are the edges a conflict check
                        # or a privilege wall actually reads.
                        "governing": services.ontology.is_governing(a.predicate),
                        "matter_id": a.matter_id,
                    }
                )
                next_frontier.update({a.subject_id, a.object_id})
        seen_nodes |= frontier
        frontier = next_frontier - seen_nodes
        if not frontier:
            break

    node_ids = seen_nodes | frontier
    unique_edges = {e["assertion_id"]: e for e in edges}

    # Entity ids are `kind:slug`, so the label and type come free. The renderer sizes
    # and colours nodes by `type` and needs a readable `label` — without them it drew
    # unlabelled dots and no edges, because `NODE_RADIUS[undefined]` is undefined.
    return {
        "nodes": [_node(n) for n in sorted(node_ids)],
        "edges": list(unique_edges.values()),
        "confidence_floor": settings.min_confidence_floor,
    }


def _node(entity_id: str) -> dict[str, Any]:
    kind, _, rest = entity_id.partition(":")
    if not rest:
        kind, rest = "entity", entity_id
    return {
        "id": entity_id,
        "type": kind,
        "label": rest.replace("-", " ").replace("_", " "),
    }
