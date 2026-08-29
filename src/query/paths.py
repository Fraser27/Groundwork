"""Chains of verified relationships, assembled from the edges a walk already returned.

Why this exists: a reader handed `sam-parker PLACED_ORDER ord-4417`, `rtn-912 RETURN_OF ord-4417`
and `curtis APPROVED_RETURN rtn-912` as three rows among two hundred has to perform the join
itself, and on the answer path the only reader is a language model. Measured on the retail demo:
all three edges came back, none adjacent in the list, every one at `hops=1` with distractors
differing by four characters. Handing over one chain instead makes the join the graph's, which
makes it deterministic and gives every hop an `assertion_id` a person can take to `get_provenance`.

Assembled, never traversed. Every step comes from a hit list that already passed `edge_scope` and
the trust floor, so this module reads no store and holds no scope logic of its own -- which is also
why it lives here and not in `src/graph/`. A chain cannot contain a fact its reader could not
already see, and cannot widen an answer beyond the edges the walk was governed to return.

Not bounded by `graph_expand_depth`, and that is not an oversight: depth bounds how far the walk
travels from its seeds, while a chain is a connection *between* edges that came back. Two facts
each one hop from the same cited page are a two-hop connection to each other, and refusing to say
so would hide the thing the walk actually found.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

#: Longest chain assembled. Past four hops a chain stops being an argument anyone can follow, and
#: the enumeration to find it grows faster than its worth.
MAX_CHAIN_LENGTH = 4

#: Chains returned after ranking. A reader who needs the fiftieth is reading the flat edge list.
MAX_CHAINS = 12

#: Neighbours explored per node. A hub -- a matter with two hundred documents, a table with forty
#: columns -- would otherwise make enumeration quadratic in its degree for nothing: the two
#: hundredth edge off a hub says nothing the first twelve did not. Truncated by confidence, the same
#: principle as `expand`'s cut, so what survives is the strongest rather than an insertion order.
MAX_FANOUT = 12

#: Node expansions allowed in total. A backstop, not the working bound: `MAX_FANOUT` is what
#: actually shapes the search, and a real walk of a few hundred edges does not come near this.
#: Present because the cost is exponential in `MAX_CHAIN_LENGTH` and a pathological graph should
#: return fewer chains rather than stall the answer.
_VISIT_BUDGET = 100_000

#: Node kinds a chain may not pass through.
#:
#: Documents for the reason `expand` keeps them out of its frontier: a document node touches every
#: fact extracted from it, so a chain through one would report "these appeared in the same file" as
#: a relationship, and would multiply a single connection into one copy per citing document.
#: Provenance is not lost by excluding them -- each step carries its own `assertion_id`, which is
#: the honest route back to a page. Catalog nodes because `HAS_COLUMN` is schema, not evidence, and
#: a forty-column table would otherwise generate a chain for every pair of its columns.
_SKIP_PREFIXES = ("document:", "source:", "table:", "column:")


def _confidence(hit: dict[str, Any]) -> float:
    value = hit.get("confidence")
    return float(value) if isinstance(value, int | float) else 0.0


def _joinable(hit: Any) -> bool:
    if not isinstance(hit, dict) or not isinstance(hit.get("assertion_id"), str):
        return False
    subject, obj = hit.get("subject_id"), hit.get("object_id")
    if not isinstance(subject, str) or not isinstance(obj, str) or subject == obj:
        return False
    return not (subject.startswith(_SKIP_PREFIXES) or obj.startswith(_SKIP_PREFIXES))


def _step(hit: dict[str, Any], *, entered_from: str) -> dict[str, Any]:
    """One hop, with the direction the chain reads it in.

    `reversed` because the chain may traverse an edge against its own direction, and rendering
    `A REPRESENTS B` for a chain that went B to A would state the relationship backwards. The
    stored subject and object are left as they are, so the step is still the assertion as written.
    """
    return {
        "assertion_id": hit["assertion_id"],
        "subject_id": hit["subject_id"],
        "predicate": hit.get("predicate"),
        "object_id": hit["object_id"],
        "epistemic_class": hit.get("epistemic_class"),
        "confidence": _confidence(hit),
        "reversed": entered_from == hit["object_id"],
    }


def _canonical(nodes: Sequence[str]) -> tuple[str, ...]:
    """One key for a chain and the same chain walked backwards. A to C and C to A are one
    connection, and returning both would show a reader the same finding twice."""
    forward = tuple(nodes)
    return min(forward, tuple(reversed(forward)))


def _subchain_keys(nodes: Sequence[str]) -> set[tuple[str, ...]]:
    """Every contiguous stretch of two hops or more inside this chain.

    Used to suppress them: a four-hop chain contains three three-hop chains and six two-hop ones,
    and listing all ten would fill the panel with restatements of one connection.
    """
    keys: set[tuple[str, ...]] = set()
    for start in range(len(nodes)):
        for end in range(start + 3, len(nodes) + 1):
            if end - start < len(nodes):
                keys.add(_canonical(nodes[start:end]))
    return keys


def _adjacency(hits: Sequence[Any]) -> dict[str, list[tuple[str, dict[str, Any]]]]:
    adjacency: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for hit in hits:
        if not _joinable(hit):
            continue
        subject, obj = hit["subject_id"], hit["object_id"]
        adjacency.setdefault(subject, []).append((obj, hit))
        adjacency.setdefault(obj, []).append((subject, hit))
    for neighbours in adjacency.values():
        # Assertion id last so a tie resolves the same way twice: one question returning different
        # chains on two runs is indistinguishable from the graph having changed.
        neighbours.sort(key=lambda n: (-_confidence(n[1]), n[1]["assertion_id"]))
        del neighbours[MAX_FANOUT:]
    return adjacency


def chains(
    hits: Sequence[Any],
    *,
    max_length: int = MAX_CHAIN_LENGTH,
    limit: int = MAX_CHAINS,
) -> list[dict[str, Any]]:
    """Multi-hop connections inside `hits`, longest first, each one maximal.

    Only chains of two hops or more. A one-hop chain is the edge, which the flat list already shows,
    and presenting it as a connection would suggest the graph had found something it had not.

    A chain's confidence is its weakest hop. Any other number would overstate it -- a chain is an
    argument, and an argument is as good as its worst step. This is also what keeps presence claims
    out of the top of the list without a second rule: `MENTIONS` sits exactly on the trust floor by
    design, so a chain resting on one sorts below every chain that does not.
    """
    adjacency = _adjacency(hits)
    if not adjacency:
        return []

    found: dict[tuple[str, ...], tuple[list[str], list[dict[str, Any]]]] = {}
    budget = _VISIT_BUDGET

    def walk(node: str, nodes: list[str], steps: list[dict[str, Any]]) -> None:
        nonlocal budget
        if len(steps) >= 2:
            found.setdefault(_canonical(nodes), (list(nodes), list(steps)))
        if len(steps) >= max_length or budget <= 0:
            return
        for other, hit in adjacency.get(node, ()):
            if other in nodes:
                continue
            budget -= 1
            if budget <= 0:
                return
            nodes.append(other)
            steps.append(_step(hit, entered_from=node))
            walk(other, nodes, steps)
            nodes.pop()
            steps.pop()

    for start in sorted(adjacency):
        walk(start, [start], [])

    # Longest first, because a long connection is the thing the flat list could not show. Within a
    # length the weakest hop decides, then the first id so the order is reproducible.
    ranked = sorted(
        found.values(),
        key=lambda c: (-len(c[1]), -min(s["confidence"] for s in c[1]), c[1][0]["assertion_id"]),
    )

    kept: list[dict[str, Any]] = []
    covered: set[tuple[str, ...]] = set()
    for nodes, steps in ranked:
        if _canonical(nodes) in covered:
            continue
        covered |= _subchain_keys(nodes)
        kept.append(
            {
                "nodes": list(nodes),
                "steps": list(steps),
                "confidence": min(s["confidence"] for s in steps),
            }
        )
        if len(kept) >= limit:
            break
    return kept
