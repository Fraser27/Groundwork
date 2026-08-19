"""Parsing the `when`/`then` patterns a rule is written in.

A rule in a pack looks like this:

    when:
      - "(c:Counsel)-[:REPRESENTS]->(p:Party)"
      - "(m:Matter)-[:ADVERSE_TO]->(p:Party)"
    then: "(m)-[:POTENTIAL_CONFLICT]->(p)"

Shared variables are the join. `p` appears in both premises, so the rule fires only where the
*same* party is on both sides — which is what makes it a conflict rather than two unrelated
facts. `then` reuses `m` and `p`, so the conclusion is bound to the entities that matched.

This is deliberately a small subset of Cypher's syntax and not Cypher itself. A rule is
declarative data in a YAML file that a non-programmer may edit, so the grammar stays close to
one edge with typed endpoints: no optional matches, no negation, no property filters. Each of
those would be a feature request from a real firm, and each rests a conclusion on the *absence*
of a fact — which has no assertion id, so a proof tree citing it would have a hole where its
reason should be.

Variable-length paths were refused alongside them and are not any more, because they fail that
test differently: every step of a path is an existing, signed-off assertion, so the proof tree
stays whole. They are allowed narrowly — over a predicate the pack declares `transitive`,
bounded by `MAX_PATH_HOPS`, and with a lower bound of exactly one hop.

An intermediate *fact* was the bigger gap, and it needed no syntax at all: `AFFILIATE_OF` plus
a three-premise rule already finds a conflict through a group company. Paths earn their keep
only where the chain's length is unknown — an ownership ladder, a run of `OVERRULES`.

Types on the endpoints are optional in `then` (already bound) and expected in `when`. They are
not checked *here* — this module only reads the grammar — but they are no longer merely
documentation, and the split is worth stating because the two halves catch different things:

- A **premise** type is enforced by the reasoner, which drops a candidate fact whose endpoint is
  the wrong kind. `conflict_check` writes `(m:Matter)-[:ADVERSE_TO]->(p:Party)` while
  `ADVERSE_TO` also legally accepts a Party subject, so without that filter the rule bound `m`
  to whichever the extractor happened to produce — and a conflict check whose correctness is a
  coin flip is the failure this codebase most needs to avoid.
- A **conclusion**'s endpoints are enforced by `build_assertion`, against the predicate's
  declared domain and range. That is the backstop: it holds for every write, including one from
  a rule whose premises carry no types at all.

This once said inferring a kind from an id prefix "would be a guess". It is not: `entity_kinds`
is closed, `canonical_entity_id` mints the prefix, `build_assertion` refuses a miscased one, and
the extractor drops a claim whose kind the pack does not declare. Every edge carries a declared
kind on both ends by construction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: `(var:Type)-[:PREDICATE]->(var:Type)`, with types optional and whitespace tolerated.
#: `*1..N` after the predicate makes it a bounded path. Anchored, so a pattern with anything
#: trailing is rejected rather than partly read.
_EDGE = re.compile(
    r"^\s*\(\s*(?P<sv>[A-Za-z_][A-Za-z0-9_]*)\s*(?::\s*(?P<st>[A-Za-z_][A-Za-z0-9_]*)\s*)?\)"
    r"\s*-\s*\[\s*:\s*(?P<pred>[A-Za-z_][A-Za-z0-9_]*)\s*"
    r"(?:\*\s*(?P<min>\d+)\s*\.\.\s*(?P<max>\d+)\s*)?\]\s*->\s*"
    r"\(\s*(?P<ov>[A-Za-z_][A-Za-z0-9_]*)\s*(?::\s*(?P<ot>[A-Za-z_][A-Za-z0-9_]*)\s*)?\)\s*$"
)

#: How far a path premise may walk. Three, because a proof tree has to be readable: `premises`
#: is a flat tuple, so a reader sees N sibling assertions rather than a rendered chain, and past
#: three that stops being something to defend in front of a regulator.
#:
#: Fixed here rather than configurable per tenant. A rule's shape is baked into `method`
#: (`rule:x@v1`) which `_compute_id` hashes, so a per-tenant depth would let one tenant's `@v1`
#: mean something different from another's with no version change to show for it.
MAX_PATH_HOPS = 3


class PatternError(ValueError):
    """A rule pattern that cannot be read.

    Raised at pack load, never at inference time. A rule nobody can parse is a rule that
    would silently never fire, and a conflict check that never fires looks exactly like a
    clean conflict check.
    """


@dataclass(frozen=True)
class EdgePattern:
    """One `(subject)-[:PREDICATE]->(object)` triple from a rule."""

    subject_var: str
    predicate: str
    object_var: str
    subject_type: str | None = None
    object_type: str | None = None

    min_hops: int = 1
    max_hops: int = 1
    """How many edges this premise spans. Both 1 for an ordinary edge, so the common case is
    literally a path of length one and needs no separate code path."""

    @property
    def variables(self) -> frozenset[str]:
        return frozenset({self.subject_var, self.object_var})

    @property
    def is_path(self) -> bool:
        return self.max_hops > 1


def parse_edge(pattern: str) -> EdgePattern:
    match = _EDGE.match(pattern or "")
    if match is None:
        raise PatternError(
            f"cannot read rule pattern {pattern!r}; expected (var:Type)-[:PREDICATE]->(var:Type)"
        )
    min_hops = int(match.group("min") or 1)
    max_hops = int(match.group("max") or 1)
    if min_hops != 1:
        # `*2..3` asserts the absence of a direct edge, which is negation wearing a path's
        # clothes: nothing in the graph witnesses "there is no one-hop link", so no assertion id
        # can stand for it and the proof tree would have a hole where its reason should be.
        raise PatternError(
            f"path pattern {pattern!r} starts at {min_hops} hops; the lower bound must be 1, "
            "because requiring a longer path is a claim that no shorter one exists and nothing "
            "in the graph can evidence that"
        )
    if max_hops > MAX_PATH_HOPS:
        raise PatternError(
            f"path pattern {pattern!r} allows {max_hops} hops; at most {MAX_PATH_HOPS} is "
            "readable in a proof tree, which is what a conclusion has to be defended from"
        )
    return EdgePattern(
        subject_var=match.group("sv"),
        predicate=match.group("pred"),
        object_var=match.group("ov"),
        subject_type=match.group("st"),
        object_type=match.group("ot"),
        min_hops=min_hops,
        max_hops=max_hops,
    )


@dataclass(frozen=True)
class ParsedRule:
    """A rule with its patterns read, ready to evaluate."""

    rule_id: str
    version: str
    premises: tuple[EdgePattern, ...]
    conclusion: EdgePattern
    min_premise_class: str
    description: str = ""

    @property
    def method(self) -> str:
        return f"rule:{self.rule_id}@{self.version}"

    @property
    def join_variables(self) -> frozenset[str]:
        """Variables appearing in more than one premise. These are what make it a join."""
        seen: dict[str, int] = {}
        for premise in self.premises:
            for var in premise.variables:
                seen[var] = seen.get(var, 0) + 1
        return frozenset(v for v, n in seen.items() if n > 1)

    @property
    def disconnected_premises(self) -> tuple[int, ...]:
        """Indices of premises with no chain of shared variables back to the first.

        `join_variables` being non-empty is not the same test once there are three premises:
        `(c,q) (q,m) (x,y)` shares `q` and so passes it, while `(x,y)` joins nothing and
        cross-products against every match. Reachability is the property actually wanted.
        """
        reached = set(self.premises[0].variables)
        remaining = list(enumerate(self.premises[1:], start=1))
        grew = True
        while grew:
            grew = False
            for item in list(remaining):
                if self.premises[item[0]].variables & reached:
                    reached |= self.premises[item[0]].variables
                    remaining.remove(item)
                    grew = True
        return tuple(i for i, _ in remaining)


def parse_rule(rule: object) -> ParsedRule:
    """Read a `RuleDef` into something evaluable, refusing anything unsound.

    Four rejections, each because the alternative is a rule that appears to work:

    - **No premises.** `build_assertion` refuses an INFERRED assertion with no premises, so
      such a rule could never write anything.
    - **A single premise.** Not unsound, but every rule of this shape so far is a join, and a
      one-premise rule is a rename rather than an inference. Refused until something needs it.
    - **An unbound conclusion variable.** `then` may only use variables the premises bound;
      otherwise there is nothing to attach the conclusion to.
    - **A premise that does not join.** Premises that never join produce a cross product: every
      REPRESENTS paired with every ADVERSE_TO, flagging conflicts between unrelated parties.
      This is the one that would be actively dangerous. Every premise must reach the first
      through shared variables, not merely share one with *some* other premise — at two
      premises those coincide, at three they do not.
    """
    rule_id = getattr(rule, "id", "")
    premises = tuple(parse_edge(p) for p in getattr(rule, "when", ()) or ())
    if not premises:
        raise PatternError(f"rule {rule_id!r} has no `when` patterns, so it can never fire")
    if len(premises) < 2:
        raise PatternError(
            f"rule {rule_id!r} has one premise; a rule with nothing to join is a rename, "
            "not an inference"
        )

    conclusion = parse_edge(getattr(rule, "then", ""))
    if conclusion.is_path:
        # A conclusion is one edge the rule writes. `*1..3` there would ask for an unspecified
        # number of edges between two bound endpoints, which names no particular fact.
        raise PatternError(
            f"rule {rule_id!r} concludes a path; `then` writes exactly one edge, so a hop range "
            "there does not name a fact to write"
        )
    bound = frozenset().union(*(p.variables for p in premises))
    unbound = conclusion.variables - bound
    if unbound:
        raise PatternError(
            f"rule {rule_id!r} concludes about {sorted(unbound)}, which no premise binds"
        )

    parsed = ParsedRule(
        rule_id=rule_id,
        version=getattr(rule, "version", "v1"),
        premises=premises,
        conclusion=conclusion,
        min_premise_class=getattr(rule, "min_premise_class", "EXTRACTED_DET"),
        description=getattr(rule, "description", ""),
    )
    stranded = parsed.disconnected_premises
    if stranded:
        raise PatternError(
            f"rule {rule_id!r} has premises {[getattr(rule, 'when', ())[i] for i in stranded]} "
            "sharing no variable with the rest, so they would match every combination of "
            "unrelated facts"
        )
    return parsed
