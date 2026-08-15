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
declarative data in a YAML file that a non-programmer may edit, so the grammar is one edge
with typed endpoints and nothing else: no optional matches, no negation, no property filters,
no variable-length paths. Every one of those would be a feature request from a real firm and
every one changes what a proof tree means, so they are refused now rather than half-supported.

Types on the endpoints are optional in `then` (already bound) and expected in `when`, where
they document what the rule is about. They are not enforced against the entity list here:
an extractor names entities `party:acme`, and inferring a *kind* from an id prefix would be a
guess. The predicate's declared domain/range is where that check belongs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: `(var:Type)-[:PREDICATE]->(var:Type)`, with types optional and whitespace tolerated.
#: Anchored, so a pattern with anything trailing is rejected rather than partly read.
_EDGE = re.compile(
    r"^\s*\(\s*(?P<sv>[A-Za-z_][A-Za-z0-9_]*)\s*(?::\s*(?P<st>[A-Za-z_][A-Za-z0-9_]*)\s*)?\)"
    r"\s*-\s*\[\s*:\s*(?P<pred>[A-Za-z_][A-Za-z0-9_]*)\s*\]\s*->\s*"
    r"\(\s*(?P<ov>[A-Za-z_][A-Za-z0-9_]*)\s*(?::\s*(?P<ot>[A-Za-z_][A-Za-z0-9_]*)\s*)?\)\s*$"
)


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

    @property
    def variables(self) -> frozenset[str]:
        return frozenset({self.subject_var, self.object_var})


def parse_edge(pattern: str) -> EdgePattern:
    match = _EDGE.match(pattern or "")
    if match is None:
        raise PatternError(
            f"cannot read rule pattern {pattern!r}; expected (var:Type)-[:PREDICATE]->(var:Type)"
        )
    return EdgePattern(
        subject_var=match.group("sv"),
        predicate=match.group("pred"),
        object_var=match.group("ov"),
        subject_type=match.group("st"),
        object_type=match.group("ot"),
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


def parse_rule(rule: object) -> ParsedRule:
    """Read a `RuleDef` into something evaluable, refusing anything unsound.

    Four rejections, each because the alternative is a rule that appears to work:

    - **No premises.** `build_assertion` refuses an INFERRED assertion with no premises, so
      such a rule could never write anything.
    - **A single premise.** Not unsound, but every rule of this shape so far is a join, and a
      one-premise rule is a rename rather than an inference. Refused until something needs it.
    - **An unbound conclusion variable.** `then` may only use variables the premises bound;
      otherwise there is nothing to attach the conclusion to.
    - **No shared variable.** Premises that never join produce a cross product: every
      REPRESENTS paired with every ADVERSE_TO, flagging conflicts between unrelated parties.
      This is the one that would be actively dangerous.
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
    if not parsed.join_variables:
        raise PatternError(
            f"rule {rule_id!r} has no variable shared between premises, so it would match "
            "every combination of unrelated facts"
        )
    return parsed
