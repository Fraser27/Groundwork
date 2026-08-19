"""The reasoner: one engine, rules declared per domain.

Domain-agnostic by construction, which is the whole point. This module contains no legal or
clinical knowledge whatsoever — it reads `when`/`then` patterns out of whichever ontology pack
the tenant runs and evaluates them. The legal pack's conflict check and the healthcare pack's
contraindication alert are the *same* code path over different data:

    legal:      REPRESENTS + ADVERSE_TO   joined on the party      -> POTENTIAL_CONFLICT
    legal:      CITES + OVERRULES         joined on the authority  -> RELIES_ON_STALE_AUTHORITY
    healthcare: PRESCRIBED + ALLERGIC_TO  joined on the medication -> CONTRAINDICATION_ALERT

A domain-specific reasoner would mean a new pack needs new Python, which would make
"domain-agnostic" marketing rather than architecture.

**Why the conclusions matter.** A conflict is almost never visible in one document: the firm
acts for Northwind against Calder in one matter, and a facility agreement filed under another
matter mentions in passing that its client holds 18% of Calder. Neither document is wrong;
together they say the firm is attacking a company its own client part-owns. The same shape
holds for stale authority — advice correct when written, resting on a case another document
records as overruled. These conclusions exist only in the join, which is why conflict checking
is definitionally cross-matter and why a tenant gets one graph rather than one per matter.

**Three constraints inherited rather than reimplemented:**

- Every inference carries its premises, so `build_assertion` refuses one without them and the
  proof tree always resolves to the documents underneath.
- An inference is never more confident than its weakest premise. `build_assertion` enforces
  the ceiling; this module supplies the premise confidences that define it.
- **Only signed-off facts are premises.** A conflict flag resting on an unreviewed model guess
  would be worse than no flag, because someone would rely on it. `min_premise_class` alone
  cannot express this: a governing predicate may never be EXTRACTED_DET (a quote match proves
  presence, not significance), so a floor of EXTRACTED_DET would mean these rules could never
  fire on anything read from a document at all. Review state is the gate that actually
  distinguishes "a person approved this" from "a model proposed it", so the engine requires
  both — a class at or above the floor, *and* a fact that is live rather than pending.

Evaluation is a hash join in Python over live assertions, not Cypher. Deliberate, for now: a
rule is a join over a tenant's *approved* facts, that set is already in memory on the read
path, and expressing the same thing as generated Cypher would put query construction outside
`src/graph/` where the working agreement forbids it. Facts are indexed by predicate once per
pass, so a rule sees only the facts it could match rather than all of them; it will still need
revisiting when a tenant's live graph outgrows a process, and the interface here does not
change when it does.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.graph.assertions import (
    Assertion,
    EpistemicClass,
    ReviewState,
    SourceLocator,
    build_assertion,
)
from src.graph.scope import AuthContext
from src.ontology.patterns import ParsedRule, PatternError, parse_rule

logger = logging.getLogger(__name__)

#: Trust order, weakest last. `min_premise_class` names a floor, so a rule accepts its own
#: class and anything stronger. Written out rather than derived from the enum's order so that
#: adding a class cannot silently reorder what a rule will fire on.
_STRENGTH: dict[EpistemicClass, int] = {
    EpistemicClass.DECLARED: 0,
    EpistemicClass.EXTRACTED_DET: 1,
    EpistemicClass.EXTRACTED_MODEL: 2,
    EpistemicClass.INFERRED: 3,
    EpistemicClass.PREDICTED: 4,
}

#: A conclusion is only as good as its weakest premise, and it should not be presented as
#: equal to it either: an inference adds a step, and a step can be wrong even when every
#: premise is right. Applied on top of the ceiling `build_assertion` enforces, and once per
#: edge crossed — so a chain is less certain than a direct join, which is the honest ordering.
INFERENCE_CONFIDENCE_FACTOR = 0.95


#: Review states a rule will accept as a premise. AUTO_ASSERTED covers facts that needed no
#: person (a system of record said so, or a check confirmed presence); APPROVED covers the ones
#: a person signed off. PENDING is deliberately absent: firing on it would let an unreviewed
#: model guess produce a conflict flag someone then relies on.
SIGNED_OFF = frozenset({ReviewState.AUTO_ASSERTED, ReviewState.APPROVED})


def accepts(assertion: Assertion, floor: str) -> bool:
    """Whether a fact is strong enough to be a premise for a rule requiring `floor`.

    Two independent conditions, because they catch different things. The epistemic floor is
    about *how* the fact was reached; the review state is about *whether anyone stands behind
    it*. A pack can loosen the first, but no pack setting lets a rule fire on something
    unreviewed.

    An unknown floor is treated as the strictest, not the loosest: a typo must not quietly
    widen what a rule fires on.
    """
    if assertion.review_state not in SIGNED_OFF:
        return False
    if assertion.superseded_at is not None:
        # Withdrawn. A conclusion must not outlive the reason it was drawn.
        return False
    try:
        required = _STRENGTH[EpistemicClass(floor)]
    except (KeyError, ValueError):
        logger.warning("unknown min_premise_class %r, requiring DECLARED", floor)
        required = _STRENGTH[EpistemicClass.DECLARED]
    return _STRENGTH[assertion.epistemic_class] <= required


@dataclass(frozen=True)
class Inference:
    """One conclusion, with everything needed to defend it."""

    assertion: Assertion
    rule_id: str
    premise_ids: tuple[str, ...]

    @property
    def explanation(self) -> str:
        return f"{self.rule_id} fired on {len(self.premise_ids)} premises"


@dataclass
class ReasonerReport:
    """What a pass did, in terms an administrator can read."""

    inferences: list[Inference] = field(default_factory=list)
    rules_evaluated: int = 0
    rules_skipped: dict[str, str] = field(default_factory=dict)
    facts_considered: int = 0

    conclusions_refused: list[str] = field(default_factory=list)
    """Matches an invariant refused, one line each.

    Distinct from `rules_skipped`, which is a rule that could never fire. This is a rule that
    fired and whose conclusion the contract rejected — which became an ordinary outcome once
    declared endpoint kinds were enforced, since a rule can bind a variable to the wrong sort of
    thing. Reported rather than only logged: a rule that draws nothing must not look like a rule
    that found nothing."""

    @property
    def count(self) -> int:
        return len(self.inferences)

    def to_dict(self) -> dict[str, Any]:
        return {
            "inferences": [
                {
                    "assertion_id": i.assertion.assertion_id,
                    "rule_id": i.rule_id,
                    "subject_id": i.assertion.subject_id,
                    "predicate": i.assertion.predicate,
                    "object_id": i.assertion.object_id,
                    "confidence": i.assertion.confidence,
                    "premises": list(i.premise_ids),
                    "matter_id": i.assertion.matter_id,
                }
                for i in self.inferences
            ],
            "count": self.count,
            "rules_evaluated": self.rules_evaluated,
            "rules_skipped": self.rules_skipped,
            "conclusions_refused": list(self.conclusions_refused),
            "facts_considered": self.facts_considered,
        }


def _binding_key(pattern_vars: tuple[str, ...], binding: dict[str, str]) -> tuple[str, ...]:
    return tuple(binding[v] for v in pattern_vars)


class Reasoner:
    """Evaluates a pack's rules over a tenant's live facts."""

    def __init__(self, ontology: Any) -> None:
        self.ontology = ontology
        self._rules: list[ParsedRule] = []
        self._skipped: dict[str, str] = {}
        for rule in getattr(ontology, "rules", ()):
            try:
                self._rules.append(parse_rule(rule))
            except PatternError as e:
                # Reported, never fatal. One unreadable rule must not stop the others from
                # firing, and a rule that cannot be parsed is recorded so it does not simply
                # look like a rule that found nothing.
                self._skipped[getattr(rule, "id", "?")] = str(e)
                logger.warning("skipping rule: %s", e)

    @property
    def rules(self) -> list[ParsedRule]:
        return list(self._rules)

    def run(self, ctx: AuthContext, facts: list[Assertion]) -> ReasonerReport:
        """Evaluate every rule over `facts`, returning conclusions without writing them.

        Writing is the caller's job, through `ReviewQueue.stage`, so the review gate applies
        to inferences exactly as it does to extractions. A reasoner that wrote straight to the
        graph would be a code path that opts out of review.
        """
        report = ReasonerReport(
            rules_evaluated=len(self._rules),
            rules_skipped=dict(self._skipped),
            facts_considered=len(facts),
        )
        known = self.ontology.governing_predicates | self.ontology.descriptive_predicates
        # Built once per pass rather than rescanned per premise. `accepts` is deliberately not
        # applied here: the floor is per rule, so folding it in would need one index per floor.
        by_predicate: dict[str, list[Assertion]] = {}
        for fact in facts:
            by_predicate.setdefault(fact.predicate, []).append(fact)

        for rule in self._rules:
            if rule.conclusion.predicate not in known:
                # Should be caught at pack load. Kept because writing an undeclared predicate
                # would be rejected by build_assertion anyway, and a clear skip reason beats
                # an exception from two layers down.
                report.rules_skipped[rule.rule_id] = (
                    f"{rule.conclusion.predicate} is not in the {self.ontology.domain} vocabulary"
                )
                continue
            for inference in self._fire(ctx, rule, by_predicate, known, report):
                report.inferences.append(inference)

        if report.count:
            logger.info(
                "inferred %d facts for %s from %d rules",
                report.count,
                ctx.tenant_id,
                len(self._rules),
            )
        return report

    def _fire(
        self,
        ctx: AuthContext,
        rule: ParsedRule,
        by_predicate: dict[str, list[Assertion]],
        known: frozenset[str],
        report: ReasonerReport,
    ) -> list[Inference]:
        """Join the premises and build one conclusion per distinct match.

        Bindings are grown one premise at a time: candidates for the next pattern are filtered
        to those agreeing with what is already bound, which is what makes the shared variable a
        join rather than a cross product.
        """
        # (bindings, premise assertions) pairs. Starts with one empty binding so the first
        # premise is grown by the same code as the rest -- a path premise may come first.
        partial: list[tuple[dict[str, str], list[Assertion]]] = [({}, [])]

        for pattern in rule.premises:
            candidates = [
                f
                for f in by_predicate.get(pattern.predicate, ())
                if accepts(f, rule.min_premise_class)
            ]
            grown: list[tuple[dict[str, str], list[Assertion]]] = []
            for binding, used in partial:
                for start, end, chain in self._match(pattern, candidates):
                    if binding.get(pattern.subject_var, start) != start:
                        continue
                    if binding.get(pattern.object_var, end) != end:
                        continue
                    spent = {u.assertion_id for u in used}
                    if any(f.assertion_id in spent for f in chain):
                        # A fact cannot be two different premises of one conclusion. Without
                        # this a symmetric predicate joins to itself and the rule "proves"
                        # something from a single fact. Across the whole chain, not just its
                        # first edge, or a path could re-use an edge another premise already
                        # spent and the conclusion would rest on fewer facts than it cites.
                        continue
                    merged = dict(binding)
                    merged[pattern.subject_var] = start
                    merged[pattern.object_var] = end
                    grown.append((merged, [*used, *chain]))
            partial = grown
            if not partial:
                return []

        # Shortest first, so where two chains reach one conclusion the cited proof is the
        # tightest one. `premise_ids` breaks the tie, because two runs of one conflict check
        # citing different premises is indistinguishable from the graph having changed.
        partial.sort(key=lambda row: (len(row[1]), tuple(f.assertion_id for f in row[1])))

        out: list[Inference] = []
        seen: set[tuple[str, ...]] = set()
        conclusion_vars = (rule.conclusion.subject_var, rule.conclusion.object_var)
        for binding, used in partial:
            key = _binding_key(conclusion_vars, binding)
            if key in seen:
                # The same conclusion reached by different premise sets. One edge, not two:
                # a second identical claim adds noise, and `assertion_id` is content-addressed
                # so it would collapse in the store anyway.
                continue
            seen.add(key)
            inferred = self._build(ctx, rule, binding, used, known, report)
            if inferred is not None:
                out.append(inferred)
        return out

    def _match(
        self, pattern: Any, candidates: list[Assertion]
    ) -> list[tuple[str, str, list[Assertion]]]:
        """Every (start, end, facts) this premise matches: one edge, or a chain of them.

        An ordinary premise is `max_hops == 1`, so it is literally the one-hop case of the walk
        and needs no separate branch.
        """
        if not pattern.is_path:
            return [(f.subject_id, f.object_id, [f]) for f in candidates]

        # (start, current end, facts, nodes on this path). `visited` is per path rather than
        # global, mirroring `expand()`: two paths may legitimately share a node, but one path
        # revisiting a node is a cycle, and `A OVERRULES B OVERRULES A` would otherwise walk to
        # the hop cap and conclude something about A from A.
        walks: list[tuple[str, str, list[Assertion], set[str]]] = [
            (f.subject_id, f.object_id, [f], {f.subject_id, f.object_id}) for f in candidates
        ]
        out: list[tuple[str, str, list[Assertion]]] = [(s, e, c) for s, e, c, _ in walks]

        for _ in range(pattern.max_hops - 1):
            extended: list[tuple[str, str, list[Assertion], set[str]]] = []
            for start, end, chain, visited in walks:
                spent = {f.assertion_id for f in chain}
                for f in candidates:
                    if f.subject_id != end or f.assertion_id in spent or f.object_id in visited:
                        continue
                    extended.append((start, f.object_id, [*chain, f], visited | {f.object_id}))
            if not extended:
                break
            out.extend((s, e, c) for s, e, c, _ in extended)
            walks = extended
        return out

    def _build(
        self,
        ctx: AuthContext,
        rule: ParsedRule,
        binding: dict[str, str],
        premises: list[Assertion],
        known: frozenset[str],
        report: ReasonerReport,
    ) -> Inference | None:
        subject = binding[rule.conclusion.subject_var]
        obj = binding[rule.conclusion.object_var]
        confidences = tuple(p.confidence for p in premises)
        premise_ids = tuple(p.assertion_id for p in premises)

        # One step of doubt per edge crossed, not per rule fired. A walked chain is more
        # inferential than a direct join -- more places for a correct premise to lead to a wrong
        # conclusion -- and applying the factor once meant a 3-hop conclusion was presented as
        # firmly as a 1-hop one. `extra_hops` is 0 for every non-path rule, so their arithmetic
        # is unchanged and `assertion_id` values already in the store still reconcile.
        extra_hops = len(premises) - len(rule.premises)
        confidence = min(confidences) * INFERENCE_CONFIDENCE_FACTOR ** (1 + extra_hops)

        # The matter is inherited only when every premise agrees. A conclusion drawn *across*
        # two matters belongs to neither: stamping it with one would hide it from the other,
        # and a cross-matter conflict hidden from one of its own matters is the failure the
        # whole design is trying to prevent.
        matters = {p.matter_id for p in premises}
        matter_id = matters.pop() if len(matters) == 1 else None

        try:
            assertion = build_assertion(
                tenant_id=ctx.tenant_id,
                subject_id=subject,
                predicate=rule.conclusion.predicate,
                object_id=obj,
                epistemic_class=EpistemicClass.INFERRED,
                method=rule.method,
                confidence=confidence,
                # A rule read no document, so there is no page to cite. The locator names the
                # rule as the source and the premises carry the documents, which is what the
                # proof tree unwinds into. Claiming a page here would be a fabricated citation.
                source_locator=SourceLocator(
                    source_id="reasoner", table=rule.rule_id, column=rule.version
                ),
                matter_id=matter_id,
                premises=premise_ids,
                premise_confidences=confidences,
                rule_id=rule.rule_id,
                rule_version=rule.version,
                allowed_predicates=known,
                endpoint_kinds=self.ontology.endpoint_kinds(rule.conclusion.predicate),
            )
        except Exception as e:
            # An invariant refused it. Skipped rather than raised: one bad conclusion must not
            # abort a pass that is otherwise producing good ones. Reported as well as logged,
            # because with endpoint kinds enforced this is now an ordinary outcome -- a rule can
            # bind a variable to the wrong sort of thing -- and a rule that draws nothing must not
            # be indistinguishable from a rule that found nothing.
            report.conclusions_refused.append(f"{rule.rule_id}: {e}")
            logger.warning("rule %s produced an invalid assertion: %s", rule.rule_id, e)
            return None

        return Inference(assertion=assertion, rule_id=rule.rule_id, premise_ids=premise_ids)


def infer_and_stage(
    ontology: Any, review_queue: Any, ctx: AuthContext, *, job_id: str = "reasoner"
) -> ReasonerReport:
    """Run the pack's rules over a tenant's live facts and stage what they conclude.

    One definition, called by the `/reason` endpoint and by the ingest pipeline. Two copies would
    drift, and the drift would run in the dangerous direction: an endpoint that stages conclusions
    and an ingest path that quietly does not, so whether a conflict exists would depend on how the
    document happened to arrive.

    **Staged, never written live.** An inference passes the same review gate as an extraction,
    because a conflict flag nobody signed off is still one somebody would act on.

    Premises are the *live* assertions rather than everything visible: a conclusion may only rest
    on facts somebody stands behind. Idempotent, because `assertion_id` is content-addressed -- a
    second pass over unchanged facts converges instead of duplicating.
    """
    live = [r.assertion for r in review_queue.live_assertions(ctx)]
    report = Reasoner(ontology).run(ctx, live)
    if report.inferences:
        review_queue.stage(ctx, [i.assertion for i in report.inferences], job_id=job_id)
    return report
