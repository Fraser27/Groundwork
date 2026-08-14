"""Cascading retraction — invariant 5, the half `build_assertion` cannot enforce.

`build_assertion` guarantees that an INFERRED assertion names its premises. This
module guarantees the converse: when a premise stops being true, every conclusion
resting on it stops too, transitively.

Why it has to live here: the contract validates one candidate assertion in isolation,
and this needs to walk the premise DAG. `assertions.py` says as much in its docstring.

Why it matters more than it looks: the failure is silent. Retract the extraction that
said Counsel represents Party, leave the conflict flag standing, and the graph now
asserts a conflict whose stated basis has been withdrawn. Every "why do you believe
this?" answer for that flag cites a retracted premise — which is precisely the
question the whole system exists to answer well.

**Append-only. Nothing is ever DELETEd.** Retraction sets `superseded_at`, so a
bitemporal read with `as_of` before that timestamp still shows the fact, and still
shows the conflict flag it supported. "What did the file show when we advised?" has to
remain answerable after a correction — that is the whole point of transaction time.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.documents.review import AssertionRecord, Lifecycle, ReviewQueue
from src.graph.assertions import EpistemicClass
from src.graph.scope import AuthContext

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RetractionError(ValueError):
    pass


@dataclass(frozen=True)
class Retraction:
    """What one retraction did, for the audit log.

    `cascaded` is separate from `root` because they are different events to a
    reviewer: one person decided one thing, and N conclusions fell as a consequence.
    Collapsing them would misreport a single decision as N decisions.
    """

    root: str
    cascaded: tuple[str, ...]
    reason: str
    retracted_by: str
    retracted_at: str
    orphaned_rules: tuple[str, ...] = field(default=())

    @property
    def total(self) -> int:
        return 1 + len(self.cascaded)


def dependency_closure(
    queue: ReviewQueue, ctx: AuthContext, assertion_id: str
) -> list[AssertionRecord]:
    """Every current INFERRED assertion transitively resting on `assertion_id`.

    Breadth-first over the reverse premise index. Two details that are not optional:

    - `seen` guards against cycles. The premise graph should be a DAG, but a buggy
      rule that infers a fact from its own consequence would otherwise loop forever,
      and this code path runs during a correction — the worst time to hang.
    - Already-superseded assertions are skipped rather than re-stamped. Retracting
      twice must not rewrite the first retraction's timestamp, or the bitemporal
      history stops reflecting when we actually stopped believing it.
    """
    closure: list[AssertionRecord] = []
    seen = {assertion_id}
    frontier: deque[str] = deque([assertion_id])

    while frontier:
        current = frontier.popleft()
        for dependent in queue.store.dependents_of(ctx.tenant_id, current):
            if dependent.assertion_id in seen:
                continue
            seen.add(dependent.assertion_id)
            if not dependent.is_current:
                continue
            closure.append(dependent)
            frontier.append(dependent.assertion_id)
    return closure


def retract(
    queue: ReviewQueue,
    ctx: AuthContext,
    assertion_id: str,
    *,
    reason: str,
    dry_run: bool = False,
) -> Retraction:
    """Retract an assertion and everything inferred from it.

    `dry_run` returns the same closure without writing. The UI is expected to use it:
    "this will also withdraw 4 conclusions, including a conflict flag" is information a
    reviewer needs *before* deciding, not after.
    """
    if not reason:
        raise RetractionError("a retraction must carry a reason — this is the audit record")

    record = queue.fetch(ctx, assertion_id)
    if not record.is_current:
        raise RetractionError(
            f"{assertion_id} was already retracted at {record.assertion.superseded_at}"
        )

    cascade = dependency_closure(queue, ctx, assertion_id)
    at = _now()

    if not dry_run:
        _supersede(queue, record, reason=reason, by=ctx.user_id, at=at)
        for dependent in cascade:
            _supersede(
                queue,
                dependent,
                reason=f"premise {assertion_id} retracted: {reason}",
                by=ctx.user_id,
                at=at,
            )
        logger.info(
            "%s retracted %s, cascading to %d inferences", ctx.user_id, assertion_id, len(cascade)
        )

    return Retraction(
        root=assertion_id,
        cascaded=tuple(d.assertion_id for d in cascade),
        reason=reason,
        retracted_by=ctx.user_id,
        retracted_at=at,
        orphaned_rules=tuple(
            sorted({d.assertion.rule_id for d in cascade if d.assertion.rule_id})
        ),
    )


def retract_document(
    queue: ReviewQueue, ctx: AuthContext, document_id: str, *, reason: str
) -> list[Retraction]:
    """Retract every assertion sourced from a document, plus their cascades.

    The case this exists for: a re-parse with a fixed extractor, or a document
    withdrawn from a matter. Roots are retracted one at a time and the closure is
    recomputed each time, so an assertion caught by an earlier cascade is skipped
    rather than double-stamped.
    """
    roots = [
        r
        for r in queue.visible(ctx)
        if r.assertion.source_locator.document_id == document_id
        and r.is_current
        # An INFERRED assertion cited to a document is a consequence, not a root; it
        # falls out of the cascade of whatever extraction it rests on.
        and r.assertion.epistemic_class is not EpistemicClass.INFERRED
    ]
    results: list[Retraction] = []
    for root in roots:
        if not root.is_current:
            continue
        results.append(retract(queue, ctx, root.assertion_id, reason=reason))
    return results


def supersede(
    queue: ReviewQueue,
    ctx: AuthContext,
    *,
    old_assertion_id: str,
    new_record: AssertionRecord,
    reason: str,
) -> Retraction:
    """Replace an assertion with a corrected one, cascading the retraction.

    The re-extraction path: `regex:citation@v3` produced this, `@v4` produces
    something better, and the two must not coexist. The old one's dependents cascade
    because their premise is genuinely gone — the rules will re-fire against the new
    assertion, which is right, since a rule's conclusion should rest on the premise
    that is actually current.
    """
    if new_record.assertion.tenant_id != ctx.tenant_id:
        raise RetractionError("replacement assertion belongs to another tenant")
    result = retract(queue, ctx, old_assertion_id, reason=reason)
    queue.store.put(new_record)
    return result


def _supersede(
    queue: ReviewQueue, record: AssertionRecord, *, reason: str, by: str, at: str
) -> None:
    record.assertion.superseded_at = at
    record.retracted_reason = reason
    record.retracted_by = by
    # Lifecycle stays LIVE if it was live. `superseded_at` is what removes it from
    # reads (see edge_scope), and rewriting lifecycle would lose the fact that this
    # assertion was once believed and acted upon.
    if record.lifecycle is Lifecycle.STAGED:
        record.lifecycle = Lifecycle.DISCARDED
    queue.store.put(record)


def explain_retraction(
    queue: ReviewQueue, ctx: AuthContext, assertion_id: str
) -> dict[str, object]:
    """Why an assertion is no longer believed, for the audit UI."""
    record = queue.fetch(ctx, assertion_id)
    return {
        "assertion_id": assertion_id,
        "is_current": record.is_current,
        "superseded_at": record.assertion.superseded_at,
        "retracted_by": record.retracted_by,
        "reason": record.retracted_reason,
        "premises": list(record.assertion.premises),
        "rule_id": record.assertion.rule_id,
    }


def would_cascade(queue: ReviewQueue, ctx: AuthContext, assertion_ids: Iterable[str]) -> set[str]:
    """Union of what retracting each of `assertion_ids` would invalidate."""
    affected: set[str] = set()
    for aid in assertion_ids:
        affected.add(aid)
        affected.update(d.assertion_id for d in dependency_closure(queue, ctx, aid))
    return affected
