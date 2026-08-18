"""Merging two entity ids that turned out to name one thing.

The third kind of correction, and it is genuinely distinct from the two that already exist:

- a **wipe** leaves inferences standing, because the belief was honestly held at the time;
- a **retraction** cascades, because the fact was wrong;
- a **merge** cascades too, but for the opposite reason — the fact was *right*, and stated about
  a fork of the node it belongs to.

Leaving a conclusion standing after a merge would be the worst of the three outcomes: there would
be two conflict edges for one conflict, one of them citing a premise that is now closed, and every
"why do you believe this" answer for it would resolve to a superseded assertion.

**No id is ever rewritten.** `Assertion._compute_id` hashes `subject_id` and `object_id`, so
changing an endpoint necessarily changes the assertion's identity. That makes a merge N supersedes
rather than an UPDATE — which is not a workaround but the honest shape: each original has its own
provenance, and each closure is its own audit event.

**Never automatic, at any confidence.** `entity_blocking_keys` finds *candidates*; a person
decides. The same key that catches a spelling variant also catches a genuine sibling company, and
merging those would convert an affiliate conflict into a false direct one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.documents.retract import Retraction, would_cascade
from src.documents.retract import supersede as retract_supersede
from src.documents.review import AssertionRecord, Lifecycle, ReviewQueue
from src.graph.scope import AuthContext

logger = logging.getLogger(__name__)


class MergeError(ValueError):
    pass


@dataclass(frozen=True)
class Merge:
    """What a merge did, or would do when `dry_run`.

    `affected` and `cascaded` are separate for the reason `Retraction` splits them: one person
    decided one thing, and N conclusions fell as a consequence. Reporting them together would
    misstate a single decision as many.
    """

    losing_id: str
    winning_id: str
    affected: tuple[str, ...]
    """Current assertions naming `losing_id` at either end."""

    cascaded: tuple[str, ...] = field(default=())
    """Conclusions that fall because a premise among `affected` closes, and are not themselves in
    `affected`.

    Often empty even when conclusions do fall, and that is not a bug: a `POTENTIAL_CONFLICT` about
    the losing id names it directly, so it is restated as one of `affected` rather than cascading.
    A conclusion appears here only when it rests on an affected premise *without* naming the
    merged entity — a stale-authority flag on a document that mentioned it, say."""

    rewritten: tuple[str, ...] = field(default=())
    """Ids of the reviewer's replacement assertions. Empty for a dry run."""

    dry_run: bool = False

    @property
    def total(self) -> int:
        return len(self.affected) + len(self.cascaded)


def plan_merge(
    queue: ReviewQueue, ctx: AuthContext, *, losing_id: str, winning_id: str
) -> Merge:
    """What merging these two would touch, writing nothing.

    Not optional in the UI. "This will also withdraw a conflict flag" is something a reviewer
    needs *before* deciding, which is the argument `retract.retract` already makes for its own
    `dry_run`.
    """
    affected = _affected(queue, ctx, losing_id)
    cascaded = would_cascade(queue, ctx, [r.assertion_id for r in affected]) - {
        r.assertion_id for r in affected
    }
    return Merge(
        losing_id=losing_id,
        winning_id=winning_id,
        affected=tuple(sorted(r.assertion_id for r in affected)),
        cascaded=tuple(sorted(cascaded)),
        dry_run=True,
    )


def merge_entities(
    queue: ReviewQueue,
    ctx: AuthContext,
    *,
    losing_id: str,
    winning_id: str,
    reason: str,
    allowed_predicates: frozenset[str] | None = None,
    canonical_entity_id: object = None,
) -> Merge:
    """Restate every claim about `losing_id` as a claim about `winning_id`.

    Refuses rather than guesses in four cases, each of which would otherwise leave the graph
    worse than the fork did:

    - **No reason.** Same rule as a correction: the reason is the record of why.
    - **The same id twice.** Nothing to merge, and it would close assertions to replace them
      with identical ones.
    - **Different kinds.** `party:acme` and `court:acme` are unrelated; merging across kinds
      would move a fact onto a node of the wrong type.
    - **A losing or winning id not in normal form.** A merge that minted a fresh fork would be
      self-defeating.
    """
    if not reason:
        raise MergeError("a merge must carry a reason: it is the record of why")
    if losing_id == winning_id:
        raise MergeError("the two ids are the same, so there is nothing to merge")
    _refuse_cross_kind(losing_id, winning_id)
    if canonical_entity_id is not None:
        for entity_id in (losing_id, winning_id):
            canonical = canonical_entity_id(entity_id)  # type: ignore[operator]
            if canonical != entity_id:
                raise MergeError(
                    f"{entity_id!r} is not in canonical form (would be {canonical!r}); "
                    "merging into a fork would defeat the point"
                )

    affected = _affected(queue, ctx, losing_id)
    if not affected:
        raise MergeError(f"no current assertion names {losing_id!r}")

    cascaded: set[str] = set()
    rewritten: list[str] = []
    for record in affected:
        if not record.is_current:
            # An earlier cascade in this same loop reached it. Superseding twice would rewrite
            # the first retraction's timestamp and lose when we actually stopped believing it.
            continue
        original = record.assertion
        replacement = queue.reviewers_version(
            ctx,
            original,
            subject_id=winning_id if original.subject_id == losing_id else original.subject_id,
            predicate=original.predicate,
            object_id=winning_id if original.object_id == losing_id else original.object_id,
            allowed_predicates=allowed_predicates,
        )
        new_record = AssertionRecord(
            assertion=replacement, lifecycle=Lifecycle.LIVE, job_id=record.job_id
        )
        new_record.review_note = reason
        new_record.corrects = original.assertion_id

        # `retract.supersede` in this order deliberately: it retracts-with-cascade and *then*
        # stores the replacement. `ReviewQueue.supersede` closes the original itself, so calling
        # that first would leave `retract` refusing an already-retracted assertion.
        result: Retraction = retract_supersede(
            queue,
            ctx,
            old_assertion_id=original.assertion_id,
            new_record=new_record,
            reason=f"merged {losing_id} into {winning_id}: {reason}",
        )
        cascaded.update(result.cascaded)
        rewritten.append(replacement.assertion_id)

    logger.info(
        "merged %s into %s for %s: %d restated, %d cascaded",
        losing_id,
        winning_id,
        ctx.tenant_id,
        len(rewritten),
        len(cascaded),
    )
    return Merge(
        losing_id=losing_id,
        winning_id=winning_id,
        affected=tuple(sorted(r.assertion_id for r in affected)),
        cascaded=tuple(sorted(cascaded)),
        rewritten=tuple(rewritten),
    )


def _affected(queue: ReviewQueue, ctx: AuthContext, losing_id: str) -> list[AssertionRecord]:
    return [
        r
        for r in queue.visible(ctx)
        if r.is_current and losing_id in (r.assertion.subject_id, r.assertion.object_id)
    ]


def _refuse_cross_kind(losing_id: str, winning_id: str) -> None:
    # Case-insensitive, so `Party:X` and `party:y` are the *same* kind here and fall through to
    # the canonical-form check, which is the accurate complaint about `Party:X`. Comparing raw
    # would report "different kinds" for two parties, sending a reviewer looking for the wrong
    # problem.
    losing_kind, _, _ = losing_id.lower().partition(":")
    winning_kind, _, _ = winning_id.lower().partition(":")
    if losing_kind != winning_kind:
        raise MergeError(
            f"cannot merge {losing_id!r} into {winning_id!r}: they are different kinds "
            f"({losing_kind!r} and {winning_kind!r}), so they are not one entity"
        )
