"""What the graph refuses to let through, and the one place that decides it.

Shared by `resolver` and `planner` on purpose. Two implementations of a veto diverge, and the
divergence is silent: an answer that passed the check on one endpoint and would have been
screened on the other looks exactly like a clean answer.

Never asks a model. An ethical screen is a recorded decision on `AuthContext`; a rule block is
an inference, but one the ontology restricts to DECLARED and EXTRACTED_DET premises. If a model
decided whether the patient was allergic, a hallucination would prescribe penicillin.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.graph.scope import AuthContext

logger = logging.getLogger(__name__)

#: Ids on a row that a block can match. `matter_id` is the join key rather than a fuzzy name
#: match: it already exists on every chunk and every assertion, so a mis-join is not possible.
SEED_KEYS = ("document_id", "subject_id", "object_id", "matter_id")


class BlockCheckUnavailable(RuntimeError):
    """The rule-block half of the wall could not be evaluated.

    Raised rather than returning no blocks, because those two are indistinguishable to a caller
    and the wrong one of them is a governance control reporting "nothing refused" when it was
    structurally incapable of refusing. Whoever catches this owes the caller a warning.
    """


@dataclass
class Block:
    """Something the graph refuses to let through, and why.

    A block is a recorded fact, not a judgement: an ethical screen, or a rule that fired on
    DECLARED or EXTRACTED_DET premises. It names its reason because a silent block is the
    failure `scope.py` exists to prevent, where a clean-looking answer is clean only because
    the inconvenient part was invisible.
    """

    subject: str
    reason: str
    rule: str = ""
    matter_id: str | None = None

    contact: str | None = None
    """Who to ask, for a screen. Disclosed at the same level as `list_matters`: within one firm
    a screen names the matter and a contact, because a conflict check that comes back clean only
    because the matching matter was invisible is the harm the wall exists to prevent. None for a
    rule block, which is not somebody's decision to explain."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "reason": self.reason,
            "rule": self.rule,
            "matter_id": self.matter_id,
            "contact": self.contact,
        }


@dataclass(frozen=True)
class Screen:
    """The blocks in force for one question, and the test a row must pass to survive them."""

    blocks: list[Block] = field(default_factory=list)

    seeds: tuple[str, ...] = ()
    """What the wall was asked about. Reported because "nothing refused" over zero seeds and over
    forty are different facts, and only the second is reassuring."""

    degraded: str = ""
    """Why the rule-block half did not run, or empty when it did.

    A screened-only result and a fully evaluated one are the same object otherwise, so without
    this the caller cannot tell a clean wall from a wall that failed open."""

    def __bool__(self) -> bool:
        return bool(self.blocks)

    @property
    def cleared(self) -> int:
        """Seeds no block named. Zero when degraded: nothing was cleared, it was skipped."""
        if self.degraded:
            return 0
        subjects = self.subjects
        return sum(1 for s in set(self.seeds) if s not in subjects)

    def trace(self, *, items_withheld: int) -> dict[str, Any]:
        """The counters the trace UI reads. `items_withheld` is the caller's, not ours: only the
        caller knows how many rows it actually dropped."""
        return {
            "seeds_considered": len(set(self.seeds)),
            "subjects_cleared": self.cleared,
            "items_withheld": items_withheld,
            "degraded": self.degraded or None,
            "blocks": self.to_dict(),
        }

    @property
    def subjects(self) -> set[str]:
        return {b.subject for b in self.blocks if b.subject}

    @property
    def matters(self) -> set[str | None]:
        return {b.matter_id for b in self.blocks if b.matter_id}

    def allows(self, row: Any) -> bool:
        """False for a row a block names. Non-dict rows pass: there is no id to match on."""
        if not isinstance(row, dict):
            return True
        subjects, matters = self.subjects, self.matters
        return not (
            row.get("subject_id") in subjects
            # `object_id` too. It is in `SEED_KEYS`, so a blocked entity reaching the wall as the
            # object of an edge produced a block and then survived it: an assertion "d1 MENTIONS
            # party:calder" was named a conflict and handed to the synthesiser anyway. The wall has
            # to test the same ends it seeds from, or it labels rather than withholds.
            or row.get("object_id") in subjects
            or row.get("document_id") in subjects
            or row.get("matter_id") in matters
        )

    def keep(self, rows: list[Any]) -> list[Any]:
        if not self.blocks:
            return rows
        return [row for row in rows if self.allows(row)]

    def to_dict(self) -> list[dict[str, Any]]:
        return [b.to_dict() for b in self.blocks]


def seeds_from(rows: Any) -> list[str]:
    """Entity, document and matter ids a result touched, for grounding."""
    if not isinstance(rows, list):
        return []
    seeds: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in SEED_KEYS:
            value = row.get(key)
            if isinstance(value, str) and value:
                seeds.append(value)
    return seeds


def blocks_for(
    ctx: AuthContext,
    *,
    graph_reader: Any | None = None,
    seeds: list[str] | None = None,
) -> Screen:
    """The vetoes in force for this caller and these seeds.

    Two sources, both deterministic. An ethical screen is a recorded decision on `AuthContext`
    and applies whether or not a graph reader is wired; a rule block needs the graph and the
    seeds a result touched.

    Degrades loudly, and the choice is deliberate. Failing closed on a graph error would refuse
    every answer for as long as the graph was unwell, which is its own outage; failing open
    silently is how this half of the wall came to be missing for the life of the feature. So it
    fails open and *says so* on the `Screen`, and every caller turns that into a warning the
    reader sees rather than a debug line nobody reads.

    Takes no confidence floor. The governance floor decides what may *inform* an answer; a veto
    is not informing one. See `GraphReader.blocking_facts`.
    """
    blocks: list[Block] = []
    degraded = ""

    for matter_id in sorted(ctx.matter_denylist):
        blocks.append(
            Block(
                subject=matter_id,
                reason=ctx.screen_reasons.get(matter_id) or "You are screened from this matter.",
                rule="ethical_screen",
                matter_id=matter_id,
                contact=ctx.screen_contacts.get(matter_id),
            )
        )

    if graph_reader is not None and seeds:
        try:
            found_rows = graph_reader.blocking_facts(ctx, seeds)
        except BlockCheckUnavailable as e:
            degraded = str(e)
            logger.warning("rule blocks not evaluated: %s", e)
        except Exception as e:  # noqa: BLE001
            # No `AttributeError` branch. A reader without `blocking_facts` is not a legacy
            # deployment to tolerate -- it is the state this whole path was in, and swallowing it
            # is what made a missing veto indistinguishable from a clean one.
            degraded = f"the block check failed: {e}"
            logger.warning("rule blocks not evaluated: %s", e)
        else:
            for found in found_rows:
                blocks.append(
                    Block(
                        subject=str(found.get("subject_id", "")),
                        reason=str(found.get("reason") or found.get("predicate") or "blocked"),
                        rule=str(found.get("rule") or found.get("predicate") or ""),
                        matter_id=found.get("matter_id"),
                    )
                )

    return Screen(blocks=blocks, seeds=tuple(seeds or ()), degraded=degraded)


#: Shown when the rule-block half failed. Names the half that still ran, because "the wall is
#: broken" and "one of the wall's two sources is broken" call for different reactions.
DEGRADED_WARNING = (
    "The ethical screens on your account were applied, but the graph could not be checked for "
    "conflicts or other rule-based blocks, so this answer may include evidence that would "
    "normally be withheld. Treat it as incomplete for conflict-checking purposes."
)
