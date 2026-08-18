"""Tier 2 and 3: reading assertions to answer a question.

Every read goes through `scope.edge_scope`, so an answer can only ever rest on facts
the caller is allowed to see and that clear the trust floor. That is what makes the
`assertions_used` list on a `Resolution` a real audit trail rather than a label: each
id in it resolves to a document span or a proof tree.

Term matching here is lexical — entity ids and predicates, not embeddings. That is
deliberate for tier 2: "which matters involve Acme Corporation" should find
`party:acme-corporation` by name, and doing it lexically means the result is
reproducible and explainable. Tier 3 is where semantics come in.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from src.documents.review import ReviewQueue
from src.graph.assertions import SIGNED_OFF_STATES
from src.graph.scope import AuthContext, TrustFilter
from src.query.blocks import BlockCheckUnavailable

logger = logging.getLogger(__name__)

#: Words that carry no signal for entity matching. Kept small on purpose — an
#: aggressive stop list drops real search terms ("the Crown", "In re Smith").
_NOISE = frozenset(
    {
        "a", "an", "and", "any", "are", "our", "all", "about", "by", "did", "do",
        "does", "for", "from", "give", "has", "have", "how", "in", "involve",
        "involved", "involves", "is", "it", "list", "many", "me", "much", "of",
        "on", "or", "show", "that", "their", "the", "to", "us", "was", "we",
        "what", "when", "where", "which", "who", "whom", "with", "you",
    }
)

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'&.-]*")

#: Passage fields that name a graph node, and the prefix that node id carries.
#:
#: Ids the passage already holds, never a name read out of its text. Nothing may run a model
#: between retrieval and traversal or the set of facts an answer consulted stops being
#: reproducible for the same question. `matter_id` is safe as a join key for the reason
#: `blocks.SEED_KEYS` gives: it is on every chunk and every assertion, so a mis-join cannot
#: happen -- unlike matching a party by name.
_PASSAGE_SEED_FIELDS = (("document_id", "document"), ("matter_id", "matter"))


def passage_seeds(passages: Any) -> list[str]:
    """Graph nodes the retrieved passages name, in both id forms.

    Both forms because a passage carries a bare id (`doc-...`, `M-1`) while the node is
    `document:doc-...` per `DocumentMeta.entity_id`. Seeding only the bare id matched nothing on
    the first frontier and every hybrid answer came back with no related facts -- a silent empty
    rather than an error, which is why it survived. A seed no assertion carries costs one set
    lookup; being wrong the other way costs the graph half of the answer.
    """
    if not isinstance(passages, list):
        return []
    seeds: list[str] = []
    for passage in passages:
        if not isinstance(passage, dict):
            continue
        for key, prefix in _PASSAGE_SEED_FIELDS:
            value = passage.get(key)
            if isinstance(value, str) and value:
                seeds.append(value)
                seeds.append(f"{prefix}:{value}")
    return seeds


def terms_of(question: str) -> list[str]:
    """Content words, lowercased, order preserved."""
    return [
        w.lower()
        for w in _WORD.findall(question)
        if w.lower() not in _NOISE and len(w) > 1
    ]


def _entity_label(entity_id: str) -> str:
    """`party:acme-corporation` -> `acme corporation`.

    Matching happens on the label rather than the raw id so a question phrased in
    ordinary words can reach an id built by slugging.
    """
    _, _, rest = entity_id.partition(":")
    return (rest or entity_id).replace("-", " ").replace("_", " ").lower()


def _routes(assertion: Any) -> frozenset[str]:
    """Frontier ids that reach this assertion: its two endpoints, and its source document.

    The source document is a route because the graph's spine is matter -> document -> fact and a
    model-extracted fact has the document as *neither* endpoint. Without it a document seed
    reached such a fact only through an incidental `MENTIONS` -- which sits exactly on the trust
    floor by design, so raising the floor severed the graph and tier 3 answered with no facts at
    all. Both id forms, because the node is `document:<id>` (`DocumentMeta.entity_id`) while
    `source_locator` holds the bare `doc-...`.

    No approval check here on purpose. `_readable` already admits only signed-off, in-scope,
    trusted-enough records, and a second copy of that policy is how the trust conditions came to
    disagree once already.
    """
    routes = {assertion.subject_id, assertion.object_id}
    document_id = getattr(assertion.source_locator, "document_id", None)
    if document_id:
        routes |= {document_id, f"document:{document_id}"}
    return frozenset(routes)


def _seed_set(seeds: list[str]) -> frozenset[str]:
    """Seeds plus their bare forms, so `document:d1` and `d1` both match one entity.

    Callers already pass both -- `Resolver._try_hybrid` learned to the hard way -- but a caller
    that passes only one must not get a veto that silently fails to match, which is the whole
    failure mode this method exists to close.
    """
    out: set[str] = set()
    for seed in seeds:
        if not isinstance(seed, str) or not seed:
            continue
        out.add(seed)
        _, sep, rest = seed.partition(":")
        if sep and rest:
            out.add(rest)
    return frozenset(out)


def _touches(entity_id: str, wanted: frozenset[str]) -> bool:
    if entity_id in wanted:
        return True
    _, sep, rest = entity_id.partition(":")
    return bool(sep and rest and rest in wanted)


def _block_reason(ontology: Any, predicate: str, other_id: str) -> str:
    """A sentence naming the predicate and the other end.

    The predicate's wording comes from the pack, not from here: `blocks_for` renders an empty
    reason as "blocked", which is a refusal with no explanation -- worse than no block, because it
    looks like a screen nobody can appeal.

    The other end is the raw entity id, not a prettified label. `_entity_label` turns
    `matter:m-2` into "m 2", which is not a thing anyone can look up.
    """
    pdef = getattr(ontology, "predicates", {}).get(predicate)
    label = getattr(pdef, "label", None) or predicate.replace("_", " ").lower()
    return f"{label[:1].upper()}{label[1:]} involving {other_id}."


@dataclass
class Hit:
    """One assertion that matched, with why it matched."""

    assertion_id: str
    subject_id: str
    predicate: str
    object_id: str
    epistemic_class: str
    confidence: float
    matter_id: str | None
    matched_on: list[str]
    source: dict[str, Any]

    hops: int | None = None
    """How far from a cited passage a walked edge was reached. None for a term match, where
    `matched_on` is the explanation instead."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "assertion_id": self.assertion_id,
            "subject_id": self.subject_id,
            "predicate": self.predicate,
            "object_id": self.object_id,
            "epistemic_class": self.epistemic_class,
            "confidence": self.confidence,
            "matter_id": self.matter_id,
            "matched_on": self.matched_on,
            "hops": self.hops,
            "source": self.source,
        }


class GraphReader:
    """Reads assertions for the resolver.

    Backed by `ReviewQueue` for now, which is where assertions currently live. When
    the graph writer lands this becomes `GraphClient.read_scoped` — the interface the
    resolver depends on does not change, which is the point of keeping it narrow.
    """

    def __init__(self, review_queue: ReviewQueue, *, ontology: Any | None = None) -> None:
        self._queue = review_queue
        self._ontology = ontology
        """Supplies the governing/descriptive split `expand()` ranks on. Optional: without it
        every predicate ranks alike and ordering falls back to confidence, which is the old
        behaviour minus the truncation bug rather than a new failure."""

    def _is_governing(self, predicate: str) -> bool:
        return self._ontology is not None and self._ontology.is_governing(predicate)

    def blocking_facts(self, ctx: AuthContext, seeds: list[str]) -> list[dict[str, Any]]:
        """Facts that FORBID an answer about these seeds, as opposed to `expand()`'s that inform one.

        Which predicates veto is the pack's call, not this module's: `blocks:` on a governing
        predicate names the tainted end of the edge. Hardcoding `POTENTIAL_CONFLICT` here would put
        a legal-specific list in domain-agnostic code and, worse, put it outside the closed
        vocabulary — so a pack could declare a veto that nothing enforced.

        One hop, not a walk. A blocking fact has to *touch* the evidence to veto it; two hops out
        it is a fact about something adjacent, and blocking on that withholds most of a
        well-connected tenant. This is the opposite mistake from `expand()`, which wants reach.

        Never truncated, and no `limit` parameter to add one. `expand()` caps its output because a
        fact it drops is a fact the answer lacks; a veto it dropped would be an answer that looks
        cleared. Blocking predicates are rule conclusions over signed-off premises, so the set is
        small by construction.

        **No confidence floor, and no parameter to set one.** Every other gate still applies —
        tenant, matter wall, review state, `is_current` — but a block is not evidence to be
        weighed against a threshold, it is a refusal. Filtering refusals by confidence means the
        *least* certain conflict is the one silently ignored, and a check that returns nothing
        because the veto fell 0.03 under a floor is indistinguishable from a clean check. It
        matters more now that a conclusion decays per hop: a conflict found four steps out is
        exactly the one nobody would spot by hand. The confidence still travels on the block, so
        a reviewer sees how firm it is rather than having it decided for them.

        **Ordered by reach, furthest first**, which is the opposite of ordering by confidence and
        deliberately so. A conflict the firm found by combining five facts across three matters is
        the one a partner would never have spotted unaided; a direct one they very likely already
        know. Alphabetical order — what this returned before — put `matter:a-1` above a five-fact
        finding about `party:zenith` for no reason a reader could defend.
        """
        if self._ontology is None:
            # Fails rather than returning no blocks. Without a pack the blocking vocabulary is
            # unknowable, and "I found no vetoes" would be a claim this reader cannot make.
            raise BlockCheckUnavailable(
                "no ontology pack is wired into the graph reader, so which predicates veto an "
                "answer is unknown"
            )

        blocking = self._ontology.blocking_predicates
        if not blocking:
            return []

        wanted = _seed_set(seeds)
        if not wanted:
            return []

        out: dict[tuple[str, str, str], dict[str, Any]] = {}
        for record in self._readable(ctx, 0.0):
            a = record.assertion
            if a.predicate not in blocking:
                continue
            if not (
                _touches(a.subject_id, wanted)
                or _touches(a.object_id, wanted)
                or (a.matter_id is not None and a.matter_id in wanted)
            ):
                continue
            for end in self._ontology.blocked_endpoints(a.predicate):
                tainted = a.subject_id if end == "subject" else a.object_id
                other = a.object_id if end == "subject" else a.subject_id
                out[(tainted, a.predicate, other)] = {
                    "subject_id": tainted,
                    "reason": _block_reason(self._ontology, a.predicate, other),
                    "rule": a.rule_id or a.predicate,
                    "matter_id": a.matter_id,
                    # Reported rather than filtered on. The refusal stands either way; how firm
                    # it is is a reviewer's judgement, not this module's.
                    "confidence": a.confidence,
                    # How many signed-off facts had to be combined to see this at all. 0 for a
                    # fact somebody stated outright. This is the honest measure of "would anyone
                    # have found this unaided" -- and it is deliberately not confidence, which
                    # moves the other way: reach makes a finding less certain and more valuable.
                    "premise_count": len(a.premises),
                }

        # Furthest reach first, then the firmer of two equals, then the id so a tie is broken the
        # same way twice -- two runs of one conflict check listing refusals in a different order
        # is indistinguishable from the graph having changed.
        return [
            out[key]
            for key in sorted(
                out, key=lambda k: (-out[k]["premise_count"], -out[k]["confidence"], k)
            )
        ]

    def unreviewed_blocks(self, ctx: AuthContext, seeds: list[str]) -> list[dict[str, Any]]:
        """Blocking facts about these seeds that nobody has reviewed yet.

        Not vetoes, and deliberately a separate method so they cannot become vetoes by accident.
        A PENDING conflict is a *proposal*: refusing on it would let an unreviewed derivation
        withhold evidence, which is the thing `SIGNED_OFF` exists to prevent.

        But "nothing refused" over a graph holding a conflict awaiting review is a true sentence
        that reads as a false one. A reader is entitled to know both that the check came back
        clean and that there is a conflict about this party in the queue — those are different
        facts, and only the first is reassuring on its own.

        Same tenant and matter walls as every other read; only the review-state gate differs.
        """
        if self._ontology is None:
            raise BlockCheckUnavailable(
                "no ontology pack is wired into the graph reader, so which predicates veto an "
                "answer is unknown"
            )
        blocking = self._ontology.blocking_predicates
        wanted = _seed_set(seeds)
        if not blocking or not wanted:
            return []

        out: dict[tuple[str, str], dict[str, Any]] = {}
        for record in self._queue.visible(ctx):
            a = record.assertion
            if not record.is_current or a.predicate not in blocking:
                continue
            if a.review_state in SIGNED_OFF_STATES:
                # Already a veto, reported by `blocking_facts`. Naming it here as well would
                # double every refusal and make this advisory look like a second, weaker wall.
                continue
            if not (
                _touches(a.subject_id, wanted)
                or _touches(a.object_id, wanted)
                or (a.matter_id is not None and a.matter_id in wanted)
            ):
                continue
            out[(a.subject_id, a.object_id)] = {
                "subject_id": a.subject_id,
                "object_id": a.object_id,
                "predicate": a.predicate,
                "rule": a.rule_id or a.predicate,
                "matter_id": a.matter_id,
                "assertion_id": a.assertion_id,
                "confidence": a.confidence,
            }
        return [out[key] for key in sorted(out)]

    def _readable(self, ctx: AuthContext, min_confidence: float) -> list[Any]:
        """Current, in-scope, trusted-enough assertions.

        `visible()` already applies the tenant and matter walls. The trust half is
        `TrustFilter`, the same object `scope.edge_scope` renders into Cypher — it used to be
        a second hand-written copy of those conditions, which is how the two came to disagree
        about which facts an answer may rest on.
        """
        trust = TrustFilter.for_context(ctx, min_confidence=min_confidence)
        return [
            record
            for record in self._queue.visible(ctx)
            if record.is_current and trust.matches(record.assertion)
        ]

    def search(
        self,
        ctx: AuthContext,
        question: str,
        *,
        min_confidence: float = 0.8,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        """Assertions whose entities or predicate match the question.

        Ranked by how many query terms matched, then by confidence. Returning the
        matched terms is what lets the UI say *why* a fact was included.
        """
        terms = terms_of(question)
        if not terms:
            return []

        scored: list[tuple[int, float, Hit]] = []
        for record in self._readable(ctx, min_confidence):
            a = record.assertion
            haystack = " ".join(
                (
                    _entity_label(a.subject_id),
                    _entity_label(a.object_id),
                    a.predicate.replace("_", " ").lower(),
                )
            )
            matched = [t for t in terms if t in haystack]
            if not matched:
                continue
            scored.append(
                (
                    len(matched),
                    a.confidence,
                    Hit(
                        assertion_id=a.assertion_id,
                        subject_id=a.subject_id,
                        predicate=a.predicate,
                        object_id=a.object_id,
                        epistemic_class=a.epistemic_class.value,
                        confidence=a.confidence,
                        matter_id=a.matter_id,
                        matched_on=matched,
                        source=a.source_locator.to_dict(),
                    ),
                )
            )

        scored.sort(key=lambda row: (-row[0], -row[1]))
        return [hit.to_dict() for _, _, hit in scored[:limit]]

    def expand(
        self,
        ctx: AuthContext,
        seed_ids: list[str],
        *,
        depth: int = 2,
        min_confidence: float = 0.8,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Walk out from seed entities or documents along trusted edges.

        Used by tier 3: vector search supplies the passages, this supplies the
        verified relationships around them. Breadth-first and depth-capped, because
        an unbounded walk on a well-connected graph returns the whole tenant.

        A fact is reachable from its endpoints *and* from the document it was extracted from --
        see `_routes`. That is the spine the design intends, matter -> document -> fact, and it
        was missing: the only bridge from a document to a model-extracted fact was an incidental
        `MENTIONS`, so `REPRESENTS` at 0.95 was unreachable while a 0.80 presence claim was not.

        Ranked before truncating, which it previously was not: the walk cut off mid-hop at
        `limit`, so which facts survived came down to the store's insertion order. With the
        catalog's DECLARED edges sitting at 1.00 the cap filled with `HAS_COLUMN` schema noise
        and an approved `ADVERSE_TO` never made the list.

        Governing predicates lead, then confidence, then hop distance. Confidence alone is not
        enough and the reason is worth stating: a Glue declaration genuinely *is* 1.00 — the
        catalog said so — so no honest confidence number puts `ADVERSE_TO` ahead of it, and
        lowering it would be false modesty about a system of record. What separates them is
        whether the predicate carries consequence, which is the ontology's call. Same ranking
        key as the graph overview in `routes_catalog._graph_overview`, for the same reason.
        """
        records = self._readable(ctx, min_confidence)
        frontier = {s for s in seed_ids if s}
        seen_nodes: set[str] = set()
        found: dict[str, tuple[int, Any]] = {}

        for hop in range(1, max(1, depth) + 1):
            next_frontier: set[str] = set()
            for record in records:
                a = record.assertion
                if not _routes(a) & frontier:
                    continue
                # Endpoints only. The source document is a way *in* to a fact, not a node the walk
                # then leaves from: pushing it on would make every fact sharing a document one hop
                # from every other, which on a 300-page bundle is the unbounded walk the depth cap
                # exists to prevent.
                next_frontier.update({a.subject_id, a.object_id})
                # First seen wins. A later pass re-matches an edge whose endpoint is still in the
                # frontier, and overwriting would relabel a direct edge as two hops out -- the
                # shortest distance is the one that means anything.
                if a.assertion_id not in found:
                    found[a.assertion_id] = (hop, a)
            seen_nodes |= frontier
            frontier = next_frontier - seen_nodes
            if not frontier:
                break

        # `assertion_id` last so a tie is resolved the same way twice. Two runs of one question
        # returning different facts is indistinguishable from the graph having changed.
        ranked = sorted(
            found.values(),
            key=lambda row: (
                not self._is_governing(row[1].predicate),
                -row[1].confidence,
                row[0],
                row[1].assertion_id,
            ),
        )
        return [
            Hit(
                assertion_id=a.assertion_id,
                subject_id=a.subject_id,
                predicate=a.predicate,
                object_id=a.object_id,
                epistemic_class=a.epistemic_class.value,
                confidence=a.confidence,
                matter_id=a.matter_id,
                # A walked edge did not match a word, so `matched_on` stays empty and `hops`
                # carries the reason instead: how far from a cited passage it was reached.
                # Without it a reader cannot tell a fact stated in the document from one two
                # steps away, which is the difference between quoting and inferring.
                matched_on=[],
                hops=hop,
                source=a.source_locator.to_dict(),
            ).to_dict()
            for hop, a in ranked[:limit]
        ]
