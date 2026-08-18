"""The assertion contract — every fact in the graph is an Assertion.

Nothing writes an edge directly. Everything goes through `build_assertion`, which
is why the invariants below hold for the whole graph rather than just the code
paths someone remembered to check.

Storage is deliberately denormalised (see `to_edge_props` vs `to_node_props`):

    (:Matter)-[:ADVERSE_TO {assertion_id, epistemic_class, confidence, ...}]->(:Party)
         plus
    (:Assertion {assertion_id, method, source_locator, valid_from, ...})
         -[:PREMISE]->(:Assertion)

The edge carries only what a filtered traversal needs, so "walk edges I trust"
stays one hop. Full provenance and the premise DAG live on the :Assertion node and
are read when a user asks *why*. Pure reification would make every read path three
times the hops; edge-properties alone could not express a proof tree.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from src.constants import DEFAULT_MIN_CONFIDENCE


class EpistemicClass(str, Enum):
    """How we came to believe an assertion.

    This is the axis that makes the graph defensible: a parsed citation and an
    LLM's opinion are both edges, and without this they are indistinguishable at
    query time. Ordered loosely by how much weight a reviewer should give them.
    """

    DECLARED = "DECLARED"
    """Asserted by a system of record — a Glue catalog, a case management export.
    Not inferred, not extracted. If the source says it, it is true by definition."""

    EXTRACTED_DET = "EXTRACTED_DET"
    """A claim a deterministic check *confirmed*.

    Note what this does and does not mean. It is not "a regex found it" — there is no
    regex layer any more. A model proposes that some text appears on some page, and a
    string search against the chunk either confirms it or does not. The proposal is
    probabilistic; the confirmation is not, and the confirmation is what this class
    records.

    Only presence claims can qualify, because only presence is checkable. "This quote
    appears on page 7" is verifiable. "This holding undercuts that argument" is not."""

    EXTRACTED_MODEL = "EXTRACTED_MODEL"
    """A model's *interpretation*, which nothing can mechanically confirm.

    Everything semantic lands here — whether a citation was relied on or distinguished,
    whether one clause supersedes another, whether a party is adverse. This is the class
    the review queue exists for, and splitting it from verified presence is what keeps
    that queue small enough to actually be read."""

    INFERRED = "INFERRED"
    """Derived by a rule from other assertions. Carries `premises`, so it can
    always be explained as a proof tree."""

    PREDICTED = "PREDICTED"
    """A topological guess (link prediction). A research hint, never a fact.
    Excluded from retrieval unless explicitly requested."""


#: Classes safe to write straight into the live graph.
#:
#: DECLARED comes from a system of record. EXTRACTED_DET was mechanically confirmed —
#: a model said the text was there and a string search agreed. Neither rests on an
#: unverified judgement, so neither needs a human signature.
#:
#: The split matters for a practical reason: if every model output required review, a
#: 300-page bundle would produce hundreds of pending items, and a queue nobody can
#: clear gets rubber-stamped — which destroys the guarantee the queue exists to give.
AUTO_ASSERT_CLASSES = frozenset({EpistemicClass.DECLARED, EpistemicClass.EXTRACTED_DET})

#: Predicates a *verified presence* claim may use. Presence is all a quote-match can
#: establish, so anything implying significance — CITES implies reliance, ADVERSE_TO
#: implies a position — must stay EXTRACTED_MODEL and be reviewed.
#:
#: This is the fix for a real bug in an earlier design: the parser auto-asserted CITES,
#: so "the court declined to follow Brown" was recorded as reliance on Brown.
PRESENCE_PREDICATES = frozenset({"MENTIONS"})


#: Never returned by default retrieval — surfaced only through the explicit
#: "suggestions" endpoint, and always labelled as a guess.
SUGGESTION_ONLY_CLASSES = frozenset({EpistemicClass.PREDICTED})


#: Bottom of the band a fact must reach to inform an answer. Same number as the retrieval
#: floor, because it *is* the retrieval floor read from the write side.
ANSWERABLE_FLOOR = DEFAULT_MIN_CONFIDENCE

#: What a descriptive claim may ever be worth: the floor exactly, never above it.
#:
#: The ordering this fixes was inverted in production. `MENTIONS` means only "this string
#: appears in this document" and was written at 0.95, while an `ADVERSE_TO` a partner had
#: personally approved sat at 0.55 — so retrieval filled with the claim nobody needed and
#: dropped the one a conflict check reads. Which predicates are weak is the ontology's call
#: rather than a list here: descriptive predicates are exactly the ones a pack declines to
#: govern, which is the same distinction in the legal and healthcare packs both.
DESCRIPTIVE_CONFIDENCE = ANSWERABLE_FLOOR


#: A kind prefix, as it must appear in a stored entity id. Lowercase because `entity_kind_of`
#: lowercases before comparing against the vocabulary, so `Party:acme` passes the kind guard and
#: then MERGEs as a node distinct from `party:acme`.
_ENTITY_KIND_PREFIX = re.compile(r"^[a-z][a-z0-9_]*$")


def _reject_unnormalised(label: str, entity_id: str) -> None:
    """Refuse an id that is not in normal form, on the two rules that need no vocabulary.

    `Ontology.canonical_entity_id` is where an id gets *fixed*; this is the belt behind it, in
    the shape of `assertion_queries._TYPE_SAFE` — "unreachable and unchecked must not become the
    same thing". It cannot normalise, because that needs the pack's entity kinds and this module
    deliberately does not import the ontology (which is why `allowed_predicates` is passed in).
    But whitespace and a miscased prefix are purely syntactic, so they can be refused here, where
    nothing can bypass it.

    Deliberately narrow. It does **not** require a kind prefix at all, because that check needs
    the vocabulary and callers legitimately write ids like `Matter-4471`. It does **not** touch
    the part after the colon: `matter:NTL-2026-0114` is a case-management reference and
    `table:src-1:legal.matters` a Glue name, so case, dots and further colons are all meaningful
    there.
    """
    if not entity_id:
        return
    if entity_id != entity_id.strip() or any(c.isspace() for c in entity_id):
        raise AssertionError_(
            f"{label} {entity_id!r} contains whitespace; an id is a key, and "
            "'party: Calder Shipping AG' would MERGE as a node distinct from "
            "'party:calder-shipping-ag' while looking like the same company"
        )
    kind, sep, _ = entity_id.partition(":")
    if sep and not _ENTITY_KIND_PREFIX.match(kind):
        raise AssertionError_(
            f"{label} {entity_id!r} has a kind prefix that is not lowercase; the kind guard "
            "lowercases before checking the vocabulary, so this would pass it and then become a "
            "second node for one entity"
        )


def answerable_confidence(confidence: float, *, governing: bool = True) -> float:
    """Where a claim lands once a human has approved it.

    Rescales into `[ANSWERABLE_FLOOR, 1.0]` rather than multiplying. A flat multiplier gets
    both ends wrong: 0.79 x 1.3 is 1.027, which `build_assertion` refuses as out of range,
    and 0.55 x 1.3 is 0.715, still under the floor and so still unreachable. Rescaling
    preserves the order the model reported while guaranteeing both bounds by construction.

    A descriptive claim lands *on* the floor rather than in the band. Approval still makes it
    answerable — a person vouched for it — but it must not outrank a governing fact, which is
    the inversion this whole change exists to undo.

    Takes the stored confidence, not the model's self-report: a cap a model could undo by
    getting itself approved would not be a cap. `raw_confidence` keeps the self-report.
    """
    if not governing:
        return DESCRIPTIVE_CONFIDENCE
    return ANSWERABLE_FLOOR + confidence * (1.0 - ANSWERABLE_FLOOR)


class ReviewState(str, Enum):
    AUTO_ASSERTED = "AUTO_ASSERTED"
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


#: Signed off, one way or the other: a system of record or a check said so, or a person did.
#: The states retrieval admits and `promote` moves to live.
SIGNED_OFF_STATES = frozenset({ReviewState.AUTO_ASSERTED, ReviewState.APPROVED})


@dataclass(frozen=True)
class ReviewPolicy:
    """Tenant policy that can only ever *tighten* the default review rules.

    Exists because the settings it carries were previously stored, validated, warned
    about in the UI, and read by nothing — an administrator could switch a toggle,
    believe the system's behaviour had changed, and be wrong. A toggle that does nothing
    is worse than no toggle.

    Deliberately one-directional: every field here can send more work to review, never
    less. There is no setting that makes an unreviewed model claim live, because that is
    the guarantee the whole contract rests on.
    """

    auto_assert_verified: bool = True
    """When False, even a quote-confirmed presence claim waits for a human. A firm may
    want this while it is still building trust in the pipeline."""

    require_review_for_governing: bool = True
    """Force review for governing predicates whatever their class. Currently redundant
    for EXTRACTED_DET — the presence rule already refuses those — but it also catches
    DECLARED, so a case-management export claiming ADVERSE_TO still gets read by a human
    before it can drive a conflict check."""

    governing_predicates: frozenset[str] = frozenset()
    """Which predicates count as governing. Empty means the check cannot fire, so the
    caller must pass the ontology's set for `require_review_for_governing` to mean
    anything."""


#: Applied when a caller supplies no policy: the safe defaults, which are also the
#: behaviour the tests and the contract describe.
DEFAULT_REVIEW_POLICY = ReviewPolicy()


def _derive_review_state(
    epistemic_class: EpistemicClass,
    predicate: str,
    policy: ReviewPolicy | None,
) -> ReviewState:
    policy = policy or DEFAULT_REVIEW_POLICY

    if epistemic_class not in AUTO_ASSERT_CLASSES:
        return ReviewState.PENDING
    if epistemic_class is EpistemicClass.EXTRACTED_DET and not policy.auto_assert_verified:
        return ReviewState.PENDING
    if policy.require_review_for_governing and predicate in policy.governing_predicates:
        return ReviewState.PENDING
    return ReviewState.AUTO_ASSERTED


@dataclass(frozen=True)
class SourceLocator:
    """Where an assertion came from, in terms a person can actually check.

    Unstructured provenance is **file, page, and the verbatim quote**. That combination
    is what a lawyer would use by hand: open the PDF at that page, search for that
    sentence. An off-the-shelf viewer can do it with no coordinate mapping.

    Character offsets used to be the primary mechanism and were a false precision: they
    index Textract's *reconstructed text buffer*, not the PDF, so a viewer cannot seek to
    them — and re-parsing after a Textract version change shifts every offset, silently
    repointing stored assertions. They remain here as optional debug metadata, never as
    the citation.

    `quote` is load-bearing rather than decorative: it is what the viewer searches for,
    what the reviewer reads, and what `span_sha256` is taken over so a later change to
    the underlying document is detectable.

    No presigned URL is stored. They expire in hours, and an audit trail holding an
    expiring credential is provenance that stops resolving. The S3 key is stored; the
    URL is minted per request.
    """

    # Unstructured
    document_id: str | None = None
    filename: str | None = None
    """Original filename, shown to the reviewer. The S3 key is derived from
    `document_id`, so this is for display rather than retrieval."""

    page: int | None = None
    chunk_id: str | None = None
    quote: str | None = None
    """Verbatim text from the document. Must appear in the source; a paraphrase would
    make the citation unverifiable."""

    span_sha256: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    """Debug only. Offsets into the extracted text buffer, not the PDF."""

    # Structured
    source_id: str | None = None
    table: str | None = None
    column: str | None = None
    query_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.document_id is None and self.source_id is None:
            raise ValueError("SourceLocator needs either document_id or source_id")
        if self.document_id is not None and not self.quote:
            # A document citation with nothing to search for cannot be checked, which
            # defeats the point of recording it.
            raise ValueError("a document SourceLocator requires a quote to cite")

    @property
    def is_document(self) -> bool:
        return self.document_id is not None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class Assertion:
    """One fact, with everything needed to defend or retract it."""

    tenant_id: str
    subject_id: str
    predicate: str
    object_id: str
    epistemic_class: EpistemicClass
    method: str
    """Versioned and specific: `regex:bluebook_citation@v3`, `llm:claude-sonnet-5`,
    `rule:conflict_check@v2`, `glue:catalog_scan`. Version matters — when an
    extractor improves you supersede its old output rather than silently mixing
    generations."""
    confidence: float
    """How much the system trusts this *now*. Moves: a reviewer's approval rescales it into
    the answerable band. Read `raw_confidence` for what the extractor originally claimed."""

    source_locator: SourceLocator

    matter_id: str | None = None
    """Matter scoping lives on the assertion, not in a separate graph, so an
    ethical wall is a policy change rather than a data migration."""

    raw_confidence: float | None = None
    """The extractor's own self-report, before any cap or approval bump.

    Deliberately the *pre-clamp* number rather than the pre-approval one. Both were being
    discarded — `effective_model_confidence` stored only `min(raw, cap)` — and the pre-clamp
    value is the strictly larger fact: the clamp is recoverable from it plus the cap, and the
    pre-approval value is recoverable from `confidence` by inverting the rescale, whereas
    nothing recovers a self-report that was never written down. None for a fact that never
    self-reported: a catalog scan and a reviewer's correction assert their confidence rather
    than estimating it."""

    premises: tuple[str, ...] = ()
    rule_id: str | None = None
    rule_version: str | None = None

    # Bitemporal. World time: when the fact was true. Transaction time: when we
    # learned it. Both from day one — retrofitting either is brutal, and legal
    # work genuinely needs "what did the file show on the date we advised".
    valid_from: str | None = None
    valid_until: str | None = None
    recorded_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    superseded_at: str | None = None

    review_state: ReviewState = ReviewState.PENDING
    reviewed_by: str | None = None
    reviewed_at: str | None = None

    assertion_id: str = ""

    def __post_init__(self) -> None:
        if not self.assertion_id:
            self.assertion_id = self._compute_id()

    def _compute_id(self) -> str:
        """Content-addressed, so re-ingesting the same document is a no-op.

        `method` is part of the hash: re-extracting with `@v4` produces a new
        assertion that supersedes the `@v3` one instead of colliding with it.
        """
        payload = json.dumps(
            {
                "t": self.tenant_id,
                "s": self.subject_id,
                "p": self.predicate,
                "o": self.object_id,
                "m": self.method,
                "loc": self.source_locator.to_dict(),
                "vf": self.valid_from,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

    @property
    def is_current(self) -> bool:
        return self.superseded_at is None

    def to_edge_props(self) -> dict[str, Any]:
        """The fast path: just enough to filter a traversal without a second hop.

        `raw_confidence` is absent on purpose: nothing filters on it, and the edge earns its
        cheapness by carrying only what `edge_scope` reads."""
        return {
            "assertion_id": self.assertion_id,
            "tenant_id": self.tenant_id,
            "matter_id": self.matter_id,
            "epistemic_class": self.epistemic_class.value,
            "confidence": self.confidence,
            "review_state": self.review_state.value,
            "superseded_at": self.superseded_at,
        }

    def to_node_props(self) -> dict[str, Any]:
        """The audit path: full provenance, read when someone asks why."""
        props = {
            "assertion_id": self.assertion_id,
            "tenant_id": self.tenant_id,
            "matter_id": self.matter_id,
            "subject_id": self.subject_id,
            "predicate": self.predicate,
            "object_id": self.object_id,
            "epistemic_class": self.epistemic_class.value,
            "method": self.method,
            "confidence": self.confidence,
            "raw_confidence": self.raw_confidence,
            "rule_id": self.rule_id,
            "rule_version": self.rule_version,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "recorded_at": self.recorded_at,
            "superseded_at": self.superseded_at,
            "review_state": self.review_state.value,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at,
        }
        props.update({f"loc_{k}": v for k, v in self.source_locator.to_dict().items()})
        return {k: v for k, v in props.items() if v is not None}


class AssertionError_(ValueError):
    """Raised when a candidate assertion violates a contract invariant."""


def build_assertion(
    *,
    tenant_id: str,
    subject_id: str,
    predicate: str,
    object_id: str,
    epistemic_class: EpistemicClass,
    method: str,
    confidence: float,
    source_locator: SourceLocator,
    matter_id: str | None = None,
    raw_confidence: float | None = None,
    premises: tuple[str, ...] = (),
    premise_confidences: tuple[float, ...] = (),
    rule_id: str | None = None,
    rule_version: str | None = None,
    valid_from: str | None = None,
    valid_until: str | None = None,
    allowed_predicates: frozenset[str] | None = None,
    policy: ReviewPolicy | None = None,
) -> Assertion:
    """The only sanctioned way to create an assertion.

    Enforces the invariants that make the graph auditable. Every one of these is
    cheap here and expensive to bolt on later:

    1. tenant_id, method and a source locator are mandatory. An unattributed edge
       is not defensible.
    2. INFERRED requires premises, so a derived fact can always be unwound into a
       proof tree.
    3. An inference is never more confident than its weakest premise. Without this
       a chain of guesses can launder itself into a certainty.
    4. Closed-vocabulary predicates are rejected at the boundary. A conflict check
       that misses `is_counsel_to` because the graph also holds `REPRESENTS` is a
       silent, dangerous failure.
    5. Review state follows from epistemic class and tenant policy, never from the
       caller — so no code path can opt itself out of review. `policy` may only ever
       send *more* to review; there is no setting that makes an unreviewed model claim
       live.
    6. Entity ids carry no whitespace and no miscased kind prefix. Both would MERGE as a
       second node for one entity, and a conflict check joining on the other one finds
       nothing while reporting clean. `Ontology.canonical_entity_id` fixes an id; this
       refuses one nothing fixed.

    Cascading retraction is the sixth invariant and lives in
    `src.documents.retract`, since it must walk the premise graph rather than
    inspect a single candidate.
    """
    if not tenant_id:
        raise AssertionError_("tenant_id is required, untenanted edges leak across firms")
    if not method:
        raise AssertionError_("method is required, and must be versioned (e.g. regex:citation@v3)")
    if not 0.0 <= confidence <= 1.0:
        raise AssertionError_(f"confidence must be in [0,1], got {confidence}")
    _reject_unnormalised("subject_id", subject_id)
    _reject_unnormalised("object_id", object_id)
    if raw_confidence is not None and not 0.0 <= raw_confidence <= 1.0:
        raise AssertionError_(f"raw_confidence must be in [0,1], got {raw_confidence}")

    if allowed_predicates is not None and predicate not in allowed_predicates:
        raise AssertionError_(
            f"predicate {predicate!r} is not in the closed vocabulary. "
            "Governing predicates (conflicts, privilege, deadlines, citations) are "
            "closed on purpose: a query that silently misses a synonym is worse "
            "than a write that fails loudly. Map it to an approved predicate, or "
            "classify it as descriptive."
        )

    if epistemic_class is EpistemicClass.EXTRACTED_DET:
        # A quote-match proves the text is present, nothing more. Letting a verified
        # presence claim carry CITES would auto-assert reliance the check never
        # established — the exact bug this class was reshaped to prevent.
        if predicate not in PRESENCE_PREDICATES:
            raise AssertionError_(
                f"{predicate!r} cannot be EXTRACTED_DET: a verified quote establishes "
                f"only that text appears, so presence predicates "
                f"({sorted(PRESENCE_PREDICATES)}) are the only ones a check can support. "
                "Anything implying significance must be EXTRACTED_MODEL and reviewed."
            )
        if not source_locator.quote:
            raise AssertionError_(
                "EXTRACTED_DET requires the quote that was verified, without it the "
                "confirmation cannot be repeated"
            )

    if epistemic_class is EpistemicClass.INFERRED:
        if not premises:
            raise AssertionError_(
                "INFERRED assertions require premises, an inference with no stated "
                "basis cannot be explained or retracted correctly"
            )
        if premise_confidences:
            ceiling = min(premise_confidences)
            if confidence > ceiling:
                raise AssertionError_(
                    f"INFERRED confidence {confidence} exceeds weakest premise {ceiling}; "
                    "inference cannot manufacture certainty"
                )
    elif premises:
        raise AssertionError_(
            f"premises are only meaningful for INFERRED, got {epistemic_class.value}"
        )

    review_state = _derive_review_state(epistemic_class, predicate, policy)

    return Assertion(
        tenant_id=tenant_id,
        subject_id=subject_id,
        predicate=predicate,
        object_id=object_id,
        epistemic_class=epistemic_class,
        method=method,
        confidence=confidence,
        source_locator=source_locator,
        matter_id=matter_id,
        raw_confidence=raw_confidence,
        premises=tuple(premises),
        rule_id=rule_id,
        rule_version=rule_version,
        valid_from=valid_from,
        valid_until=valid_until,
        review_state=review_state,
    )
