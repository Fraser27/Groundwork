"""Runtime-tunable governance settings.

Everything here is deliberately *not* a constant. These are the knobs an
administrator has to be able to turn without a redeploy: trust thresholds, model
ids, and the kill switch. Same pattern as rosetta-sdl's `system_config` node —
defaults come from env, overrides are persisted in the graph and survive a restart.

The reason this file exists rather than a pile of module-level constants: a firm
onboarding a new practice area will want a different confidence floor, and a model
deprecation should be a settings change rather than a release.

One coupling is load-bearing enough to enforce in code rather than document:
`model_confidence_cap` must stay below `min_confidence_floor`. That gap is what
guarantees an unreviewed model assertion cannot shape an answer even if the review
gate were bypassed — defence in depth, not just policy. `validate()` refuses any
combination that closes it.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any

# Imported rather than redeclared. These previously existed in both modules with
# *different* values, and since the pipeline reads config while the UI showed
# governance, an administrator could see a model id the system was not using.
from src.constants import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EXTRACTION_MODEL,
    DEFAULT_OCR_MODEL,
    DEFAULT_SYNTHESIS_MODEL,
)

#: The question-answering model. Named separately from synthesis because a firm may
#: want a cheaper model reading questions than writing answers.
DEFAULT_QUERY_MODEL = DEFAULT_SYNTHESIS_MODEL
DEFAULT_ENRICHMENT_MODEL = DEFAULT_OCR_MODEL

#: The resolution tiers that exist. Written out rather than imported from `query.resolver`, which
#: imports this module -- so a test pins the two together instead, because a second list that can
#: drift from the enum is exactly how a retired tier survived in a default for a day.
KNOWN_TIERS = frozenset({1, 2, 3})


class GovernanceError(ValueError):
    """Raised when a settings change would break a safety invariant."""


@dataclass
class GovernanceSettings:
    """Tenant-scoped governance configuration.

    Field order matters only for how the UI groups them; see `FIELD_HELP` for the
    text shown next to each control.
    """

    # ── Trust thresholds ──────────────────────────────────────────────────────
    min_confidence_floor: float = 0.8
    """Assertions below this do not inform answers. They remain visible in the
    review queue — this gates retrieval, not storage."""

    model_confidence_cap: float = 0.79
    """Ceiling applied to every model extraction. Must stay below
    `min_confidence_floor`, so an unreviewed model claim sits under the retrieval
    floor by construction.

    It caps the *unreviewed* claim only. Approval rescales into
    `[min_confidence_floor, 1.0]` (`assertions.answerable_confidence`), so this no longer
    doubles as a permanent ceiling -- which it was, and which made approving a fact a no-op
    for retrieval."""

    auto_assert_deterministic: bool = True
    """Whether quote-verified presence claims go live without review. Turning this off
    sends even confirmed quotes to the queue — slower, but a firm may want it during
    onboarding while it builds trust in the pipeline."""

    require_review_for_governing: bool = True
    """Force review for governing predicates (conflicts, privilege, deadlines)
    regardless of epistemic class. Belt-and-braces for the predicates where a wrong
    answer is an exposure rather than an embarrassment."""

    # ── Kill switches ─────────────────────────────────────────────────────────
    block_ungoverned_queries: bool = False
    """Refuse SQL that a model wrote, rather than a metric compiled.

    Gates tier 3's SQL lane and nothing else. Not the tier: on, the question still returns its
    passages and its graph facts, `lanes_skipped["sql"]` names the reason, and the attempt is
    recorded for an administrator -- a question people keep asking is a governed metric waiting to
    be written, and that backlog is the setting's other half. Refusing the whole tier would turn a
    switch that removes an ungoverned capability into one that removes governed answers.

    It gated the retired fourth tier before that tier was found never to have existed, so for a
    time it was a control that reported itself active over nothing. For a legal product that is the
    worst direction for a failure, which is why it is tested against the lane rather than the
    setting."""

    allowed_tiers: frozenset[int] = frozenset({1, 2, 3})
    """Which resolution tiers may ever run for this tenant. A hard cap, not a default:
    a caller asking for a tier that is not in here is refused rather than silently
    served a different one, because "answered at a tier you disallowed" and "answered
    at the tier you asked for" must not look the same.

    Distinct from `block_ungoverned_queries`, which is specifically about tier 4 and
    logs its refusals for an admin to read. This is the general form, so a firm can
    also forbid, say, the hybrid tier while keeping metrics and graph traversal."""

    router_enabled: bool = True
    """Choose which tiers to run by similarity rather than trying them in order.

    Off is not a degraded mode, it is the previous behaviour: every permitted tier is attempted
    in sequence until one answers. Worth turning off while a routing index is still being
    built, since an empty index degrades to the same thing but pays a search to discover it."""

    router_min_similarity: float = 0.25
    """The floor below which a match does not count at all.

    Answers one question only: did anything in the index resemble this question? When the answer
    is no the router degrades and every permitted tier is tried, because similarity is not
    calibrated and picking the least bad layer would be a guess dressed as a decision. This is
    deliberately not a relevance threshold, and raising it far is how a tenant ends up routing
    nothing."""

    router_margin: float = 0.35
    """How much worse than the best-scoring layer a layer may be and still be searched.

    0 searches only the winning layer; 1.0 searches everything above the floor. Relative rather
    than absolute because no cosine value means "relevant" across questions of different lengths
    and phrasings — the answerable question is how much worse than the best is still worth
    trying, and this is that question."""

    router_metric_boost: float = 0.05
    """How much a governed metric outranks an equally-scoring layer.

    The protection that makes it safe for the router to be able to skip tier 1 at all: without
    it, a paraphrase scoring fractionally low routes away from deterministic compiled SQL and
    into tier 4, where a model writes the query. That is the failure the router was built to
    remove, so it must not be reintroduced by the router itself. Cannot rescue a layer that
    failed the floor — the floor is a question about the data, and a preference must not answer
    it."""

    block_model_extraction: bool = False
    """Stop reading documents with a model. Since extraction is now model-only, this
    halts new facts from documents entirely — upload, transcription and search still
    work, so a document remains findable and readable."""

    # ── Models ────────────────────────────────────────────────────────────────
    query_model: str = DEFAULT_QUERY_MODEL
    """The model that writes tier 3's SQL when no approved metric covers the question.

    Read by `Services.build_sql_lane`. It was reserved and read by nothing for most of this
    project's life, which is the same failure shape as the kill switch above: a setting an
    administrator can change and that controls nothing."""

    retrieval_agent_model: str = DEFAULT_QUERY_MODEL
    """The model that drives the Retrieval agent's tool-calling loop.

    Separate from `query_model` on purpose, even though they default the same. That one decides
    which model writes SQL; changing who writes a query must not silently change who drives the
    loop that decides whether to run one at all."""

    ocr_model: str = DEFAULT_OCR_MODEL
    """Vision model that transcribes document pages. Deliberately separate from the
    extraction model: transcription is mechanical and a cheap model does it well, so
    paying extraction rates for it is waste. Changing it does not invalidate stored
    text — the method string records which model produced each transcription."""

    extraction_model: str = DEFAULT_EXTRACTION_MODEL
    enrichment_model: str = DEFAULT_ENRICHMENT_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    """Changing this is a data migration: existing vectors were written by the old
    model and mixing generations degrades retrieval silently. The UI warns and
    requires re-embedding."""

    # ── Ontology ──────────────────────────────────────────────────────────────
    ontology_domain: str = ""
    """Which pack governs this tenant's writes. Empty means "whatever this process booted with",
    resolved by `Services.settings_for` from `GroundworkConfig.ontology_pack`.

    Empty rather than naming a pack here, because a default in this dataclass is a *second*
    independent default: it silently outranked `GROUNDWORK_ONTOLOGY_PACK`, so a deployment could
    boot one vocabulary, report it at `/health`, and validate every write against another."""

    enforce_closed_vocabulary: bool = True
    """Reject governing predicates outside the pack. Disabling this is how predicate
    sprawl starts, and sprawl is what makes a conflict check silently miss rows."""

    # ── Retrieval ─────────────────────────────────────────────────────────────
    vector_top_k: int = 20
    graph_expand_depth: int = 2

    updated_by: str | None = field(default=None)
    updated_at: str | None = field(default=None)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not 0.0 <= self.min_confidence_floor <= 1.0:
            raise GovernanceError("min_confidence_floor must be in [0,1]")
        if not 0.0 <= self.model_confidence_cap <= 1.0:
            raise GovernanceError("model_confidence_cap must be in [0,1]")
        if self.model_confidence_cap >= self.min_confidence_floor:
            raise GovernanceError(
                f"model_confidence_cap ({self.model_confidence_cap}) must stay below "
                f"min_confidence_floor ({self.min_confidence_floor}). The gap is what "
                "keeps an unreviewed model assertion out of answers even if the review "
                "gate is bypassed; closing it removes that guarantee."
            )
        if self.vector_top_k < 1:
            raise GovernanceError("vector_top_k must be >= 1")
        if not 1 <= self.graph_expand_depth <= 5:
            raise GovernanceError(
                "graph_expand_depth must be 1-5; deeper traversals fan out badly and "
                "pull in weakly related matters"
            )
        # Cosine, so -1 is meaningful in principle. Bounded at 0 because a negative floor admits
        # items pointing away from the question, which is never what an administrator means.
        if not 0.0 <= self.router_min_similarity <= 1.0:
            raise GovernanceError("router_min_similarity must be in [0,1]")
        if not 0.0 <= self.router_margin <= 1.0:
            raise GovernanceError(
                "router_margin must be in [0,1]: 0 searches only the best-scoring layer and "
                "1 searches every layer above the similarity floor"
            )
        if not 0.0 <= self.router_metric_boost <= 1.0:
            raise GovernanceError("router_metric_boost must be in [0,1]")
        # Validated rather than trusted to every default and decoder being right. A `4` survived
        # here from `from_env`'s default long after the tier was retired, and it only surfaced
        # because the Admin page displayed it -- a tier the code has no member for reaches
        # `Tier(4)` in the resolver and raises where a refusal belongs.
        if unknown := sorted(t for t in self.allowed_tiers if t not in KNOWN_TIERS):
            raise GovernanceError(
                f"allowed_tiers contains {unknown}, which is not a resolution tier. "
                f"Permitted: {sorted(KNOWN_TIERS)}."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_env(cls) -> GovernanceSettings:
        def _f(name: str, default: float) -> float:
            raw = os.getenv(name)
            return float(raw) if raw else default

        def _b(name: str, default: bool) -> bool:
            raw = os.getenv(name)
            return raw.lower() in {"1", "true", "yes"} if raw else default

        def _tiers(name: str, default: frozenset[int]) -> frozenset[int]:
            """A comma-separated tier list, e.g. "1,2". Ignores anything unparseable
            rather than failing: a typo in a cap must not stop the API starting, and an
            empty result falls back to the default rather than silently allowing none."""
            raw = os.getenv(name)
            if not raw:
                return default
            # `[1-3]`, so a stale `GROUNDWORK_ALLOWED_TIERS=1,2,3,4` from before the fourth tier
            # was retired drops the 4 rather than resurrecting a tier that no longer exists.
            parsed = {int(p) for p in re.findall(r"[1-3]", raw)}
            return frozenset(parsed) if parsed else default

        return cls(
            min_confidence_floor=_f("GROUNDWORK_MIN_CONFIDENCE", 0.8),
            model_confidence_cap=_f("GROUNDWORK_MODEL_CONFIDENCE_CAP", 0.79),
            auto_assert_deterministic=_b("GROUNDWORK_AUTO_ASSERT_DET", True),
            require_review_for_governing=_b("GROUNDWORK_REVIEW_GOVERNING", True),
            block_ungoverned_queries=_b("GROUNDWORK_BLOCK_UNGOVERNED", False),
            allowed_tiers=_tiers("GROUNDWORK_ALLOWED_TIERS", frozenset({1, 2, 3})),
            router_enabled=_b("GROUNDWORK_ROUTER_ENABLED", True),
            router_min_similarity=_f("GROUNDWORK_ROUTER_MIN_SIMILARITY", 0.25),
            router_margin=_f("GROUNDWORK_ROUTER_MARGIN", 0.35),
            router_metric_boost=_f("GROUNDWORK_ROUTER_METRIC_BOOST", 0.05),
            block_model_extraction=_b("GROUNDWORK_BLOCK_MODEL_EXTRACTION", False),
            query_model=os.getenv("GROUNDWORK_QUERY_MODEL", DEFAULT_QUERY_MODEL),
            retrieval_agent_model=os.getenv("GROUNDWORK_RETRIEVAL_AGENT_MODEL", DEFAULT_QUERY_MODEL),
            ocr_model=os.getenv("GROUNDWORK_OCR_MODEL", DEFAULT_OCR_MODEL),
            extraction_model=os.getenv("GROUNDWORK_EXTRACTION_MODEL", DEFAULT_EXTRACTION_MODEL),
            enrichment_model=os.getenv("GROUNDWORK_ENRICHMENT_MODEL", DEFAULT_ENRICHMENT_MODEL),
            embedding_model=os.getenv("GROUNDWORK_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            # No fallback pack: empty defers to the boot pack, which `GROUNDWORK_ONTOLOGY_PACK`
            # already sets. This variable exists to override that for one tenant.
            ontology_domain=os.getenv("GROUNDWORK_ONTOLOGY_DOMAIN", ""),
            enforce_closed_vocabulary=_b("GROUNDWORK_CLOSED_VOCABULARY", True),
            vector_top_k=int(os.getenv("GROUNDWORK_VECTOR_TOP_K", "20")),
            graph_expand_depth=int(os.getenv("GROUNDWORK_GRAPH_DEPTH", "2")),
        )

    def apply(self, patch: dict[str, Any], *, updated_by: str) -> GovernanceSettings:
        """Return a new validated instance with `patch` applied.

        Returns rather than mutates so a rejected change cannot leave the live
        settings half-updated.
        """
        known = {f.name for f in fields(self)} - {"updated_by", "updated_at"}
        unknown = set(patch) - known
        if unknown:
            raise GovernanceError(f"unknown settings: {sorted(unknown)}")

        merged = {**self.to_dict(), **patch}
        merged["updated_by"] = updated_by
        merged.pop("updated_at", None)
        candidate = GovernanceSettings(**merged)
        return candidate

    def effective_model_confidence(self, raw: float) -> float:
        """Clamp a model extractor's self-reported confidence."""
        return min(raw, self.model_confidence_cap)

    def with_raised_floor(self, requested: float | None) -> GovernanceSettings:
        """These settings, with the confidence floor raised for one question.

        **Raised only.** A request may be stricter than its tenant and never laxer: the floor is a
        governance control, so a caller able to lower it could opt out of one from the Ask page.
        The same rule the tier router follows -- a request narrows, never widens.

        The clamp lives here rather than in a route handler so both endpoints inherit it. A floor
        honoured by one handler and not the other is how `/query` and `/query/compose` come to
        disagree about the same question, which has happened often enough here to design out.
        """
        if requested is None or requested <= self.min_confidence_floor:
            return self
        return replace(self, min_confidence_floor=min(requested, 1.0))


#: Plain-language help for each control, surfaced as UI tooltips. Written for a
#: lawyer-administrator, not a data engineer — hence "will not be used in answers"
#: rather than "excluded from the retrieval candidate set".
FIELD_HELP: dict[str, str] = {
    "min_confidence_floor": (
        "How certain the system must be before a fact is used in an answer. Facts below "
        "this still appear in the review queue, they just do not influence results. "
        "Raising it makes answers more conservative and may leave questions unanswered."
    ),
    "model_confidence_cap": (
        "The highest confidence an AI-extracted fact may claim while it is still unreviewed. "
        "Kept just below the confidence floor on purpose, so an AI claim can never influence "
        "an answer until a person has approved it. The system will not let you raise this to "
        "or above the floor. Approving a fact lifts it above the floor, in the order the AI "
        "reported, so this is a gate rather than a permanent ceiling."
    ),
    "auto_assert_deterministic": (
        "Whether a fact goes straight into the knowledge graph when its quoted words were "
        "found on the page. The system checks the quote is really there, so nothing is "
        "taken on the AI's word, but the check only settles that the words appear, never "
        "what they mean. Switch off to have a person confirm even those."
    ),
    "require_review_for_governing": (
        "Always require human approval for the relationships that carry legal "
        "consequence: who represents whom, who is adverse to whom, privilege, and "
        "deadlines. Recommended on."
    ),
    "block_ungoverned_queries": (
        "Stop the AI writing its own database queries. Questions are still answered from "
        "approved metrics, the knowledge graph and document passages, so this removes one way "
        "of answering rather than refusing the question. Blocked attempts are recorded so you "
        "can see which figures people want and write a metric for them."
    ),
    "allowed_tiers": (
        "Which ways of answering a question are permitted at all. Users choose freely "
        "within this list; anything outside it is refused rather than quietly answered "
        "a different way. 1 is an approved metric, 2 is the knowledge graph, 3 combines "
        "passages, graph relationships and table schemas. Removing 3 also removes AI-written "
        "SQL; to keep 3 and remove only that, use the ungoverned-queries switch above."
    ),
    "router_enabled": (
        "Choose which ways of answering to try by comparing the question against what this "
        "tenant actually holds, instead of trying each in turn. Turn it off and every permitted "
        "route is attempted in order, which is how the system behaved before."
    ),
    "router_min_similarity": (
        "How closely something must resemble the question to count as a match at all. This is "
        "only a detector for 'nothing here looks related': when nothing clears it, every "
        "permitted route is tried rather than a guess being made. Raise it far and questions "
        "stop being routed."
    ),
    "router_margin": (
        "How much less relevant than the best match a route may look and still be searched. 0 "
        "searches only the strongest match; 1 searches everything that cleared the floor. It is "
        "expressed as a comparison rather than a score because relevance numbers are not "
        "percentages and cannot be read as one."
    ),
    "router_metric_boost": (
        "How much an approved metric is favoured over an equally relevant-looking route. It "
        "exists so a rephrased question still reaches a governed metric rather than falling "
        "through to AI-written SQL. It cannot make an unrelated metric look related."
    ),
    "block_model_extraction": (
        "Stop using AI to read documents for facts. Uploading, page transcription and "
        "search keep working, so documents stay findable, but no new relationships are "
        "proposed from them."
    ),
    "query_model": (
        "The AI model that writes SQL for a question no approved metric covers. Read on the "
        "tier 3 path, so a change takes effect on the next question. It never writes SQL for a "
        "governed metric: those are compiled from a definition somebody approved."
    ),
    "retrieval_agent_model": (
        "The AI model that drives the Retrieval agent, which answers by calling this system's "
        "own tools and shows every call it made. Deliberately separate from the query model: "
        "which model writes a query and which one decides whether to run one are different "
        "choices, and a firm may want to pay for a stronger model on only one of them."
    ),
    "ocr_model": (
        "The AI model that reads document pages and turns them into text. It also "
        "describes charts, diagrams, signature blocks and handwriting, which plain OCR "
        "cannot. A cheaper, faster model is the right choice here, reading words off a "
        "page is mechanical work. Changing it does not affect documents already "
        "processed; each one records which model read it."
    ),
    "extraction_model": "The AI model used to read documents and propose relationships.",
    "enrichment_model": (
        "The AI model used to write descriptions for database tables and columns. A "
        "smaller, cheaper model is usually enough here."
    ),
    "embedding_model": (
        "The model that converts document text into vectors for similarity search. "
        "CHANGING THIS REQUIRES RE-PROCESSING EVERY DOCUMENT, existing vectors came "
        "from the old model and mixing them degrades search quality without any visible "
        "error."
    ),
    "ontology_domain": (
        "Which vocabulary of entities and relationships to use. 'legal' covers matters, "
        "parties, counsel, authorities and deadlines."
    ),
    "enforce_closed_vocabulary": (
        "Reject relationships the ontology does not define. This is what stops the same "
        "idea being recorded five different ways, if that happens, a conflict check can "
        "return no results and look like a clean report. Strongly recommended on."
    ),
    "vector_top_k": "How many document passages to retrieve before expanding into the graph.",
    "graph_expand_depth": (
        "How many relationship hops to follow out from a retrieved passage. Two is "
        "usually right; more pulls in weakly related matters."
    ),
}
