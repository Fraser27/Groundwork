"""The single document extraction path: one model call per chunk.

There is no regex layer any more, and its absence is the design. Allowlists of court
names and reporter abbreviations silently missed every court nobody had enumerated, and
the offsets they returned indexed the reconstructed text buffer rather than the PDF, so
no viewer could seek to them. Provenance is now file, page and verbatim quote — what a
lawyer would use by hand.

What replaces it is a split by *checkability*, not by who proposed the claim:

- The model says some text appears in the passage. A string search either confirms it or
  does not. Confirmed, that is EXTRACTED_DET and auto-asserts — but only ever as
  MENTIONS, because presence is the whole of what a quote-match establishes. The old
  parser auto-asserted CITES, so "the court declined to follow Brown" was recorded as
  reliance on Brown.
- The model says one holding undercuts another. Nothing can confirm that, so it is
  EXTRACTED_MODEL, capped below the retrieval floor, and waits for a human.

The split is what keeps the review queue readable: if every model claim needed review, a
300-page bundle would produce hundreds of pending items and a queue nobody can clear
gets rubber-stamped.

Two guards throughout, both because model output is untrusted input. Predicates are
grounded by exact match with no LLM adjudication, so an invented predicate is dropped
rather than quietly widening the schema. Quotes must be present in the chunk, because a
citation nobody can search for cannot be checked.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from src.constants import DEFAULT_EXTRACTION_MODEL
from src.documents.models import Chunk
from src.graph.assertions import (
    DESCRIPTIVE_CONFIDENCE,
    PRESENCE_PREDICATES,
    Assertion,
    AssertionError_,
    EpistemicClass,
    ReviewPolicy,
    build_assertion,
)
from src.ontology.loader import EntityDef, Ontology, PredicateDef

logger = logging.getLogger(__name__)

#: What an entity claim is emitted as. `build_assertion` refuses any other predicate for
#: EXTRACTED_DET, so this is the only edge a verified quote can auto-assert.
MENTIONS = "MENTIONS"

#: Ceiling on interpretive confidence regardless of what the model self-reports. Its
#: stated certainty is not evidence, and the retrieval floor is 0.80 — so an unreviewed
#: interpretation sits below the floor by construction and cannot shape an answer.
MAX_MODEL_CONFIDENCE = 0.79

#: Confidence for a quote-verified presence claim. Not 1.0: the search proved the text
#: is there, but the entity id it was normalised to is still the model's work.
#:
#: Kept at 0.95 rather than lowered, and unreachable today. `PRESENCE_PREDICATES` is `{MENTIONS}`
#: and MENTIONS is descriptive, so every verified claim currently lands on the floor instead.
#: This is what a *governing* predicate would be worth if a check ever became able to confirm
#: one -- deleting it would leave that case scored by accident, and 0.95 is the value the
#: reasoning behind it still supports.
VERIFIED_PRESENCE_CONFIDENCE = 0.95

#: Versions the *check*, not the model. If quote matching ever changes, this bumps and
#: the previous generation is superseded rather than mixed with the new one.
QUOTE_CHECK = "verify:quote@v1"

SYSTEM_PROMPT = """\
You read documents for a system where every claim must be defensible to a regulator.

Return two kinds of claim about the passage, and keep them separate, because they are \
checked differently:

1. `entities` — something named in the passage: a person, an organisation, a reference \
number, a date, a provision. This is a claim about PRESENCE ONLY. It is verified by \
searching the passage for your quote, so the quote must be copied exactly.

2. `relationships` — something the passage supports that no search could confirm: that one \
provision replaces an earlier one, that a decision was taken under a clause rather than \
merely citing it, that one organisation directs another. These go to a human reviewer.

Rules:
- Use ONLY predicates from `allowed_predicates`. An unknown predicate is discarded.
- Read the `meaning` on every predicate and entity kind before choosing one, and choose on \
what it says rather than on which name reads closest. Terms that look interchangeable are the \
ones whose `meaning` explains why they are not, and the difference is usually which of them a \
reviewer can act on.
- `quote` MUST be copied verbatim from the passage. A paraphrase is discarded, because a \
citation nobody can search for cannot be checked.
- Use the shortest quote that carries the claim.
- `id` is `<kind>:<slug>`, the kind lowercased and the slug a lowercase hyphenated form of the \
words the passage itself uses for the thing. Two documents that name one thing the same way \
must produce the same id, or they hold it as two nodes and nothing joins them.
- The prefix MUST be one of `entity_kinds`, and `kind` must match it. Do not invent a kind, \
however well it fits: an unlisted one is discarded, so the entity is lost rather than \
approximated. Use the closest listed kind, or omit the claim if none fits.
- A predicate in `allowed_predicates` may carry `subject_kinds` and `object_kinds`. Those are \
enforced: `subject_id` must be of a listed subject kind and `object_id` of a listed object kind, \
or the claim is REJECTED. The two ends are not interchangeable. A predicate declaring \
`subject_kinds: ["A"]` and `object_kinds: ["B"]` is written \
`{"subject_id": "a:...", "object_id": "b:..."}` and never the reverse.
- Write the direction the predicate's `meaning` describes, not the direction the sentence \
happens to use. "A is owned by B" and "B owns A" are one fact, and both are written with B as \
the subject of an ownership predicate. Do not also emit the inverse: two opposite claims about \
one relationship contradict each other, and both will be believed.
- `this_matter` is the case or engagement this document belongs to, and is available as a \
subject id. Use it for any claim about that rather than about the paper. It may be null, in \
which case omit claims that would need it.
- Do not guess. If the passage does not clearly support a claim, omit it.
- Within that limit, state every relationship the passage does support, including ones that \
read as routine. A finding is often two ordinary facts from two documents, so a fact omitted \
for being unremarkable is the half that was missing.
- Return ONLY a JSON object, no prose.

Schema:
{"entities": [{"id": str, "kind": str, "quote": str}],
 "relationships": [{"subject_id": str, "predicate": str, "object_id": str, \
"quote": str, "confidence": float, "reasoning": str}]}"""


class BedrockLike(Protocol):
    def converse(self, **kwargs: Any) -> dict[str, Any]: ...


class ModelExtractionFailed(RuntimeError):
    """The model returned something unusable, or could not be reached."""


@dataclass(frozen=True)
class ProposedClaim:
    """A model claim before any check.

    Entity and relationship claims collapse into one shape deliberately: what decides
    how a claim is treated is its grounded predicate, never which list it arrived in.
    A model cannot promote its own claim to auto-assert by filing it under `entities`.
    """

    subject_id: str
    predicate: str
    object_id: str
    quote: str
    confidence: float = 0.5
    reasoning: str = ""


def _meaning(defn: PredicateDef | EntityDef) -> str:
    """A term's `description` and `help` as one string.

    Joined rather than sent as two fields: the pack splits them by who is reading, and a model
    offered a one-line gloss beside a longer note will act on the gloss. For the terms that
    actually get confused, the distinction only exists in the `help`.
    """
    return " ".join(part for part in (defn.description, defn.help) if part)


def build_prompt(
    chunk: Chunk,
    *,
    predicates: Mapping[str, PredicateDef],
    entities: Mapping[str, EntityDef],
    unit_slug: str = "matter",
) -> str:
    """The user turn: one passage, the closed vocabulary with what each term means, nothing else.

    No list of known courts or reporters. That was the allowlist failure — anything not
    enumerated was absent from the graph with no error anywhere.

    The vocabulary goes out with its `description` and `help`, not only its names, and that is
    the difference between a closed vocabulary and a usable one. Measured on the retail pack:
    `Company` and `Merchant` arrived as two bare words, so a non-trading holding company was
    typed as a seller and a rule whose chain must end at a Merchant found nothing — while
    `Company.help` says in as many words that calling it a Merchant asserts with a citation
    that it trades here, which the page denies. `APPROVES_EXCEPTION_FOR` was read as
    `AUTHORISED_BY` for the same reason, and its `help` distinguishes the two explicitly. The
    answers were written down in the pack and never sent.

    Each predicate's declared subject and object kinds ride along too, because the vocabulary
    being closed is only half of what a write must satisfy: a predicate declared Matter -> Party
    refuses a party-to-party claim. Sending names without shapes asked the model to guess an
    orientation and then rejected the guess silently.

    `this_matter` offers the chunk's own organising unit as a subject. Without it the model has no
    id for "the engagement this document belongs to" and no way to state the fact a conflict check
    reads. It is sent as an *entity id*, because a bare filing reference is not one: the model
    echoed `NTL` back as an `ADVERSE_TO` subject and `validate` dropped the claim for an unknown
    entity kind, which starved the conflict rule while the document looked like it had nothing
    to say.
    """
    return json.dumps(
        {
            "page": chunk.page,
            "passage": chunk.text,
            # One list per vocabulary, carrying everything about a term. Names here and meanings
            # or shapes alongside would be two renderings of one closed set, which is two places
            # for it to disagree.
            "allowed_predicates": [
                {
                    "predicate": p,
                    "meaning": _meaning(pdef),
                    **(
                        {
                            "subject_kinds": list(pdef.domain),
                            "object_kinds": list(pdef.range),
                        }
                        if pdef.domain
                        else {}
                    ),
                }
                for p, pdef in sorted(predicates.items())
            ],
            "entity_kinds": [
                {"kind": k, "meaning": _meaning(edef)} for k, edef in sorted(entities.items())
            ],
            "this_matter": f"{unit_slug}:{chunk.matter_id}" if chunk.matter_id else None,
        },
        indent=2,
    )


def parse_response(payload: str) -> list[ProposedClaim]:
    """Pull claims out of the model's reply.

    Tolerates prose around the JSON (models add it despite instructions) but not
    malformed structure — a half-parsed claim is worse than none.
    """
    match = re.search(r"\{.*\}", payload, re.DOTALL)
    if not match:
        raise ModelExtractionFailed("no JSON object in model response")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise ModelExtractionFailed(f"model response was not valid JSON: {e}") from e

    claims: list[ProposedClaim] = []
    for raw in data.get("entities") or []:
        try:
            claims.append(
                ProposedClaim(
                    # Subject is filled in from the chunk: an entity claim is always
                    # "this document mentions X". Confidence is not read from the model
                    # at all — `validate` sets it once the quote check has run.
                    subject_id="",
                    predicate=MENTIONS,
                    object_id=str(raw["id"]),
                    quote=str(raw.get("quote", "")),
                )
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("discarding malformed entity %s: %s", raw, e)

    for raw in data.get("relationships") or []:
        try:
            claims.append(
                ProposedClaim(
                    subject_id=str(raw.get("subject_id", "")),
                    predicate=str(raw["predicate"]),
                    object_id=str(raw["object_id"]),
                    quote=str(raw.get("quote", "")),
                    confidence=float(raw.get("confidence", 0.5)),
                    reasoning=str(raw.get("reasoning", "")),
                )
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("discarding malformed relationship %s: %s", raw, e)
    return claims


def document_entity_id(chunk: Chunk) -> str:
    """Graph node id for the chunk's document, matching `DocumentMeta.entity_id`.

    Derived rather than passed in, so the subject of an assertion and the document its
    locator cites cannot drift apart.
    """
    return f"document:{chunk.document_id}"


def matter_entity_id(chunk: Chunk, ontology: Ontology | None = None) -> str | None:
    """Graph node id for the chunk's organising unit, or None when the document is unfiled.

    `Chunk.matter_id` is a filing reference (`NTL`), not an entity id, and the kind it belongs to
    is whatever the pack calls its organising unit. Hardcoding `matter:` minted an id of a kind
    only the legal pack declares, so under healthcare `entity_kind_of` returned None and every
    claim about the encounter was dropped -- the same silent drop that starved the conflict rules,
    one pack over.

    The scoping key stays `matter_id`; this is the *prefix* the pack declares for it. Falls back to
    `matter:` when no pack is supplied or none declares a unit, which is the legal default.
    """
    if not chunk.matter_id:
        return None
    unit = ontology.organising_unit if ontology is not None else None
    return f"{unit.slug if unit is not None else 'matter'}:{chunk.matter_id}"


def locate_quote(chunk: Chunk, quote: str) -> tuple[int, int] | None:
    """Global offsets of the verbatim span, tolerating only whitespace differences.

    Whitespace is normalised because a transcription's line breaks legitimately differ
    from how a model echoes a passage back. Anything beyond that is paraphrase and the
    claim is dropped: a quote the reader cannot find in the document is not provenance.
    """
    if not quote.strip():
        return None
    index = chunk.text.find(quote)
    if index >= 0:
        return chunk.char_start + index, chunk.char_start + index + len(quote)

    pattern = r"\s+".join(re.escape(tok) for tok in quote.split())
    match = re.search(pattern, chunk.text)
    if match is None:
        return None
    return chunk.char_start + match.start(), chunk.char_start + match.end()


class ModelExtractor:
    def __init__(
        self,
        ontology: Ontology,
        *,
        bedrock: BedrockLike | None = None,
        bedrock_factory: Callable[[], BedrockLike] | None = None,
        model_id: str = DEFAULT_EXTRACTION_MODEL,
        region: str | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        settings: Any | None = None,
    ) -> None:
        self.ontology = ontology
        # A tenant's governance settings, when supplied, replace the module defaults for
        # the confidence cap and the review policy. Previously these were configurable in
        # Admin and read by nothing, so a lowered cap had no effect.
        self.settings = settings
        self.model_id = model_id
        self.region = region
        self.max_tokens = max_tokens
        self.temperature = temperature
        """Omitted from the request by default. The newest Claude models reject the
        parameter outright — sending 0.0 for reproducibility's sake failed every
        extraction with a ValidationException, which is worse than not pinning it."""

        self._bedrock = bedrock
        self._bedrock_factory = bedrock_factory

    @property
    def bedrock(self) -> BedrockLike:
        if self._bedrock is None:
            factory = self._bedrock_factory
            if factory is None:
                import boto3

                factory = lambda: boto3.client("bedrock-runtime", region_name=self.region)
            self._bedrock = factory()
        return self._bedrock

    @property
    def method(self) -> str:
        return f"llm:{self.model_id}"

    @property
    def verified_method(self) -> str:
        """Records both halves: the model proposed it, the quote check confirmed it."""
        return f"{self.method}+{QUOTE_CHECK}"

    def invoke(self, prompt: str) -> str:
        # Converse, not InvokeModel: the extraction model is admin-selectable across Nova and
        # Anthropic, and their native request bodies are mutually invalid.
        inference: dict[str, Any] = {"maxTokens": self.max_tokens}
        if self.temperature is not None:
            inference["temperature"] = self.temperature

        response = self.bedrock.converse(
            modelId=self.model_id,
            system=[{"text": SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig=inference,
        )
        blocks = response.get("output", {}).get("message", {}).get("content") or []
        return "".join(part.get("text", "") for part in blocks)

    def extract(self, chunk: Chunk) -> list[Assertion]:
        """One model call over one chunk, returning validated assertions.

        Per chunk rather than per document because a chunk carries a single page, and
        the page is half of the citation.
        """
        prompt = build_prompt(
            chunk,
            # Not every known predicate: a rule's conclusion is excluded, so a model cannot
            # propose `POTENTIAL_CONFLICT` as though it had read one off a page. A conflict is
            # derived from two signed-off facts and carries them as premises; one a model
            # asserted directly would look identical and defend nothing.
            predicates={
                p: pdef
                for p in self.ontology.extractable_predicates
                if (pdef := self.ontology.predicates.get(p)) is not None
            },
            entities=self.ontology.entities,
            unit_slug=(
                unit.slug if (unit := self.ontology.organising_unit) is not None else "matter"
            ),
        )
        try:
            raw = self.invoke(prompt)
        except Exception as e:
            raise ModelExtractionFailed(f"bedrock invoke failed: {e}") from e
        return self.validate(parse_response(raw), chunk=chunk)

    def extract_document(self, chunks: Sequence[Chunk]) -> list[Assertion]:
        """Extract across a whole document, collapsing duplicate claims.

        A failure anywhere aborts the pass. Staging is all-or-nothing, so a document
        half-extracted must not look fully extracted — and S3 still holds the bytes, so
        re-running costs a re-extraction and nothing else.
        """
        found: dict[str, Assertion] = {}
        for chunk in chunks:
            for assertion in self.extract(chunk):
                found[assertion.assertion_id] = assertion
        return list(found.values())

    def _clamp(self, raw: float) -> float:
        """Apply the tenant's model-confidence cap, or the module default."""
        if self.settings is not None:
            return self.settings.effective_model_confidence(raw)
        return min(max(raw, 0.0), MAX_MODEL_CONFIDENCE)

    def _presence_confidence(self, predicate: str) -> float:
        """What a quote-verified claim is worth, which depends on what it claims.

        A confirmed `MENTIONS` is certain about something nearly worthless -- the string is on
        the page -- so it sits on the floor rather than above every reviewed fact in the graph.
        Keyed off the pack's descriptive set, not a list of predicate names, so the healthcare
        pack gets the same treatment without a code change.
        """
        if predicate in self.ontology.descriptive_predicates:
            return DESCRIPTIVE_CONFIDENCE
        return VERIFIED_PRESENCE_CONFIDENCE

    def _policy(self) -> ReviewPolicy | None:
        """Translate governance settings into the review policy the contract enforces.

        `governing_predicates` comes from the ontology rather than the settings, so
        `require_review_for_governing` means "the predicates this domain calls governing"
        instead of a list an administrator has to keep in step by hand.
        """
        if self.settings is None:
            return None
        return ReviewPolicy(
            auto_assert_verified=self.settings.auto_assert_deterministic,
            require_review_for_governing=self.settings.require_review_for_governing,
            governing_predicates=self.ontology.governing_predicates,
        )

    def validate(self, proposed: Sequence[ProposedClaim], *, chunk: Chunk) -> list[Assertion]:
        """Turn claims into assertions, dropping everything that cannot be checked.

        Separate from `extract` so the whole decision path is testable without Bedrock —
        which matters, because this is the code that decides what an untrusted model may
        put in the graph.
        """
        subject_default = document_entity_id(chunk)
        assertions: dict[str, Assertion] = {}
        for claim in proposed:
            predicate = self.ontology.ground(claim.predicate)
            if predicate is None:
                logger.info("dropped ungroundable predicate %r", claim.predicate)
                continue

            if predicate in self.ontology.rule_conclusions:
                # Refused here as well as omitted from the prompt, because a prompt is a request
                # and this is the boundary. Only a rule may conclude these, and a conclusion
                # without premises defends nothing -- letting a model assert one would make a
                # guess indistinguishable from a derivation.
                logger.info("dropped %s: only a rule may conclude it", predicate)
                continue

            span = locate_quote(chunk, claim.quote)
            if span is None:
                logger.info(
                    "dropped %s: quote %r is not verbatim in %s",
                    predicate,
                    claim.quote[:60],
                    chunk.chunk_id,
                )
                continue

            subject = claim.subject_id or subject_default
            # Before the self-edge test on purpose: `party:acme` and `Party:Acme` are one entity,
            # so a claim relating them is a self-edge and the raw comparison would miss it. Also
            # before `canonical_pair`, which sorts endpoints by their raw bytes -- so without this
            # one symmetric fact spelled two ways produces two orderings and two content hashes.
            subject_canonical = self.ontology.canonical_entity_id(subject)
            object_canonical = self.ontology.canonical_entity_id(claim.object_id)
            subject = subject_canonical if subject_canonical is not None else subject
            object_id = object_canonical if object_canonical is not None else claim.object_id
            if subject == object_id:
                continue

            # Entity kinds are closed, exactly as governing predicates are. A model free to invent
            # one produces a graph nobody can query: `vessel:mv-aurelia` and `ship:mv-aurelia` are
            # two nodes and a traversal finds one of them. Refused here rather than only asked for
            # in the prompt, because a prompt is a request and this is the boundary.
            unknown = [
                e for e in (subject, object_id) if self.ontology.entity_kind_of(e) is None
            ]
            if unknown:
                logger.info(
                    "dropped %s: entity kind not in the %s vocabulary: %s",
                    predicate,
                    self.ontology.domain,
                    ", ".join(unknown),
                )
                continue

            # The one decision that matters: a quote-match can confirm presence and
            # nothing else, so only a presence predicate earns the deterministic class.
            verified = predicate in PRESENCE_PREDICATES
            # Collapse a symmetric predicate's endpoints, so "A adverse to B" and
            # "B adverse to A" are one fact rather than two competing edges.
            subject, object_id = self.ontology.canonical_pair(predicate, subject, object_id)

            try:
                assertion = build_assertion(
                    tenant_id=chunk.tenant_id,
                    subject_id=subject,
                    predicate=predicate,
                    object_id=object_id,
                    epistemic_class=(
                        EpistemicClass.EXTRACTED_DET if verified else EpistemicClass.EXTRACTED_MODEL
                    ),
                    method=self.verified_method if verified else self.method,
                    confidence=(
                        self._presence_confidence(predicate)
                        if verified
                        else self._clamp(claim.confidence)
                    ),
                    # What the model said, before the cap took it. The cap was overwriting the
                    # self-report in place, so the number a reviewer read was unattributable to
                    # anything -- neither the model's claim nor a decision anyone made.
                    raw_confidence=min(max(claim.confidence, 0.0), 1.0),
                    source_locator=chunk.to_locator(*span),
                    matter_id=chunk.matter_id,
                    allowed_predicates=self.ontology.allowed_for(predicate),
                    # After `canonical_pair`, so the check sees the orientation that will be
                    # stored rather than the one the model happened to write.
                    endpoint_kinds=self.ontology.endpoint_kinds(predicate),
                    policy=self._policy(),
                )
            except AssertionError_ as e:
                logger.info("dropped %s: %s", predicate, e)
                continue
            assertions[assertion.assertion_id] = assertion
        return list(assertions.values())
