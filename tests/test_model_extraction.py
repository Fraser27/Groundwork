"""Tests for the single extraction path.

These pin down what an *untrusted* model is allowed to put in the graph. Bedrock is a
fake throughout — an extractor you cannot test offline is an extractor nobody tests.

The split under test is the one the design rests on: a claim a string search can confirm
auto-asserts, and only ever as MENTIONS; a claim nothing can confirm goes to review below
the retrieval floor. Get that wrong in either direction and the product breaks — too
strict and the review queue is unreadable, too loose and an LLM's opinion is
indistinguishable from a checked fact.
"""

from __future__ import annotations

import io
import json

import pytest

from src.documents.extractors.model import (
    MAX_MODEL_CONFIDENCE,
    VERIFIED_PRESENCE_CONFIDENCE,
    ModelExtractionFailed,
    ModelExtractor,
    ProposedClaim,
    build_prompt,
    locate_quote,
    parse_response,
)
from src.documents.models import Chunk
from src.graph.assertions import (
    PRESENCE_PREDICATES,
    AssertionError_,
    EpistemicClass,
    ReviewState,
    build_assertion,
)
from src.ontology.loader import load_ontology

TENANT = "firm-acme"
ONTOLOGY = load_ontology("legal")

PASSAGE = (
    "The court in Roe v. Wade, 410 U.S. 113 (1973), established the framework. "
    "That holding undercuts the defendant's argument that no such right exists. "
    "The 2019 agreement supersedes the earlier memorandum of understanding."
)

QUOTE = "That holding undercuts the defendant's argument"


class FakeBedrock:
    """Returns a canned Claude response body. Records what it was asked."""

    def __init__(self, payload) -> None:
        self.payload = payload
        self.requests: list[dict] = []

    def invoke_model(self, **kw):
        self.requests.append({"modelId": kw["modelId"], "body": json.loads(kw["body"])})
        body = (
            self.payload
            if isinstance(self.payload, dict)
            else {"content": [{"text": self.payload}]}
        )
        return {"body": io.BytesIO(json.dumps(body).encode())}


def chunk(text: str = PASSAGE, *, page: int = 1, start: int = 500, **over) -> Chunk:
    return Chunk(
        document_id="doc-1",
        tenant_id=TENANT,
        filename="brief.pdf",
        ordinal=0,
        page=page,
        char_start=start,
        char_end=start + len(text),
        text=text,
        **over,
    )


def extractor(payload="{}") -> ModelExtractor:
    return ModelExtractor(ONTOLOGY, bedrock=FakeBedrock(payload))


def entity(object_id="party:acme-corporation", *, quote=QUOTE) -> ProposedClaim:
    return ProposedClaim(subject_id="", predicate="MENTIONS", object_id=object_id, quote=quote)


def interpretation(*, predicate="DISTINGUISHES", quote=QUOTE, **over) -> ProposedClaim:
    kw = {
        "subject_id": "document:doc-1",
        "predicate": predicate,
        "object_id": "authority:410-us-113",
        "quote": quote,
        "confidence": 0.65,
    }
    kw.update(over)
    return ProposedClaim(**kw)


class TestQuoteVerification:
    """A quote the reader cannot find in the document is not provenance."""

    def test_paraphrase_is_dropped(self):
        """Both kinds. A paraphrase cannot verify presence and cannot cite an
        interpretation, so neither survives it."""
        assert (
            extractor().validate([entity(quote="the holding weakens their case")], chunk=chunk())
            == []
        )
        assert (
            extractor().validate(
                [interpretation(quote="the holding basically weakens their case")],
                chunk=chunk(),
            )
            == []
        )

    def test_verbatim_quote_survives(self):
        [a] = extractor().validate([interpretation()], chunk=chunk())
        assert a.source_locator.quote == QUOTE

    def test_whitespace_differences_are_tolerated(self):
        """A transcription's line breaks legitimately differ from how a model echoes back."""
        assert (
            len(
                extractor().validate(
                    [interpretation(quote="That holding\n  undercuts")], chunk=chunk()
                )
            )
            == 1
        )

    def test_empty_quote_is_dropped(self):
        assert extractor().validate([interpretation(quote="")], chunk=chunk()) == []
        assert extractor().validate([interpretation(quote="   ")], chunk=chunk()) == []

    def test_quote_from_a_different_page_is_dropped(self):
        """Verification is against the chunk, so a claim cannot borrow another page's text."""
        other = chunk("Wholly unrelated text about procedure.", page=4, start=9000)
        assert extractor().validate([interpretation()], chunk=other) == []

    def test_locator_names_the_file_and_page(self):
        [a] = extractor().validate([interpretation()], chunk=chunk(page=7))
        loc = a.source_locator
        assert (loc.document_id, loc.filename, loc.page) == ("doc-1", "brief.pdf", 7)

    def test_locate_quote_returns_global_offsets(self):
        """Debug metadata only, but it must still index the document buffer, not the chunk."""
        span = locate_quote(chunk(), QUOTE)
        assert span is not None
        assert PASSAGE[span[0] - 500 : span[1] - 500] == QUOTE


class TestVerifiedPresence:
    """Confirmed by a string search, so it needs no human signature."""

    def test_presence_is_extracted_det_and_auto_asserts(self):
        [a] = extractor().validate([entity()], chunk=chunk())
        assert a.epistemic_class is EpistemicClass.EXTRACTED_DET
        assert a.review_state is ReviewState.AUTO_ASSERTED

    def test_predicate_is_mentions(self):
        [a] = extractor().validate([entity()], chunk=chunk())
        assert a.predicate == "MENTIONS"
        assert a.predicate in PRESENCE_PREDICATES

    def test_method_records_both_the_model_and_the_check(self):
        """The proposal was probabilistic; the confirmation was not. Both belong in the
        audit trail, and the check is versioned so it can be superseded."""
        [a] = extractor().validate([entity()], chunk=chunk())
        assert a.method.startswith("llm:")
        assert a.method.endswith("+verify:quote@v1")

    def test_confidence_is_not_certainty(self):
        """The search proved the text is there; the entity id is still the model's work."""
        [a] = extractor().validate([entity()], chunk=chunk())
        assert a.confidence == VERIFIED_PRESENCE_CONFIDENCE
        assert a.confidence < 1.0

    def test_self_reported_confidence_is_ignored(self):
        """The check's confidence, not the model's opinion of itself."""
        claim = ProposedClaim(
            subject_id="", predicate="MENTIONS", object_id="party:acme", quote=QUOTE, confidence=0.1
        )
        [a] = extractor().validate([claim], chunk=chunk())
        assert a.confidence == VERIFIED_PRESENCE_CONFIDENCE

    def test_subject_defaults_to_the_document(self):
        [a] = extractor().validate([entity()], chunk=chunk())
        assert a.subject_id == "document:doc-1"

    def test_a_significance_predicate_never_auto_asserts(self):
        """Filing CITES under entities must not buy it the deterministic class."""
        [a] = extractor().validate(
            [ProposedClaim(subject_id="", predicate="CITES", object_id="authority:x", quote=QUOTE)],
            chunk=chunk(),
        )
        assert a.epistemic_class is EpistemicClass.EXTRACTED_MODEL
        assert a.review_state is ReviewState.PENDING


class TestInterpretation:
    """Nothing can mechanically confirm it, so a human must."""

    def test_interpretive_claim_is_extracted_model_and_pending(self):
        [a] = extractor().validate([interpretation()], chunk=chunk())
        assert a.epistemic_class is EpistemicClass.EXTRACTED_MODEL
        assert a.review_state is ReviewState.PENDING

    def test_confidence_is_capped_below_the_retrieval_floor(self):
        """A model's stated certainty is not evidence."""
        [a] = extractor().validate([interpretation(confidence=0.99)], chunk=chunk())
        assert a.confidence == MAX_MODEL_CONFIDENCE
        assert a.confidence < 0.8

    def test_a_modest_self_report_is_not_inflated(self):
        [a] = extractor().validate([interpretation(confidence=0.3)], chunk=chunk())
        assert a.confidence == 0.3

    def test_negative_confidence_is_clamped(self):
        [a] = extractor().validate([interpretation(confidence=-1.0)], chunk=chunk())
        assert a.confidence == 0.0

    def test_method_carries_no_verification_claim(self):
        [a] = extractor().validate([interpretation()], chunk=chunk())
        assert "verify" not in a.method


class TestPredicateGrounding:
    def test_ungroundable_predicate_is_dropped(self):
        assert extractor().validate([interpretation(predicate="undercuts")], chunk=chunk()) == []

    def test_grounding_is_exact_not_semantic(self):
        """An LLM deciding `acts_for` means REPRESENTS is a schema decision by a model."""
        assert (
            extractor().validate([interpretation(predicate="acts_on_behalf_of")], chunk=chunk())
            == []
        )

    def test_label_form_grounds(self):
        assert (
            len(extractor().validate([interpretation(predicate="distinguishes")], chunk=chunk()))
            == 1
        )

    def test_governing_predicate_grounds_and_is_reviewed(self):
        [a] = extractor().validate([interpretation(predicate="OVERRULES")], chunk=chunk())
        assert a.predicate == "OVERRULES"
        assert a.review_state is ReviewState.PENDING


class TestPresenceContractIsEnforcedByBuildAssertion:
    """The backstop. Even if the extractor's routing were changed, the contract refuses:
    a quote-match proves text is present and nothing more. The old parser auto-asserted
    CITES, so "the court declined to follow Brown" was recorded as reliance on Brown."""

    @pytest.mark.parametrize("predicate", ["CITES", "ADVERSE_TO", "FILED_IN", "SUPERSEDES"])
    def test_non_mentions_presence_claim_is_refused(self, predicate):
        with pytest.raises(AssertionError_, match="only that text appears"):
            build_assertion(
                tenant_id=TENANT,
                subject_id="document:doc-1",
                predicate=predicate,
                object_id="authority:410-us-113",
                epistemic_class=EpistemicClass.EXTRACTED_DET,
                method="llm:opus-5+verify:quote@v1",
                confidence=VERIFIED_PRESENCE_CONFIDENCE,
                source_locator=chunk().to_locator(500, 500 + len(QUOTE)),
            )

    def test_mentions_is_permitted(self):
        a = build_assertion(
            tenant_id=TENANT,
            subject_id="document:doc-1",
            predicate="MENTIONS",
            object_id="party:acme-corporation",
            epistemic_class=EpistemicClass.EXTRACTED_DET,
            method="llm:opus-5+verify:quote@v1",
            confidence=VERIFIED_PRESENCE_CONFIDENCE,
            source_locator=chunk().to_locator(500, 500 + len(QUOTE)),
        )
        assert a.review_state is ReviewState.AUTO_ASSERTED


class TestScopeAndIdentity:
    def test_tenant_and_matter_come_from_the_chunk(self):
        [a] = extractor().validate([interpretation()], chunk=chunk(matter_id="matter-1"))
        assert (a.tenant_id, a.matter_id) == (TENANT, "matter-1")

    def test_self_referential_edge_is_dropped(self):
        assert (
            extractor().validate([interpretation(object_id="document:doc-1")], chunk=chunk()) == []
        )

    def test_duplicates_collapse(self):
        found = extractor().validate([interpretation(), interpretation()], chunk=chunk())
        assert len(found) == 1

    def test_re_extraction_is_idempotent(self):
        """Content-addressed ids, so a re-run converges rather than accumulating."""
        first = extractor().validate([entity(), interpretation()], chunk=chunk())
        second = extractor().validate([entity(), interpretation()], chunk=chunk())
        assert {a.assertion_id for a in first} == {a.assertion_id for a in second}


class TestPrompt:
    def test_no_allowlist_of_courts_or_reporters_is_supplied(self):
        """The allowlist failure is the reason the parser was deleted: a court nobody
        enumerated was silently absent from the graph."""
        payload = json.loads(
            build_prompt(chunk(), allowed_predicates=["MENTIONS"], entity_kinds=[])
        )
        assert set(payload) == {"page", "passage", "allowed_predicates", "entity_kinds"}

    def test_passage_and_page_are_sent(self):
        payload = json.loads(
            build_prompt(chunk(page=7), allowed_predicates=[], entity_kinds=["Party"])
        )
        assert payload["passage"] == PASSAGE
        assert payload["page"] == 7

    def test_closed_vocabulary_is_listed(self):
        bedrock = FakeBedrock('{"entities": [], "relationships": []}')
        ModelExtractor(ONTOLOGY, bedrock=bedrock).extract(chunk())
        sent = json.loads(bedrock.requests[0]["body"]["messages"][0]["content"])
        assert "ADVERSE_TO" in sent["allowed_predicates"]
        assert "MENTIONS" in sent["allowed_predicates"]

    def test_temperature_is_omitted_by_default(self):
        """The newest Claude models reject the parameter, and a ValidationException on
        every chunk is a worse outcome than unpinned decoding."""
        bedrock = FakeBedrock('{"entities": [], "relationships": []}')
        ModelExtractor(ONTOLOGY, bedrock=bedrock).extract(chunk())
        assert "temperature" not in bedrock.requests[0]["body"]

    def test_temperature_is_sent_when_set(self):
        bedrock = FakeBedrock('{"entities": [], "relationships": []}')
        ModelExtractor(ONTOLOGY, bedrock=bedrock, temperature=0.0).extract(chunk())
        assert bedrock.requests[0]["body"]["temperature"] == 0.0

    def test_configured_model_is_used(self):
        bedrock = FakeBedrock('{"entities": []}')
        ModelExtractor(
            ONTOLOGY, bedrock=bedrock, model_id="global.anthropic.claude-opus-5"
        ).extract(chunk())
        assert bedrock.requests[0]["modelId"] == "global.anthropic.claude-opus-5"


class TestResponseParsing:
    def test_both_claim_kinds_are_read(self):
        payload = json.dumps(
            {
                "entities": [{"id": "party:acme", "kind": "Party", "quote": "Acme"}],
                "relationships": [
                    {
                        "subject_id": "document:doc-1",
                        "predicate": "DISTINGUISHES",
                        "object_id": "authority:410-us-113",
                        "quote": "undercuts",
                        "confidence": 0.6,
                    }
                ],
            }
        )
        claims = parse_response(payload)
        assert [c.predicate for c in claims] == ["MENTIONS", "DISTINGUISHES"]

    def test_entities_get_the_document_as_subject_later(self):
        [claim] = parse_response(json.dumps({"entities": [{"id": "party:acme", "quote": "Acme"}]}))
        assert claim.subject_id == ""

    def test_tolerates_prose_around_the_json(self):
        assert parse_response('Here you go:\n{"entities": []}\nHope that helps.') == []

    def test_missing_json_raises(self):
        with pytest.raises(ModelExtractionFailed, match="no JSON"):
            parse_response("I could not find anything.")

    def test_malformed_json_raises(self):
        with pytest.raises(ModelExtractionFailed, match="not valid JSON"):
            parse_response('{"entities": [oops}')

    def test_malformed_claim_is_dropped_not_partially_parsed(self):
        payload = json.dumps(
            {
                "relationships": [
                    {"predicate": "DISTINGUISHES"},
                    {
                        "subject_id": "a",
                        "predicate": "MENTIONS",
                        "object_id": "b",
                        "quote": "x",
                    },
                ]
            }
        )
        assert len(parse_response(payload)) == 1

    def test_absent_sections_are_not_an_error(self):
        assert parse_response('{"entities": null, "relationships": null}') == []


class TestEndToEnd:
    RESPONSE = json.dumps(
        {
            "entities": [
                {"id": "party:acme-corporation", "kind": "Party", "quote": "Roe v. Wade"},
                {"id": "party:invented", "kind": "Party", "quote": "never appears here"},
            ],
            "relationships": [
                {
                    "subject_id": "document:doc-1",
                    "predicate": "DISTINGUISHES",
                    "object_id": "authority:410-us-113",
                    "quote": QUOTE,
                    "confidence": 0.95,
                },
                {
                    "subject_id": "document:doc-1",
                    "predicate": "undercuts",
                    "object_id": "authority:410-us-113",
                    "quote": QUOTE,
                    "confidence": 0.9,
                },
            ],
        }
    )

    def test_one_call_yields_both_classes_and_drops_the_rest(self):
        found = ModelExtractor(ONTOLOGY, bedrock=FakeBedrock(self.RESPONSE)).extract(chunk())
        by_class = {a.epistemic_class: a for a in found}
        assert set(by_class) == {EpistemicClass.EXTRACTED_DET, EpistemicClass.EXTRACTED_MODEL}
        assert by_class[EpistemicClass.EXTRACTED_DET].object_id == "party:acme-corporation"
        assert by_class[EpistemicClass.EXTRACTED_MODEL].confidence == MAX_MODEL_CONFIDENCE

    def test_only_the_interpretation_reaches_the_review_queue(self):
        found = ModelExtractor(ONTOLOGY, bedrock=FakeBedrock(self.RESPONSE)).extract(chunk())
        pending = [a for a in found if a.review_state is ReviewState.PENDING]
        assert [a.predicate for a in pending] == ["DISTINGUISHES"]

    def test_one_call_per_chunk(self):
        bedrock = FakeBedrock(self.RESPONSE)
        chunks = [chunk(start=500), chunk(page=2, start=2000)]
        ModelExtractor(ONTOLOGY, bedrock=bedrock).extract_document(chunks)
        assert len(bedrock.requests) == 2

    def test_document_extraction_collapses_across_chunks(self):
        """Overlapping chunks re-quote the same sentence; the graph must not double it."""
        bedrock = FakeBedrock(self.RESPONSE)
        found = ModelExtractor(ONTOLOGY, bedrock=bedrock).extract_document([chunk(), chunk()])
        assert len({a.assertion_id for a in found}) == len(found)

    def test_bedrock_failure_is_wrapped(self):
        class Broken:
            def invoke_model(self, **kw):
                raise RuntimeError("throttled")

        with pytest.raises(ModelExtractionFailed, match="bedrock invoke failed"):
            ModelExtractor(ONTOLOGY, bedrock=Broken()).extract(chunk())
