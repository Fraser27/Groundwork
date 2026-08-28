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

import json

import pytest

from src.documents.extractors.model import (
    DESCRIPTIVE_CONFIDENCE,
    MAX_MODEL_CONFIDENCE,
    SYSTEM_PROMPT,
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
from src.ontology.loader import available_domains, load_ontology

ALL_PACKS = available_domains()

# The id prefix each pack's organising unit mints, written out rather than derived.
UNIT_PREFIXES = {
    "fintech": "facility:U-1",
    "healthcare": "encounter:U-1",
    "legal": "matter:U-1",
    "retail": "case:U-1",
}

TENANT = "firm-acme"
ONTOLOGY = load_ontology("legal")

PASSAGE = (
    "The court in Roe v. Wade, 410 U.S. 113 (1973), established the framework. "
    "That holding undercuts the defendant's argument that no such right exists. "
    "The 2019 agreement supersedes the earlier memorandum of understanding."
)

QUOTE = "That holding undercuts the defendant's argument"


class FakeBedrock:
    """Returns a canned Converse response. Records what it was asked."""

    def __init__(self, payload) -> None:
        self.payload = payload
        self.requests: list[dict] = []

    def converse(self, **kw):
        self.requests.append({"modelId": kw["modelId"], "request": kw})
        return {"output": {"message": {"content": [{"text": self.payload}]}}}


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

    def test_a_descriptive_claim_lands_on_the_floor_not_above_it(self):
        """MENTIONS is certain about something nearly worthless -- the string is on the page.

        It was written at 0.95, above every reviewed fact in the graph, so retrieval filled
        with "this document contains this name" and dropped the approved ADVERSE_TO a conflict
        check reads. Descriptive claims sit on the floor: answerable, never ahead.
        """
        [a] = extractor().validate([entity()], chunk=chunk())
        assert a.confidence == DESCRIPTIVE_CONFIDENCE
        assert a.confidence < VERIFIED_PRESENCE_CONFIDENCE

    def test_self_reported_confidence_is_ignored(self):
        """The check's confidence, not the model's opinion of itself."""
        claim = ProposedClaim(
            subject_id="", predicate="MENTIONS", object_id="party:acme", quote=QUOTE, confidence=0.1
        )
        [a] = extractor().validate([claim], chunk=chunk())
        assert a.confidence == DESCRIPTIVE_CONFIDENCE
        assert a.raw_confidence == 0.1

    def test_the_score_is_keyed_off_the_pack_not_a_predicate_list(self):
        """A healthcare pack must get this without a code change, so the split has to come
        from `descriptive_predicates` rather than a hardcoded set of names."""
        from src.ontology.loader import load_ontology

        for domain in ALL_PACKS:
            ont = load_ontology(domain)
            assert "MENTIONS" in ont.descriptive_predicates, domain

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
        """Subject is an authority, not the document: `OVERRULES` is declared Authority ->
        Authority, and only a case can overrule a case. The fixture's default document subject is
        right for `DISTINGUISHES`, which a document may do."""
        claim = interpretation(predicate="OVERRULES", subject_id="authority:the-marisol-2025")
        [a] = extractor().validate([claim], chunk=chunk())
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
            build_prompt(
                chunk(), predicates={"MENTIONS": ONTOLOGY.predicates["MENTIONS"]}, entities={}
            )
        )
        assert set(payload) == {
            "page",
            "passage",
            "allowed_predicates",
            "entity_kinds",
            # The matter this document belongs to. Not an allowlist -- it is one id the chunk
            # already carries, and without it the model has no way to say "our engagement is
            # against them", which is the fact a conflict check reads.
            "this_matter",
        }

    def test_passage_and_page_are_sent(self):
        payload = json.loads(
            build_prompt(
                chunk(page=7), predicates={}, entities={"Party": ONTOLOGY.entities["Party"]}
            )
        )
        assert payload["passage"] == PASSAGE
        assert payload["page"] == 7

    def test_closed_vocabulary_is_listed(self):
        bedrock = FakeBedrock('{"entities": [], "relationships": []}')
        ModelExtractor(ONTOLOGY, bedrock=bedrock).extract(chunk())
        sent = json.loads(bedrock.requests[0]["request"]["messages"][0]["content"][0]["text"])
        listed = {p["predicate"] for p in sent["allowed_predicates"]}
        assert "ADVERSE_TO" in listed
        assert "MENTIONS" in listed

    def test_a_predicate_carries_its_declared_ends(self):
        """The vocabulary being closed is only half of what a write has to satisfy. `ADVERSE_TO` is
        Matter -> Party and a party-to-party claim is refused, so sending the name without the shape
        asked the model to guess an orientation and then dropped the claim when it guessed wrong --
        which starved the conflict rule while looking like a document with nothing to say."""
        bedrock = FakeBedrock('{"entities": [], "relationships": []}')
        ModelExtractor(ONTOLOGY, bedrock=bedrock).extract(chunk())
        sent = json.loads(bedrock.requests[0]["request"]["messages"][0]["content"][0]["text"])
        adverse = next(p for p in sent["allowed_predicates"] if p["predicate"] == "ADVERSE_TO")
        assert adverse["subject_kinds"] == ["Matter"]
        assert adverse["object_kinds"] == ["Party"]

    @pytest.mark.parametrize("domain", sorted(ALL_PACKS))
    def test_a_terms_help_reaches_the_model_and_not_only_its_description(self, domain):
        """The measured failure, and the reason the help matters more than the description.
        `Company` and `Merchant` went out as two bare words, so a non-trading holding company was
        typed as a seller and `related_party_resale`, whose chain must end at a Merchant, drew
        nothing from a memo stating the whole ownership ladder. `Company.description` alone does
        not rule that out; the help does, in as many words.

        Asserted against whatever the pack says rather than a copy of its wording. A test that
        quotes the text goes red when somebody improves the text, and it also pins one pack's
        vocabulary into a module that is supposed to have none.
        """
        ontology = load_ontology(domain)
        bedrock = FakeBedrock('{"entities": [], "relationships": []}')
        ModelExtractor(ontology, bedrock=bedrock).extract(chunk())
        sent = json.loads(bedrock.requests[0]["request"]["messages"][0]["content"][0]["text"])

        for offered, declared in (
            ({k["kind"]: k["meaning"] for k in sent["entity_kinds"]}, ontology.entities),
            (
                {p["predicate"]: p["meaning"] for p in sent["allowed_predicates"]},
                ontology.predicates,
            ),
        ):
            for name, meaning in offered.items():
                defn = declared[name]
                assert defn.description in meaning
                # Absent help must leave no trailing separator and no empty meaning.
                assert defn.help is None or defn.help in meaning
                assert meaning == meaning.strip()

    def test_every_offered_term_has_a_meaning(self):
        """A term with an empty meaning is the case this change exists to remove, and a pack is
        free to omit a description. Better caught here than by an extraction that quietly picked
        the wrong one of two words."""
        for domain in ALL_PACKS:
            bedrock = FakeBedrock('{"entities": [], "relationships": []}')
            ModelExtractor(load_ontology(domain), bedrock=bedrock).extract(chunk())
            sent = json.loads(bedrock.requests[0]["request"]["messages"][0]["content"][0]["text"])
            bare = [p["predicate"] for p in sent["allowed_predicates"] if not p["meaning"]] + [
                k["kind"] for k in sent["entity_kinds"] if not k["meaning"]
            ]
            assert bare == [], f"{domain} offers {bare} with no definition"

    def test_the_system_prompt_names_no_packs_vocabulary(self):
        """It once opened "you read legal documents" and worked its only orientation example
        through `ADVERSE_TO`. The pack is admin-selectable, so on a retail document that named a
        predicate the model could not use and described a domain it was not reading."""
        assert "legal" not in SYSTEM_PROMPT.lower()
        for domain in ALL_PACKS:
            for predicate in load_ontology(domain).predicates:
                assert predicate not in SYSTEM_PROMPT, f"{domain}'s {predicate} is hardcoded"

    def test_the_chunks_own_matter_is_offered_as_a_subject(self):
        """`ADVERSE_TO` needs a Matter subject, and the only matter id in scope is the one the
        document was filed under. A descriptive predicate carries no shape, so it is sent bare.

        The filing reference keeps its case: `Matter` declares `external_id`, so
        `canonical_entity_id` leaves the local part alone and a lowercased id would not join."""
        bedrock = FakeBedrock('{"entities": [], "relationships": []}')
        ModelExtractor(ONTOLOGY, bedrock=bedrock).extract(chunk(matter_id="NTL-2026-0114"))
        sent = json.loads(bedrock.requests[0]["request"]["messages"][0]["content"][0]["text"])
        assert sent["this_matter"] == "matter:NTL-2026-0114"
        mentions = next(p for p in sent["allowed_predicates"] if p["predicate"] == "MENTIONS")
        assert "subject_kinds" not in mentions

    def test_the_offered_matter_is_an_entity_id_not_a_bare_filing_reference(self):
        """A `Chunk.matter_id` is a filing reference (`NTL`), not an entity id. Offering it bare
        made the model echo it back as an `ADVERSE_TO` subject, and `validate` then dropped the
        claim for an unknown entity kind -- so the conflict rule stayed starved while the document
        looked like it had nothing to say. Every other id in the prompt is `kind:local`."""
        bedrock = FakeBedrock('{"entities": [], "relationships": []}')
        ModelExtractor(ONTOLOGY, bedrock=bedrock).extract(chunk(matter_id="NTL"))
        sent = json.loads(bedrock.requests[0]["request"]["messages"][0]["content"][0]["text"])
        assert sent["this_matter"] == "matter:NTL"

    def test_no_matter_is_offered_when_the_chunk_is_unfiled(self):
        """`null` is the documented signal to omit matter claims. `matter:None` would be a
        well-formed id for a matter that does not exist."""
        bedrock = FakeBedrock('{"entities": [], "relationships": []}')
        ModelExtractor(ONTOLOGY, bedrock=bedrock).extract(chunk())
        sent = json.loads(bedrock.requests[0]["request"]["messages"][0]["content"][0]["text"])
        assert sent["this_matter"] is None

    def test_every_shipped_pack_has_a_row_below(self):
        """The table is written out by hand so the expectation is independent of the slug the
        code derives it from. That only holds if a new pack is added to it, and a pack with no
        row is a pack whose prefix nothing checks."""
        assert set(UNIT_PREFIXES) == set(ALL_PACKS)

    @pytest.mark.parametrize(("domain", "expected"), sorted(UNIT_PREFIXES.items()))
    def test_the_offered_unit_takes_the_prefix_its_own_pack_declares(self, domain, expected):
        """`matter:` is the legal pack's kind. Hardcoding it minted an id of a kind no other pack
        declares, so `entity_kind_of` returned None and `validate` dropped every claim about the
        encounter or the facility -- the same silent drop that starved the conflict rules, one pack
        over. The scoping key stays `matter_id`; the prefix follows the pack."""
        bedrock = FakeBedrock('{"entities": [], "relationships": []}')
        onto = load_ontology(domain)
        ModelExtractor(onto, bedrock=bedrock).extract(chunk(matter_id="U-1"))
        sent = json.loads(bedrock.requests[0]["request"]["messages"][0]["content"][0]["text"])
        assert sent["this_matter"] == expected
        # And the id it offers is one that pack will actually accept at the boundary.
        assert onto.entity_kind_of(expected) is not None

    def test_temperature_is_omitted_by_default(self):
        """The newest Claude models reject the parameter, and a ValidationException on
        every chunk is a worse outcome than unpinned decoding."""
        bedrock = FakeBedrock('{"entities": [], "relationships": []}')
        ModelExtractor(ONTOLOGY, bedrock=bedrock).extract(chunk())
        assert "temperature" not in bedrock.requests[0]["request"]["inferenceConfig"]

    def test_temperature_is_sent_when_set(self):
        bedrock = FakeBedrock('{"entities": [], "relationships": []}')
        ModelExtractor(ONTOLOGY, bedrock=bedrock, temperature=0.0).extract(chunk())
        assert bedrock.requests[0]["request"]["inferenceConfig"]["temperature"] == 0.0

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
            def converse(self, **kw):
                raise RuntimeError("throttled")

        with pytest.raises(ModelExtractionFailed, match="bedrock invoke failed"):
            ModelExtractor(ONTOLOGY, bedrock=Broken()).extract(chunk())


class TestEntityKindsAreClosed:
    """A model may not invent an entity kind, for the same reason it may not invent a predicate.

    `vessel:mv-aurelia` and `ship:mv-aurelia` are two nodes, and a traversal finds one of them. A
    conflict check that misses half its edges because two extractions disagreed on a noun looks
    exactly like a clean conflict check -- the failure the closed vocabulary exists to prevent.

    Structural rather than descriptive, which is why this is closed while a new subject-matter tag
    is not: an entity kind decides which node a fact hangs off, so adding one is a pack decision.
    """

    def test_an_invented_kind_is_dropped(self):
        claim = interpretation(
            subject_id="vessel:mv-aurelia",
            predicate="ADVERSE_TO",
            object_id="party:acme-corporation",
        )
        assert extractor().validate([claim], chunk=chunk()) == []

    def test_a_declared_kind_survives(self):
        """`ADVERSE_TO` is Matter -> Party: this engagement of ours is against them."""
        claim = interpretation(
            subject_id="matter:ntl-2026-0114",
            predicate="ADVERSE_TO",
            object_id="party:calder-shipping-ag",
        )
        out = extractor().validate([claim], chunk=chunk())
        assert len(out) == 1
        assert (out[0].subject_id, out[0].object_id) == (
            "matter:ntl-2026-0114",
            "party:calder-shipping-ag",
        )

    def test_a_party_to_party_adversity_is_refused(self):
        """The claim that starved the conflict check, now rejected at the boundary.

        "Calder is adverse to Northwind" does not say who the firm acts for, and while the pack
        allowed it *and* declared the predicate symmetric, `canonical_pair` byte-sorted the ends --
        so a rule asking "are we against a party we also represent" bound its matter variable to
        whichever party sorted first and flagged the firm's own client. Refused loudly here rather
        than silently matching nothing later."""
        claim = interpretation(
            subject_id="party:calder-shipping-ag",
            predicate="ADVERSE_TO",
            object_id="party:northwind-trading-limited",
        )
        assert extractor().validate([claim], chunk=chunk()) == []

    def test_an_unprefixed_id_is_dropped(self):
        """A bare id cannot be placed in the vocabulary, and guessing its kind would defeat the
        point of having one."""
        claim = interpretation(
            subject_id="mv-aurelia", predicate="ADVERSE_TO", object_id="party:acme-corporation"
        )
        assert extractor().validate([claim], chunk=chunk()) == []

    def test_the_object_is_checked_too(self):
        """Both endpoints, not just the subject: an edge to an invented node is as unqueryable as
        one from it."""
        claim = interpretation(
            subject_id="party:acme-corporation",
            predicate="ADVERSE_TO",
            object_id="vessel:mv-aurelia",
        )
        assert extractor().validate([claim], chunk=chunk()) == []

    def test_the_prompt_names_the_allowed_kinds(self):
        """The model can only comply if told. Refusing at the boundary without asking first would
        discard work the model would have got right."""
        ex = extractor()
        ex.extract(chunk())
        request = ex.bedrock.requests[0]["request"]
        sent = json.dumps(request)
        assert "entity_kinds" in sent
        assert "party" in sent
        assert "invent" in request["system"][0]["text"]

    def test_a_catalog_kind_is_declared_rather_than_special_cased(self):
        """The Glue scanner mints `table:`, `column:` and `source:`. They were in the graph while no
        pack named them, so the scanner was held to a looser rule than the extractor."""
        assert {"table", "column", "source"} <= ONTOLOGY.entity_kinds


class TestEntityIdsAreCanonicalised:
    """One company, one node -- the other half of the failure the closed kind vocabulary covers.

    The kind was closed and the *name* was not, so `party:calder-shipping-ag`,
    `Party:calder-shipping-ag` and `party: Calder Shipping AG` all cleared the kind guard (which
    lowercases the prefix before comparing) and MERGEd as three nodes. A conflict check joins on
    the shared node, so the fork returns nothing and reads exactly like a clean check.
    """

    @pytest.mark.parametrize(
        "written",
        [
            "Party:calder-shipping-ag",
            "party: Calder Shipping AG",
            "party:calder_shipping_ag",
            "PARTY:CALDER-SHIPPING-AG",
            "party:calder--shipping--ag",
        ],
    )
    def test_every_spelling_lands_on_one_node(self, written):
        """And on one *assertion id*: `subject_id` is in the content hash, so a variant spelling
        forks the provenance record as well as the node."""
        [canonical] = extractor().validate([entity(object_id="party:calder-shipping-ag")], chunk=chunk())
        [variant] = extractor().validate([entity(object_id=written)], chunk=chunk())

        assert variant.object_id == "party:calder-shipping-ag"
        assert variant.assertion_id == canonical.assertion_id

    def test_a_matter_id_keeps_its_case(self):
        """`matter:NTL-2026-0114` is a case-management reference, and `GraphExplorer` rebuilds it
        from the uncased property. This is the test that stops the normaliser later being
        "simplified" into a blanket lowercase or into `discovery.enrichment._slugify`."""
        claim = interpretation(
            subject_id="document:doc-1",
            predicate="RELATES_TO_MATTER",
            object_id="matter:NTL-2026-0114",
        )
        [out] = extractor().validate([claim], chunk=chunk())
        assert out.object_id == "matter:NTL-2026-0114"

    @pytest.mark.parametrize(
        "external",
        [
            "matter:NTL-2026-0114",
            "table:src-1:legal.matters",
            "column:src-1:legal.matters.client_id",
            "document:doc-d63a8228d513541553a76672",
        ],
    )
    def test_an_external_id_is_preserved_byte_for_byte(self, external):
        """Each of these is some other system's name. Normalising one breaks the join back to it,
        and `table:src-1:legal.matters` would collapse to a single flat run."""
        assert ONTOLOGY.canonical_entity_id(external) == external

    def test_a_party_is_normalised_because_we_mint_it(self):
        """The distinction that makes this kind-aware rather than blanket: a Party's id comes from
        words on a page, so it is ours to canonicalise."""
        assert ONTOLOGY.canonical_entity_id("party: Calder Shipping AG") == (
            "party:calder-shipping-ag"
        )

    def test_a_suffix_is_never_stripped(self):
        """The crux. `Calder Shipping AG` and `Calder Shipping Ltd` are routinely a parent and its
        subsidiary -- which is what `AFFILIATE_OF` records and what `conflict_via_affiliate` needs
        both nodes to fire on. Collapsing them would turn an affiliate conflict into a direct one:
        a false positive on a live matter. Normalisation runs before `_compute_id`, so a wrong
        merge leaves no record that two names were ever collapsed."""
        assert ONTOLOGY.canonical_entity_id("party:calder-shipping-ag") != (
            ONTOLOGY.canonical_entity_id("party:calder-shipping-ltd")
        )

    def test_an_undeclared_kind_is_still_refused(self):
        """Normalisation must not launder an invented kind into a well-formed-looking id. The kind
        guard stays the thing that rejects."""
        assert ONTOLOGY.canonical_entity_id("vessel:mv-aurelia") is None
        assert ONTOLOGY.canonical_entity_id("meridian-holdings") is None

    def test_an_id_that_is_only_punctuation_is_refused(self):
        assert ONTOLOGY.canonical_entity_id("party:---") is None

    def test_a_non_ascii_name_survives(self):
        """A German or French party is ordinary in this domain, so transliterating would be a
        guess. Only case and punctuation are touched."""
        assert ONTOLOGY.canonical_entity_id("party:Müller Schiffahrt GmbH") == (
            "party:müller-schiffahrt-gmbh"
        )

    def test_a_variant_spelling_is_a_self_edge(self):
        """Normalisation runs before the self-edge check on purpose: relating `party:acme` to
        `Party:Acme` is one node related to itself, which the raw comparison would miss.

        On `AFFILIATE_OF` rather than `ADVERSE_TO`, because that one is Matter -> Party now: a
        party-to-party claim would be refused by the kind guard first and the test would pass
        without ever reaching the check it is named after."""
        claim = interpretation(
            subject_id="party:acme-corporation",
            predicate="AFFILIATE_OF",
            object_id="Party:Acme Corporation",
        )
        assert extractor().validate([claim], chunk=chunk()) == []

    def test_a_symmetric_fact_hashes_the_same_either_way(self):
        """`canonical_pair` sorts endpoints by their raw bytes, so before normalisation one
        symmetric fact spelled two ways produced two orderings and two content hashes.

        Healthcare, because the legal pack no longer declares a symmetric predicate -- `ADVERSE_TO`
        gave up `symmetric: true` when its ends became different kinds. `SAME_INGREDIENT_AS` is
        Medication -> Medication, so its ends genuinely are interchangeable."""
        onto = load_ontology("healthcare")
        med = ModelExtractor(onto, bedrock=FakeBedrock("{}"))
        lower = interpretation(
            subject_id="medication:zestril",
            predicate="SAME_INGREDIENT_AS",
            object_id="medication:prinivil",
        )
        mixed = interpretation(
            subject_id="Medication:Zestril",
            predicate="SAME_INGREDIENT_AS",
            object_id="medication:prinivil",
        )
        [a] = med.validate([lower], chunk=chunk())
        [b] = med.validate([mixed], chunk=chunk())
        assert a.assertion_id == b.assertion_id


class TestACatalogKindMustBeExternal:
    """A catalogued id is always the issuing system's name, so the two flags have to agree."""

    def test_the_shipped_packs_agree(self):
        for domain in ALL_PACKS:
            onto = load_ontology(domain)
            for e in onto.entities.values():
                if e.layer == "catalog":
                    assert e.external_id, f"{domain}: {e.id} is catalog but not external_id"

    def test_a_catalog_kind_that_would_be_rewritten_fails_the_pack(self, tmp_path):
        """Loud at load. A Glue table id quietly lowercased would break the join to Glue, and
        nothing downstream could tell that from a table nobody scanned."""
        from src.ontology import loader

        pack = tmp_path / "bad.yaml"
        pack.write_text(
            "domain: bad\nversion: 1\n"
            "entity_types:\n"
            "  - id: Table\n    label: Table\n    description: t\n    layer: catalog\n"
            "governing_predicates: []\ndescriptive_predicates: []\nrules: []\n"
        )
        with pytest.raises(ValueError, match="external_id"):
            loader._parse(__import__("yaml").safe_load(pack.read_text()))


class TestEntityLayers:
    """Which half of the graph a node belongs to, so it can be read one half at a time."""

    def test_a_legal_entity_is_domain(self):
        assert ONTOLOGY.layer_of("party:acme-corporation") == "domain"

    def test_a_catalogued_schema_node_is_catalog(self):
        assert ONTOLOGY.layer_of("column:glue-main:db.tbl.col") == "catalog"

    def test_an_unknown_kind_is_not_quietly_filed_as_domain(self):
        """`unknown` rather than a default. A filter that files an unrecognised node under `domain`
        hides the drift the vocabulary exists to surface."""
        assert ONTOLOGY.layer_of("vessel:mv-aurelia") == "unknown"
