"""Tests for the assertion contract.

These are the invariants the whole design rests on. If one of these ever goes red,
the graph has stopped being defensible — so they are worth more than their line
count suggests.
"""

from __future__ import annotations

import pytest

from src.graph.assertions import (
    AUTO_ASSERT_CLASSES,
    AssertionError_,
    EndpointKinds,
    EpistemicClass,
    ReviewState,
    SourceLocator,
    build_assertion,
)

DOC = SourceLocator(
    document_id="doc-1",
    filename="memorandum.pdf",
    page=3,
    chunk_id="doc-1:c2",
    quote="within two years of the death of the Settlor",
)
TBL = SourceLocator(source_id="src-1", table="matters", column="client_id")


def _base(**over):
    """An interpretive model claim — the common case, and the one needing review."""
    kw = dict(
        tenant_id="firm-acme",
        subject_id="Matter-4471",
        predicate="REPRESENTS",
        object_id="Party-Acme",
        epistemic_class=EpistemicClass.EXTRACTED_MODEL,
        method="llm:opus-5",
        confidence=0.7,
        source_locator=DOC,
    )
    kw.update(over)
    return build_assertion(**kw)


def _verified(**over):
    """A quote-verified presence claim: EXTRACTED_DET may only ever be MENTIONS."""
    kw = dict(
        tenant_id="firm-acme",
        subject_id="document:doc-1",
        predicate="MENTIONS",
        object_id="party:acme-corporation",
        epistemic_class=EpistemicClass.EXTRACTED_DET,
        method="llm:opus-5+verify:quote@v1",
        confidence=0.95,
        source_locator=DOC,
    )
    kw.update(over)
    return build_assertion(**kw)


class TestMandatoryProvenance:
    def test_tenant_id_required(self):
        with pytest.raises(AssertionError_, match="tenant_id"):
            _base(tenant_id="")

    def test_method_required(self):
        with pytest.raises(AssertionError_, match="method"):
            _base(method="")

    def test_confidence_bounded(self):
        with pytest.raises(AssertionError_, match="confidence"):
            _base(confidence=1.5)

    def test_locator_needs_a_source(self):
        with pytest.raises(ValueError, match="document_id or source_id"):
            SourceLocator(page=1)


class TestAutoAssertPolicy:
    """DECLARED and EXTRACTED_DET auto-assert; models go to review."""

    def test_declared_auto_asserts(self):
        a = _base(
            epistemic_class=EpistemicClass.DECLARED,
            method="glue:catalog_scan",
            source_locator=TBL,
        )
        assert a.review_state is ReviewState.AUTO_ASSERTED

    def test_verified_presence_auto_asserts(self):
        """A model proposed it, but a string search confirmed it."""
        assert _verified().review_state is ReviewState.AUTO_ASSERTED

    def test_auto_assert_set_is_exactly_these_two(self):
        assert AUTO_ASSERT_CLASSES == {
            EpistemicClass.DECLARED,
            EpistemicClass.EXTRACTED_DET,
        }

    def test_model_extraction_needs_review(self):
        a = _base(epistemic_class=EpistemicClass.EXTRACTED_MODEL, method="llm:claude-sonnet-5")
        assert a.review_state is ReviewState.PENDING

    def test_caller_cannot_opt_out_of_review(self):
        """review_state is derived, never accepted from the caller."""
        with pytest.raises(TypeError):
            build_assertion(
                tenant_id="firm-acme",
                subject_id="a",
                predicate="MENTIONS",
                object_id="b",
                epistemic_class=EpistemicClass.EXTRACTED_MODEL,
                method="llm:x",
                confidence=0.9,
                source_locator=DOC,
                review_state=ReviewState.APPROVED,
            )


class TestVerifiedPresenceIsPresenceOnly:
    """The fix for a real bug: the old parser auto-asserted CITES, so "the court
    declined to follow Brown" was recorded as reliance on Brown. A quote-match proves
    text is present and nothing more."""

    def test_mentions_is_allowed(self):
        assert _verified(predicate="MENTIONS")

    @pytest.mark.parametrize("predicate", ["CITES", "ADVERSE_TO", "REPRESENTS", "OVERRULES"])
    def test_significance_predicates_refused(self, predicate):
        with pytest.raises(AssertionError_, match="only that text appears"):
            _verified(predicate=predicate)

    def test_quote_is_mandatory(self):
        """Without the quote, the confirmation cannot be repeated."""
        with pytest.raises(AssertionError_, match="requires the quote"):
            _verified(source_locator=SourceLocator(source_id="src-1", table="t"))

    def test_interpretation_may_use_governing_predicates(self):
        """The same predicate is fine as EXTRACTED_MODEL — it just needs review."""
        a = _base(predicate="CITES", epistemic_class=EpistemicClass.EXTRACTED_MODEL)
        assert a.review_state is ReviewState.PENDING


class TestDocumentCitationsAreCheckable:
    def test_document_locator_requires_a_quote(self):
        """A citation with nothing to search for cannot be verified by hand."""
        with pytest.raises(ValueError, match="requires a quote"):
            SourceLocator(document_id="doc-1", page=2)

    def test_structured_locator_needs_no_quote(self):
        assert SourceLocator(source_id="src-1", table="matters").is_document is False

    def test_offsets_are_optional(self):
        """Offsets index the extracted text buffer, not the PDF — debug only."""
        loc = SourceLocator(document_id="d", page=1, quote="hello")
        assert loc.char_start is None


class TestInferenceInvariants:
    def test_inferred_requires_premises(self):
        with pytest.raises(AssertionError_, match="require premises"):
            _base(
                epistemic_class=EpistemicClass.INFERRED,
                method="rule:conflict_check@v1",
                predicate="ADVERSE_TO",
            )

    def test_inference_cannot_exceed_weakest_premise(self):
        """A chain of guesses must not launder itself into a certainty."""
        with pytest.raises(AssertionError_, match="cannot manufacture certainty"):
            _base(
                epistemic_class=EpistemicClass.INFERRED,
                method="rule:conflict_check@v1",
                confidence=0.99,
                premises=("p1", "p2"),
                premise_confidences=(0.9, 0.6),
            )

    def test_inference_at_or_below_ceiling_is_fine(self):
        a = _base(
            epistemic_class=EpistemicClass.INFERRED,
            method="rule:conflict_check@v1",
            confidence=0.6,
            premises=("p1", "p2"),
            premise_confidences=(0.9, 0.6),
        )
        assert a.premises == ("p1", "p2")

    def test_premises_rejected_on_non_inferred(self):
        with pytest.raises(AssertionError_, match="only meaningful for INFERRED"):
            _base(premises=("p1",))


class TestClosedPredicateVocabulary:
    """The failure this prevents: a conflict check silently missing a synonym."""

    GOVERNING = frozenset({"REPRESENTS", "ADVERSE_TO", "CITES"})

    def test_approved_predicate_passes(self):
        assert _base(predicate="REPRESENTS", allowed_predicates=self.GOVERNING)

    def test_synonym_is_rejected_loudly(self):
        with pytest.raises(AssertionError_, match="closed vocabulary"):
            _base(predicate="is_counsel_to", allowed_predicates=self.GOVERNING)

    def test_descriptive_predicates_are_open(self):
        assert _base(predicate="CONCERNS_TOPIC", allowed_predicates=None)


class TestEntityIdsMustBeInNormalForm:
    """The same failure as the closed predicate vocabulary, one level down.

    `Ontology.canonical_entity_id` fixes an id at the extraction boundary. This is the belt
    behind it, in the shape of `assertion_queries._TYPE_SAFE`: it cannot normalise, because that
    needs the pack's kinds and this module deliberately does not import the ontology, but
    whitespace and a miscased prefix are syntactic and can be refused where nothing bypasses it.
    """

    @pytest.mark.parametrize(
        "bad",
        ["party: Calder Shipping AG", "party:calder shipping", " party:acme", "party:acme "],
    )
    def test_whitespace_in_an_id_is_refused(self, bad):
        with pytest.raises(AssertionError_, match="whitespace"):
            _base(object_id=bad)

    @pytest.mark.parametrize("bad", ["Party:calder", "PARTY:calder"])
    def test_a_miscased_kind_prefix_is_refused(self, bad):
        """`entity_kind_of` lowercases before checking the vocabulary, so this passes the kind
        guard and then becomes a second node for one entity."""
        with pytest.raises(AssertionError_, match="lowercase"):
            _base(object_id=bad)

    def test_the_subject_is_checked_too(self):
        with pytest.raises(AssertionError_, match="subject_id"):
            _base(subject_id="Party:Acme Corporation")

    @pytest.mark.parametrize(
        "good",
        [
            "matter:NTL-2026-0114",
            "table:src-1:legal.matters",
            "column:src-1:legal.matters.client_id",
            "document:doc-d63a8228d513541553a76672",
            "authority:410-us-113",
            "party:müller-schiffahrt-gmbh",
        ],
    )
    def test_an_external_id_is_not_refused(self, good):
        """Uppercase, dots and a second colon are all meaningful after the prefix: these are
        other systems' names. A guard that refused them would break the join back to Glue or to
        case management, which is why this stops at the prefix."""
        assert _base(object_id=good)

    def test_an_unprefixed_id_is_not_refused(self):
        """Requiring a prefix needs the vocabulary, which this module does not have. Callers
        legitimately write `Party-Acme`, and refusing it here would be a guess about kinds."""
        assert _base(object_id="Party-Acme")


class TestDeclaredEndpointsAreEnforced:
    """Invariant 4 for entity kinds rather than predicate names.

    The incident: `POTENTIAL_CONFLICT` declares `Matter -> Party`, a rule bound its subject to a
    Party, nothing checked, and the conflict was stored Party->Party. Since it declares
    `blocks: both`, the wall then withheld the firm's own client's file -- so a question about that
    client's own counsel returned nothing at all.

    Domain and range were declared in both packs from the beginning and read only for display.
    """

    def _kinds(self, subject, object_, *, symmetric=False):
        return EndpointKinds(
            subject=frozenset(subject), object=frozenset(object_), symmetric=symmetric
        )

    def test_a_subject_of_the_wrong_kind_is_refused(self):
        with pytest.raises(AssertionError_, match="cannot be written party -> party"):
            _base(
                predicate="POTENTIAL_CONFLICT",
                subject_id="party:calder-shipping-ag",
                object_id="party:halveston-chartering-limited",
                endpoint_kinds=self._kinds(["matter"], ["party"]),
            )

    def test_an_object_of_the_wrong_kind_is_refused(self):
        with pytest.raises(AssertionError_, match="cannot be written matter -> matter"):
            _base(
                predicate="RELATES_TO_MATTER",
                subject_id="matter:mbc-2024-0431",
                object_id="matter:ntl-2026-0114",
                endpoint_kinds=self._kinds(["document", "party", "deadline"], ["matter"]),
            )

    def test_the_declared_shape_is_accepted(self):
        assert _base(
            predicate="POTENTIAL_CONFLICT",
            subject_id="matter:ntl-2026-0114",
            object_id="party:calder-shipping-ag",
            endpoint_kinds=self._kinds(["matter"], ["party"]),
        )

    def test_a_predicate_declaring_two_subject_kinds_accepts_either(self):
        """`CITES` is `[Document, Authority] -> [Authority]`, and both subject kinds are sound. This
        is the test that stops the check being written as "subject must be domain[0]" -- which would
        refuse a real fact and, worse, look like the incident had been fixed."""
        kinds = self._kinds(["document", "authority"], ["authority"])
        assert _base(
            predicate="CITES",
            subject_id="document:doc-1",
            object_id="authority:410-us-113",
            endpoint_kinds=kinds,
        )
        assert _base(
            predicate="CITES",
            subject_id="authority:410-us-113",
            object_id="authority:347-us-483",
            endpoint_kinds=kinds,
        )

    def test_a_symmetric_predicate_accepts_the_reversed_orientation(self):
        """`canonical_pair` sorts a symmetric predicate's endpoints by raw bytes *before* this
        check, so for those the reversed order has to count as legal -- otherwise canonicalisation
        could turn a sound fact into a refused one depending on how two ids happen to sort."""
        kinds = self._kinds(["matter"], ["party"], symmetric=True)
        assert _base(
            predicate="SOMETHING_SYMMETRIC",
            subject_id="party:calder-shipping-ag",
            object_id="matter:ntl-2026-0114",
            endpoint_kinds=kinds,
            allowed_predicates=None,
        )

    def test_an_unprefixed_id_is_not_refused(self):
        """`_reject_unnormalised` deliberately allows `Matter-4471`, so there is no kind to check
        and narrowing that here would be an unrelated tightening."""
        assert _base(
            predicate="POTENTIAL_CONFLICT",
            subject_id="Matter-4471",
            object_id="Party-Acme",
            endpoint_kinds=self._kinds(["matter"], ["party"]),
        )

    def test_no_declaration_means_nothing_to_check(self):
        """Load-bearing rather than defensive: every descriptive predicate declares no endpoints,
        and the Glue scanner's `HAS_TABLE`/`HAS_COLUMN` are in no pack at all. Without this the
        catalog paths stop writing."""
        assert _base(predicate="MENTIONS", endpoint_kinds=None)

    def test_the_message_names_both_the_declaration_and_what_was_written(self):
        """An operator needs to know which end was wrong, not merely that something was."""
        with pytest.raises(AssertionError_) as caught:
            _base(
                predicate="POTENTIAL_CONFLICT",
                subject_id="party:calder-shipping-ag",
                object_id="party:halveston-chartering-limited",
                endpoint_kinds=self._kinds(["matter"], ["party"]),
            )
        message = str(caught.value)
        assert "['matter'] -> ['party']" in message
        assert "party -> party" in message



class TestTheLegalPackRefusesAPartyToPartyAdversity:
    """The orientation, checked against the shipped pack rather than a hand-built `EndpointKinds`.

    `ADVERSE_TO` used to declare `[Party, Matter] -> [Party]` *and* `symmetric: true`, so
    `canonical_pair` byte-sorted the ends and nothing downstream could tell the firm's client from
    the counterparty. `conflict_check` wants `(m:Matter)-[ADVERSE_TO]->(p:Party)` -- "this
    engagement of ours is against them" -- and a party-subject claim does not say who we act for.

    Reading the pack is the point. The tests above prove the *mechanism*; this proves the pack
    actually asks for it, so relaxing the YAML cannot pass while they stay green.
    """

    def test_party_to_party_is_refused(self):
        from src.ontology.loader import load_ontology

        onto = load_ontology("legal")
        with pytest.raises(AssertionError_, match="cannot be written party -> party"):
            _base(
                predicate="ADVERSE_TO",
                subject_id="party:calder-shipping-ag",
                object_id="party:halveston-chartering-limited",
                endpoint_kinds=onto.endpoint_kinds("ADVERSE_TO"),
            )

    def test_matter_to_party_is_accepted(self):
        from src.ontology.loader import load_ontology

        onto = load_ontology("legal")
        assert _base(
            predicate="ADVERSE_TO",
            subject_id="matter:NTL-2026-0114",
            object_id="party:calder-shipping-ag",
            endpoint_kinds=onto.endpoint_kinds("ADVERSE_TO"),
        )

    def test_the_reverse_is_refused_because_it_is_no_longer_symmetric(self):
        """`party -> matter` was admitted while the predicate was symmetric with unequal ends: a
        latent second orientation no validator caught. Dropping `symmetric:` closed it."""
        from src.ontology.loader import load_ontology

        onto = load_ontology("legal")
        with pytest.raises(AssertionError_, match="cannot be written party -> matter"):
            _base(
                predicate="ADVERSE_TO",
                subject_id="party:calder-shipping-ag",
                object_id="matter:NTL-2026-0114",
                endpoint_kinds=onto.endpoint_kinds("ADVERSE_TO"),
            )


class TestTheShippedPacksDeclareUsableEndpoints:
    def test_every_conclusion_a_rule_draws_is_declared(self):
        """A pack whose rule concludes a predicate it cannot legally write would fire and be
        refused on every match -- a conflict check that silently draws nothing."""
        from src.ontology.loader import load_ontology

        for domain in ("legal", "healthcare"):
            onto = load_ontology(domain)
            for predicate in onto.rule_conclusions:
                kinds = onto.endpoint_kinds(predicate)
                if kinds is None:
                    continue
                assert kinds.subject and kinds.object, f"{domain}: {predicate} declares an end"

    def test_the_conflict_predicate_is_matter_to_party(self):
        """The declaration the incident violated, pinned so a later pack edit that widened it to
        accept a Party subject would fail here rather than in production."""
        from src.ontology.loader import load_ontology

        kinds = load_ontology("legal").endpoint_kinds("POTENTIAL_CONFLICT")
        assert kinds.subject == frozenset({"matter"})
        assert kinds.object == frozenset({"party"})


class TestIdentity:
    def test_same_fact_same_id(self):
        """Re-ingesting a document must be a no-op, not a duplicate."""
        assert _base().assertion_id == _base().assertion_id

    def test_method_version_changes_identity(self):
        """A better extractor supersedes its old output instead of colliding."""
        assert _base(method="llm:opus-5").assertion_id != _base(
            method="llm:opus-6"
        ).assertion_id

    def test_tenant_changes_identity(self):
        assert _base(tenant_id="firm-a").assertion_id != _base(tenant_id="firm-b").assertion_id

    def test_confidence_is_not_part_of_identity(self):
        """Load-bearing for the approval rescale, so pinned rather than left to inspection.

        Approval moves `confidence` on a stored assertion. That is only safe because the id
        hashes tenant/subject/predicate/object/method/locator/valid_from and nothing else -- if
        confidence entered the hash, approving a fact would fork its id and every citation,
        audit row and premise link pointing at the old one would dangle.
        """
        assert _base(confidence=0.55).assertion_id == _base(confidence=0.958).assertion_id

    def test_the_raw_score_is_not_part_of_identity_either(self):
        assert _base(raw_confidence=0.1).assertion_id == _base(raw_confidence=0.9).assertion_id

    @pytest.mark.parametrize(
        "field",
        ["tenant_id", "subject_id", "predicate", "object_id", "method", "valid_from"],
    )
    def test_exactly_these_fields_are_hashed(self, field):
        """The other half of the same guarantee: everything the hash *does* cover still moves
        the id, so a change to `_compute_id` cannot quietly narrow it."""
        changed = {
            "tenant_id": "firm-other",
            "subject_id": "Matter-9999",
            "predicate": "ADVERSE_TO",
            "object_id": "Party-Other",
            "method": "llm:opus-9",
            "valid_from": "2020-01-01T00:00:00Z",
        }[field]
        assert _base().assertion_id != _base(**{field: changed}).assertion_id

    def test_mutating_confidence_in_place_does_not_restate_the_id(self):
        """What `approve()` actually does, checked against the recomputed hash."""
        a = _base(confidence=0.55)
        before = a.assertion_id
        a.confidence = 0.91
        assert a._compute_id() == before


class TestBitemporal:
    def test_recorded_at_is_set(self):
        assert _base().recorded_at

    def test_current_until_superseded(self):
        a = _base()
        assert a.is_current
        a.superseded_at = "2026-01-01T00:00:00Z"
        assert not a.is_current


class TestStorageSplit:
    """Edge props stay lean so filtered traversal is one hop."""

    def test_edge_props_are_the_filterable_subset(self):
        props = _base().to_edge_props()
        assert set(props) == {
            "assertion_id",
            "tenant_id",
            "matter_id",
            "epistemic_class",
            "confidence",
            "review_state",
            "superseded_at",
        }

    def test_node_props_carry_full_provenance(self):
        props = _base().to_node_props()
        assert props["method"] == "llm:opus-5"
        assert props["loc_document_id"] == "doc-1"
        assert props["loc_page"] == 3
        assert props["loc_quote"].startswith("within two years")
