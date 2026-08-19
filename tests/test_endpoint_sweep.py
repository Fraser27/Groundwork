"""Withdrawing facts written before their pack's endpoints were enforced.

The incident this closes, end to end. A reviewer asked *"Who is the counsel for Halveston"* and
the ethical wall withheld all three rows — including the answer. `POTENTIAL_CONFLICT` declares
`Matter -> Party`, a rule bound its subject to a Party, and the conflict was stored `Party ->
Party`. Since that predicate declares `blocks: both`, the wall tainted both parties, one of which
the firm represents. A question about the firm's own client's counsel returned nothing.

`build_assertion` refuses that write now, but the fact was already in the graph, and a write-time
guard is prospective. This is the remediation, and the property that matters is what *survives*:
the offending assertion is retracted rather than deleted, so an as-of read before now still shows
what the graph asserted while advice rested on it.
"""

from __future__ import annotations

import pytest

from src.admin_ops import sweep_undeclared_endpoints
from src.documents.review import InMemoryAssertionStore, ReviewQueue
from src.graph.assertions import (
    EpistemicClass,
    ReviewState,
    SourceLocator,
    build_assertion,
)
from src.graph.scope import AuthContext
from src.ontology.loader import load_ontology
from src.query.blocks import blocks_for, seeds_from
from src.query.graph_reader import GraphReader

TENANT = "demo-firm"
HAL = "HAL-2025-0092"


@pytest.fixture
def ctx() -> AuthContext:
    return AuthContext(user_id="partner@firm.example", tenant_id=TENANT)


@pytest.fixture
def onto():
    return load_ontology("legal")


def _fact(subject: str, predicate: str, obj: str, *, onto, matter: str | None = HAL, inferred=False):
    """A live fact, built *without* the endpoint check so the pre-fix graph can be reproduced.

    Passing `endpoint_kinds=None` is how the graph looked before invariant 7 existed. Reproducing
    that state is the whole point: a test that could not create the bad fact could not prove the
    sweep removes it.
    """
    kw: dict = {}
    if inferred:
        kw = {
            "epistemic_class": EpistemicClass.INFERRED,
            "premises": ("premise-a", "premise-b"),
            "rule_id": "conflict_check",
            "rule_version": "v1",
        }
    else:
        kw = {"epistemic_class": EpistemicClass.EXTRACTED_MODEL}
    a = build_assertion(
        tenant_id=TENANT,
        subject_id=subject,
        predicate=predicate,
        object_id=obj,
        method="rule:conflict_check@v1" if inferred else "llm:test@v1",
        confidence=0.9,
        source_locator=SourceLocator(
            document_id="doc-hal", filename="authority-note.pdf", page=1, quote="a quoted span"
        ),
        matter_id=matter,
        endpoint_kinds=None,
        **kw,
    )
    a.review_state = ReviewState.APPROVED
    return a


def _services(onto, queue):
    class Services:
        ontology = onto
        review_queue = queue

    return Services()


def _the_incident(ctx, onto):
    """The deployed graph as the reviewer found it.

    The `INSTRUCTS` fact is the answer to "who is the counsel for Halveston". The conflict is the
    invalid one: Party->Party, so `blocks: both` taints Halveston, the firm's own client.
    """
    queue = ReviewQueue(InMemoryAssertionStore(), governing_predicates=onto.governing_predicates)
    answer = _fact(
        "party:halveston-chartering-limited", "INSTRUCTS", "counsel:sian-aldridge", onto=onto
    )
    conflict = _fact(
        "party:calder-shipping-ag",
        "POTENTIAL_CONFLICT",
        "party:halveston-chartering-limited",
        onto=onto,
        inferred=True,
    )
    queue.stage(ctx, [answer, conflict], job_id="j1")
    queue.promote(ctx, job_id="j1")
    return queue, answer, conflict


class TestTheIncidentIsReproducible:
    """A remediation test proves nothing unless it can first create the broken state.

    **What the incident now means.** When this was written `POTENTIAL_CONFLICT` declared
    `blocks: both`, and the invalid Party->Party conflict withheld the answer outright. The pack now
    declares `effect: notify`, so the same bad fact suppresses nothing -- it flags the firm's own
    client as being in conflict with itself. Still wrong and still worth sweeping, but the harm is a
    false accusation rather than a blackout. These assert the harm that exists today.
    """

    def test_the_invalid_conflict_flags_the_answer(self, ctx, onto):
        queue, _, _ = _the_incident(ctx, onto)
        reader = GraphReader(queue, ontology=onto)

        hits = reader.search(ctx, "Who is the counsel for Halveston", min_confidence=0.8)
        assert any(h["predicate"] == "INSTRUCTS" for h in hits), "the answer must be findable"

        screen = blocks_for(ctx, graph_reader=reader, seeds=seeds_from(hits))
        assert screen.advisories, "the invalid conflict must be reported, or nothing is under test"
        # Notify, so the evidence survives -- which is *why* the sweep still matters: the finding is
        # now a false statement about the client rather than a wall in front of them.
        assert screen.keep(hits) == hits

    def test_the_conflict_names_the_firms_own_client(self, ctx, onto):
        queue, _, _ = _the_incident(ctx, onto)
        reader = GraphReader(queue, ontology=onto)
        screen = blocks_for(ctx, graph_reader=reader, seeds=["party:halveston-chartering-limited"])
        flagged = {b.subject for b in screen.advisories}
        assert "party:halveston-chartering-limited" in flagged


class TestTheSweepFindsIt:
    def test_the_invalid_conflict_is_reported(self, ctx, onto):
        queue, _, conflict = _the_incident(ctx, onto)
        report = sweep_undeclared_endpoints(_services(onto, queue), ctx)

        assert [o["assertion_id"] for o in report.offenders] == [conflict.assertion_id]
        offender = report.offenders[0]
        assert offender["written"] == "party -> party"
        assert "matter" in offender["declared"]

    def test_a_blocking_offender_is_named_as_such(self, ctx, onto):
        """Urgency: an offender that vetoes is withholding evidence right now, while one that only
        informs is a correctness problem."""
        queue, _, _ = _the_incident(ctx, onto)
        report = sweep_undeclared_endpoints(_services(onto, queue), ctx)
        assert report.blocking == 1

    def test_valid_facts_are_left_alone(self, ctx, onto):
        """The guard on the guard. `INSTRUCTS` is Party -> Counsel, exactly as declared, and a
        sweep that withdrew it would be worse than the bug."""
        queue, answer, _ = _the_incident(ctx, onto)
        report = sweep_undeclared_endpoints(_services(onto, queue), ctx)
        assert answer.assertion_id not in [o["assertion_id"] for o in report.offenders]

    def test_a_predicate_with_no_declaration_is_not_an_offender(self, ctx, onto):
        """Every descriptive predicate declares no endpoints, so there is nothing to violate."""
        queue = ReviewQueue(
            InMemoryAssertionStore(), governing_predicates=onto.governing_predicates
        )
        mention = _fact("document:doc-1", "MENTIONS", "party:calder-shipping-ag", onto=onto)
        queue.stage(ctx, [mention], job_id="j1")
        queue.promote(ctx, job_id="j1")

        report = sweep_undeclared_endpoints(_services(onto, queue), ctx)
        assert report.offenders == []
        assert report.checked == 1


class TestDryRunIsTheDefault:
    def test_it_writes_nothing(self, ctx, onto):
        queue, _, conflict = _the_incident(ctx, onto)
        report = sweep_undeclared_endpoints(_services(onto, queue), ctx)

        assert report.dry_run
        assert report.retracted == []
        assert queue.fetch(ctx, conflict.assertion_id).is_current

    def test_writing_has_to_be_asked_for(self, ctx, onto):
        queue, _, conflict = _the_incident(ctx, onto)
        report = sweep_undeclared_endpoints(_services(onto, queue), ctx, dry_run=False)

        assert not report.dry_run
        assert report.retracted == [conflict.assertion_id]


class TestTheRemediation:
    def test_the_offender_is_retracted_not_deleted(self, ctx, onto):
        """An as-of read before now must still show what the graph asserted while advice rested
        on it. That is the whole reason retraction exists rather than a delete."""
        queue, _, conflict = _the_incident(ctx, onto)
        sweep_undeclared_endpoints(_services(onto, queue), ctx, dry_run=False)

        record = queue.fetch(ctx, conflict.assertion_id)
        assert record.assertion.superseded_at is not None
        assert not record.is_current
        assert record.assertion.subject_id == "party:calder-shipping-ag"

    def test_the_reason_names_the_declaration_it_violated(self, ctx, onto):
        """Six months on, "why was this withdrawn" is the only question anyone asks."""
        queue, _, conflict = _the_incident(ctx, onto)
        sweep_undeclared_endpoints(_services(onto, queue), ctx, dry_run=False)

        reason = queue.fetch(ctx, conflict.assertion_id).retracted_reason or ""
        assert "POTENTIAL_CONFLICT" in reason
        assert "party -> party" in reason

    def test_the_question_is_answerable_again(self, ctx, onto):
        """The acceptance test. Manual verification of this regresses; this does not."""
        queue, _, _ = _the_incident(ctx, onto)
        sweep_undeclared_endpoints(_services(onto, queue), ctx, dry_run=False)

        reader = GraphReader(queue, ontology=onto)
        hits = reader.search(ctx, "Who is the counsel for Halveston", min_confidence=0.8)
        screen = blocks_for(ctx, graph_reader=reader, seeds=seeds_from(hits))
        kept = screen.keep(hits)

        assert not screen, "nothing should veto once the invalid conflict is withdrawn"
        # The finding notifies rather than withholds, so "the rows survived" is true before the
        # sweep too. What the sweep changes is that the client is no longer *accused*.
        assert screen.advisories == [], "the false conflict must be gone, not merely non-blocking"
        assert any(h["predicate"] == "INSTRUCTS" for h in kept)
        assert any(h["object_id"] == "counsel:sian-aldridge" for h in kept)

    def test_a_second_sweep_finds_nothing(self, ctx, onto):
        queue, _, _ = _the_incident(ctx, onto)
        sweep_undeclared_endpoints(_services(onto, queue), ctx, dry_run=False)
        again = sweep_undeclared_endpoints(_services(onto, queue), ctx, dry_run=False)
        assert again.offenders == []


class TestItDegradesRatherThanGuesses:
    def test_no_pack_means_it_says_so(self, ctx, onto):
        """"Nothing to fix" and "nothing could be checked" must not look the same."""

        class Bare:
            ontology = None
            review_queue = None

        report = sweep_undeclared_endpoints(Bare(), ctx)
        assert report.errors
        assert report.offenders == []


class TestOverHttp:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from src.api.app import create_app
        from src.api.deps import get_services
        from src.config import AuthConfig, GraphConfig, LexGraphConfig

        cfg = LexGraphConfig(
            environment="local",
            auth=AuthConfig(dev_bypass_tenant=TENANT),
            graph=GraphConfig(uri="bolt://127.0.0.1:1", user="none", password="none"),
        )
        cfg.validate()
        return TestClient(create_app(cfg)), get_services()

    def test_the_default_is_a_preview(self, client, ctx, onto):
        c, services = client
        conflict = _fact(
            "party:calder-shipping-ag",
            "POTENTIAL_CONFLICT",
            "party:halveston-chartering-limited",
            onto=onto,
            inferred=True,
        )
        services.review_queue.stage(ctx, [conflict], job_id="j1")
        services.review_queue.promote(ctx, job_id="j1")

        body = c.post(
            f"/api/tenants/{TENANT}/admin/sweep-endpoints", json={"dry_run": True}
        ).json()
        assert body["dry_run"] is True
        assert body["retracted"] == []
        assert body["blocking"] == 1
        assert services.review_queue.fetch(ctx, conflict.assertion_id).is_current

    def test_an_explicit_write_retracts(self, client, ctx, onto):
        c, services = client
        conflict = _fact(
            "party:calder-shipping-ag",
            "POTENTIAL_CONFLICT",
            "party:halveston-chartering-limited",
            onto=onto,
            inferred=True,
        )
        services.review_queue.stage(ctx, [conflict], job_id="j1")
        services.review_queue.promote(ctx, job_id="j1")

        body = c.post(
            f"/api/tenants/{TENANT}/admin/sweep-endpoints", json={"dry_run": False}
        ).json()
        assert body["retracted"] == [conflict.assertion_id]
        assert not services.review_queue.fetch(ctx, conflict.assertion_id).is_current
