"""Merging two entity ids that turned out to name one company.

The property that matters most here is what *survives*. An id is part of an assertion's content
hash, so a merge cannot be an UPDATE — it is N supersedes, and the originals have to stay
fetchable with their own ids intact, or an `as_of` read before the merge stops answering "what did
the file show when we advised?".

The second property is the cascade. A merge is the third kind of correction: a wipe leaves
inferences standing because the belief was honestly held, a retraction cascades because the fact
was wrong, and a merge cascades because the fact was *right but stated about a fork*. Leaving a
conclusion standing would give two conflict edges for one conflict, one citing a closed premise.
"""

from __future__ import annotations

import pytest

from src.documents.merge import MergeError, merge_entities, plan_merge
from src.documents.review import ReviewQueue
from src.graph.assertions import EpistemicClass, ReviewState, SourceLocator, build_assertion
from src.graph.scope import AuthContext
from src.ontology.loader import load_ontology
from src.reasoning.engine import Reasoner

TENANT = "demo-firm"


@pytest.fixture
def ctx() -> AuthContext:
    return AuthContext(user_id="partner@firm.example", tenant_id=TENANT)


@pytest.fixture
def onto():
    return load_ontology("legal")


@pytest.fixture
def queue(onto) -> ReviewQueue:
    return ReviewQueue(
        governing_predicates=onto.governing_predicates,
        canonical_entity_id=onto.canonical_entity_id,
        entity_blocking_keys=onto.entity_blocking_keys,
    )


def fact(subject: str, predicate: str, obj: str, *, matter: str | None = "M-1", onto=None):
    a = build_assertion(
        tenant_id=TENANT,
        subject_id=subject,
        predicate=predicate,
        object_id=obj,
        epistemic_class=EpistemicClass.EXTRACTED_MODEL,
        method="llm:test@v1",
        confidence=0.9,
        source_locator=SourceLocator(document_id="d1", filename="f.pdf", page=1, quote="a quote"),
        matter_id=matter,
        allowed_predicates=onto.extractable_predicates if onto else None,
    )
    a.review_state = ReviewState.APPROVED
    return a


def _live(queue: ReviewQueue, ctx: AuthContext, assertions):
    queue.stage(ctx, assertions)
    queue.promote(ctx, job_id=None)


def _forked(queue: ReviewQueue, ctx: AuthContext, onto):
    """The firm acts for one spelling of a company and opposes the other.

    `conflict_check` finds nothing here, because the party it represents and the party it opposes
    are different nodes. This is the fork the whole mechanism exists to make visible.
    """
    represents = fact("counsel:sian-aldridge", "REPRESENTS", "party:calder-shipping-ag", onto=onto)
    adverse = fact("matter:m-2", "ADVERSE_TO", "party:calder-shipping", matter="M-2", onto=onto)
    _live(queue, ctx, [represents, adverse])
    return represents, adverse


class TestAMergeMakesTheHiddenConflictVisible:
    """The point of the feature, stated as a behaviour rather than a mechanism."""

    def test_the_fork_hides_the_conflict(self, queue, ctx, onto):
        _forked(queue, ctx, onto)
        live = [r.assertion for r in queue.visible(ctx) if r.is_current]
        assert Reasoner(onto).run(ctx, live).count == 0

    def test_merging_reveals_it(self, queue, ctx, onto):
        _forked(queue, ctx, onto)
        merge_entities(
            queue,
            ctx,
            losing_id="party:calder-shipping",
            winning_id="party:calder-shipping-ag",
            reason="the short form in the fixture note is the same company",
            allowed_predicates=onto.governing_predicates,
            canonical_entity_id=onto.canonical_entity_id,
        )
        live = [r.assertion for r in queue.visible(ctx) if r.is_current]

        report = Reasoner(onto).run(ctx, live)
        assert [i.rule_id for i in report.inferences] == ["conflict_check"]
        assert report.inferences[0].assertion.object_id == "party:calder-shipping-ag"

    def test_the_losing_id_is_gone_from_the_live_graph(self, queue, ctx, onto):
        _forked(queue, ctx, onto)
        merge_entities(
            queue,
            ctx,
            losing_id="party:calder-shipping",
            winning_id="party:calder-shipping-ag",
            reason="same company",
            allowed_predicates=onto.governing_predicates,
        )
        live = [r.assertion for r in queue.visible(ctx) if r.is_current]
        ids = {e for a in live for e in (a.subject_id, a.object_id)}
        assert "party:calder-shipping" not in ids
        assert "party:calder-shipping-ag" in ids


class TestNothingIsRewrittenInPlace:
    def test_the_original_survives_with_its_own_id(self, queue, ctx, onto):
        """`_compute_id` hashes the endpoints, so a merge cannot be an UPDATE. The original stays
        fetchable so an `as_of` read before the merge still shows what the document supported."""
        _, adverse = _forked(queue, ctx, onto)
        merge_entities(
            queue,
            ctx,
            losing_id="party:calder-shipping",
            winning_id="party:calder-shipping-ag",
            reason="same company",
            allowed_predicates=onto.governing_predicates,
        )

        record = queue.fetch(ctx, adverse.assertion_id)
        assert record.assertion.object_id == "party:calder-shipping"
        assert record.assertion.superseded_at is not None
        assert not record.is_current

    def test_the_replacement_is_declared_and_names_the_reviewer(self, queue, ctx, onto):
        """A person decided this, and the epistemic class is the axis that keeps that
        distinguishable from a model's reading."""
        _forked(queue, ctx, onto)
        result = merge_entities(
            queue,
            ctx,
            losing_id="party:calder-shipping",
            winning_id="party:calder-shipping-ag",
            reason="same company",
            allowed_predicates=onto.governing_predicates,
        )

        replacement = queue.fetch(ctx, result.rewritten[0]).assertion
        assert replacement.epistemic_class is EpistemicClass.DECLARED
        assert replacement.method == "reviewer:partner@firm.example"

    def test_the_replacement_points_back_at_what_it_replaced(self, queue, ctx, onto):
        _, adverse = _forked(queue, ctx, onto)
        result = merge_entities(
            queue,
            ctx,
            losing_id="party:calder-shipping",
            winning_id="party:calder-shipping-ag",
            reason="same company",
            allowed_predicates=onto.governing_predicates,
        )
        assert queue.fetch(ctx, result.rewritten[0]).corrects == adverse.assertion_id

    def test_the_citation_is_kept(self, queue, ctx, onto):
        """A restated claim with no citation would be an opinion."""
        _forked(queue, ctx, onto)
        result = merge_entities(
            queue,
            ctx,
            losing_id="party:calder-shipping",
            winning_id="party:calder-shipping-ag",
            reason="same company",
            allowed_predicates=onto.governing_predicates,
        )
        assert queue.fetch(ctx, result.rewritten[0]).assertion.source_locator.quote == "a quote"


class TestTheCascade:
    """A merge is the third case: the fact was right, but stated about a fork.

    The conclusions that need the *cascade* specifically are the ones resting on an affected
    premise without naming the merged entity. A conclusion that names it is restated by the loop
    anyway, so testing only that case would pass with no cascade at all — which is exactly the
    mistake this class was written to avoid making silently.
    """

    def _stale_authority_via(self, queue, ctx, onto, overruling: str):
        """`authority_stale` concludes about the *citing document* and the overruled authority.

        Merging the overruling authority therefore touches a premise of a conclusion that does not
        mention it — the shape only the cascade catches.
        """
        cites = fact("document:advice", "CITES", "authority:aquitaine", onto=onto)
        overrules = fact(overruling, "OVERRULES", "authority:aquitaine", matter="M-2", onto=onto)
        _live(queue, ctx, [cites, overrules])
        live = [r.assertion for r in queue.visible(ctx) if r.is_current]
        conclusions = [i.assertion for i in Reasoner(onto).run(ctx, live).inferences]
        assert conclusions, "the fixture proves nothing if the rule did not fire"
        queue.stage(ctx, conclusions)
        for a in conclusions:
            queue.approve(ctx, a.assertion_id)
        queue.promote(ctx, job_id=None)
        return conclusions[0], overrules

    def test_a_conclusion_resting_on_a_merged_premise_is_withdrawn(self, queue, ctx, onto):
        """It does not name the merged authority, so nothing but the cascade reaches it. Left
        standing, its "why do you believe this" resolves to a superseded assertion."""
        conclusion, overrules = self._stale_authority_via(
            queue, ctx, onto, "authority:marisol-ag"
        )
        assert "authority:marisol-ag" not in (conclusion.subject_id, conclusion.object_id)
        assert overrules.assertion_id in conclusion.premises

        merge_entities(
            queue,
            ctx,
            losing_id="authority:marisol-ag",
            winning_id="authority:marisol",
            reason="the reporter citation is the same case",
            allowed_predicates=onto.governing_predicates,
        )
        assert not queue.fetch(ctx, conclusion.assertion_id).is_current

    def test_the_merge_reports_it_as_cascaded_not_restated(self, queue, ctx, onto):
        """`affected` and `cascaded` are separate because they are different events: one person
        decided one thing, and N conclusions fell as a consequence."""
        conclusion, _ = self._stale_authority_via(queue, ctx, onto, "authority:marisol-ag")

        result = merge_entities(
            queue,
            ctx,
            losing_id="authority:marisol-ag",
            winning_id="authority:marisol",
            reason="same case",
            allowed_predicates=onto.governing_predicates,
        )
        assert conclusion.assertion_id in result.cascaded
        assert conclusion.assertion_id not in result.affected

    def test_no_live_assertion_rests_on_a_closed_premise(self, queue, ctx, onto):
        """The invariant the cascade exists for, checked directly rather than by counting."""
        self._stale_authority_via(queue, ctx, onto, "authority:marisol-ag")
        merge_entities(
            queue,
            ctx,
            losing_id="authority:marisol-ag",
            winning_id="authority:marisol",
            reason="same case",
            allowed_predicates=onto.governing_predicates,
        )

        by_id = {r.assertion_id: r for r in queue.visible(ctx)}
        for record in by_id.values():
            if not record.is_current:
                continue
            for premise in record.assertion.premises:
                assert by_id[premise].is_current, (
                    f"{record.assertion_id} is live but rests on closed premise {premise}"
                )

    def test_a_conflict_naming_the_merged_entity_is_restated(self, queue, ctx, onto):
        """The other half: a conclusion that *does* name the losing id is handled by the restate
        loop, so it is reported as affected rather than cascaded."""
        represents = fact("counsel:sian", "REPRESENTS", "party:calder-shipping", onto=onto)
        adverse = fact("matter:m-2", "ADVERSE_TO", "party:calder-shipping", matter="M-2", onto=onto)
        _live(queue, ctx, [represents, adverse])
        live = [r.assertion for r in queue.visible(ctx) if r.is_current]
        conflict = next(i.assertion for i in Reasoner(onto).run(ctx, live).inferences)
        queue.stage(ctx, [conflict])
        queue.approve(ctx, conflict.assertion_id)
        queue.promote(ctx, job_id=None)

        result = merge_entities(
            queue,
            ctx,
            losing_id="party:calder-shipping",
            winning_id="party:calder-shipping-ag",
            reason="same company",
            allowed_predicates=onto.governing_predicates,
        )
        assert conflict.assertion_id in result.affected
        assert not queue.fetch(ctx, conflict.assertion_id).is_current


class TestThePreviewWritesNothing:
    def test_it_reports_what_would_be_restated(self, queue, ctx, onto):
        _forked(queue, ctx, onto)
        plan = plan_merge(
            queue, ctx, losing_id="party:calder-shipping", winning_id="party:calder-shipping-ag"
        )
        assert plan.dry_run
        assert len(plan.affected) == 1
        assert plan.rewritten == ()

    def test_it_changes_nothing(self, queue, ctx, onto):
        _, adverse = _forked(queue, ctx, onto)
        plan_merge(
            queue, ctx, losing_id="party:calder-shipping", winning_id="party:calder-shipping-ag"
        )
        assert queue.fetch(ctx, adverse.assertion_id).is_current
        assert queue.fetch(ctx, adverse.assertion_id).assertion.object_id == "party:calder-shipping"


class TestOverHttp:
    """The wiring, which is where these go wrong."""

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
        """A merge cascades, so the safe default shows what it would do. A caller has to ask for
        the write."""
        c, services = client
        _, adverse = _forked(services.review_queue, ctx, onto)

        body = c.post(
            f"/api/tenants/{TENANT}/entities/merge",
            json={
                "losing_id": "party:calder-shipping",
                "winning_id": "party:calder-shipping-ag",
                "reason": "same company",
            },
        ).json()

        assert body["dry_run"] is True
        assert body["rewritten"] == []
        assert services.review_queue.fetch(ctx, adverse.assertion_id).is_current

    def test_an_explicit_write_restates_the_claim(self, client, ctx, onto):
        c, services = client
        _forked(services.review_queue, ctx, onto)

        r = c.post(
            f"/api/tenants/{TENANT}/entities/merge",
            json={
                "losing_id": "party:calder-shipping",
                "winning_id": "party:calder-shipping-ag",
                "reason": "same company, short form in the note",
                "dry_run": False,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["dry_run"] is False
        assert len(body["rewritten"]) == 1

        live = [rec.assertion for rec in services.review_queue.visible(ctx) if rec.is_current]
        assert "party:calder-shipping" not in {
            e for a in live for e in (a.subject_id, a.object_id)
        }

    def test_a_refused_merge_answers_409(self, client, ctx, onto):
        c, services = client
        _forked(services.review_queue, ctx, onto)

        r = c.post(
            f"/api/tenants/{TENANT}/entities/merge",
            json={
                "losing_id": "party:calder-shipping",
                "winning_id": "court:calder-shipping",
                "reason": "looked similar",
                "dry_run": False,
            },
        )
        assert r.status_code == 409
        assert "different kinds" in r.json()["detail"]


class TestARefusalIsBetterThanAGuess:
    def test_a_merge_needs_a_reason(self, queue, ctx, onto):
        _forked(queue, ctx, onto)
        with pytest.raises(MergeError, match="reason"):
            merge_entities(
                queue,
                ctx,
                losing_id="party:calder-shipping",
                winning_id="party:calder-shipping-ag",
                reason="",
            )

    def test_merging_an_id_into_itself_is_refused(self, queue, ctx, onto):
        _forked(queue, ctx, onto)
        with pytest.raises(MergeError, match="nothing to merge"):
            merge_entities(
                queue,
                ctx,
                losing_id="party:calder-shipping",
                winning_id="party:calder-shipping",
                reason="same",
            )

    def test_merging_across_kinds_is_refused(self, queue, ctx, onto):
        """`party:acme` and `court:acme` are unrelated. Merging would move a fact onto a node of
        the wrong type, which no traversal expects."""
        _live(queue, ctx, [fact("document:d1", "MENTIONS", "party:acme", onto=onto)])
        with pytest.raises(MergeError, match="different kinds"):
            merge_entities(
                queue,
                ctx,
                losing_id="party:acme",
                winning_id="court:acme",
                reason="looked similar",
            )

    def test_merging_into_a_non_canonical_id_is_refused(self, queue, ctx, onto):
        """A merge that minted a fresh fork would be self-defeating."""
        _forked(queue, ctx, onto)
        with pytest.raises(MergeError, match="canonical"):
            merge_entities(
                queue,
                ctx,
                losing_id="party:calder-shipping",
                winning_id="Party:Calder Shipping AG",
                reason="same company",
                canonical_entity_id=onto.canonical_entity_id,
            )

    def test_merging_an_id_nothing_names_is_refused(self, queue, ctx, onto):
        """Silence here would report a successful merge that did nothing, which is the shape of
        every bug this repo keeps finding."""
        _forked(queue, ctx, onto)
        with pytest.raises(MergeError, match="no current assertion"):
            merge_entities(
                queue,
                ctx,
                losing_id="party:never-existed",
                winning_id="party:calder-shipping-ag",
                reason="same company",
            )

    def test_a_merge_is_never_automatic(self):
        """`entity_blocking_keys` finds candidates; a person decides. There is no threshold at
        which this runs by itself, for the same reason there is no `approve_all(auto=True)`."""
        import inspect

        source = inspect.getsource(merge_entities)
        assert "reason" in inspect.signature(merge_entities).parameters
        assert "blocking" not in source, "a merge must not consult the candidate finder itself"
