"""Job persistence. The property under test is that a job outlives the process.

Ingestion runs in a background task that dies with its container, so the only thing
making a half-finished ingest recoverable is that its state was written before the work
was attempted. These tests assert that ordering, not that DynamoDB was called.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.documents.job_store import (
    DynamoJobStore,
    InMemoryJobStore,
    JobTracker,
    document_gsi_pk,
    documents_by_state,
    latest_per_document,
    tenant_pk,
)
from src.documents.models import DocumentMeta, IngestJob, JobState

TENANT = "firm-acme"


def doc(tenant_id: str = TENANT, content: str = "a" * 64) -> DocumentMeta:
    return DocumentMeta(
        tenant_id=tenant_id,
        bucket="lex-docs",
        key=f"processed/{tenant_id}/{content}/motion.pdf",
        filename="motion.pdf",
        media_type="application/pdf",
        content_sha256=content,
        byte_size=11,
        uploaded_by="alice",
    )


class FakeTable:
    """Enough of a boto3 Table to hold items and answer the two queries we make."""

    def __init__(self) -> None:
        self.items: dict[tuple[str, str], dict[str, Any]] = {}
        self.puts: list[dict[str, Any]] = []

    def put_item(self, **kwargs: Any) -> dict[str, Any]:
        item = kwargs["Item"]
        self.puts.append(item)
        self.items[(item["PK"], item["SK"])] = item
        return {}

    def get_item(self, **kwargs: Any) -> dict[str, Any]:
        key = kwargs["Key"]
        item = self.items.get((key["PK"], key["SK"]))
        return {"Item": item} if item else {}

    def query(self, **kwargs: Any) -> dict[str, Any]:
        values = kwargs["ExpressionAttributeValues"]
        if kwargs.get("IndexName"):
            found = [i for i in self.items.values() if i["GSI1PK"] == values[":pk"]]
            return {"Items": sorted(found, key=lambda i: i["GSI1SK"])}
        found = [
            i
            for i in self.items.values()
            if i["PK"] == values[":pk"] and i["SK"].startswith(values[":sk"])
        ]
        if ":state" in values:
            found = [i for i in found if i["state"] == values[":state"]]
        return {"Items": found}


@pytest.fixture(params=["memory", "dynamo"])
def store(request):
    """Both implementations must behave identically — the in-memory one is what tests
    and single-process dev run on, so a divergence would hide bugs until deploy."""
    if request.param == "memory":
        return InMemoryJobStore()
    return DynamoJobStore(table=FakeTable())


class TestRoundTrip:
    def test_a_job_survives_a_write_and_read(self, store):
        job = IngestJob.for_document(doc())
        store.put_job(job)
        got = store.get_job(TENANT, job.job_id)
        assert got is not None
        assert got.job_id == job.job_id
        assert got.state is JobState.REGISTERED

    def test_history_survives_the_round_trip(self, store):
        """The history is the audit trail of the ingest. Losing it loses the reason a
        job failed, which is the only thing that makes it diagnosable."""
        job = IngestJob.for_document(doc())
        job.advance(JobState.FETCHING)
        job.advance(JobState.PARSING)
        job.advance(JobState.PARSE_FAILED, reason="no vision model")
        store.put_job(job)

        got = store.get_job(TENANT, job.job_id)
        assert [h.state for h in got.history] == [
            JobState.REGISTERED,
            JobState.FETCHING,
            JobState.PARSING,
            JobState.PARSE_FAILED,
        ]
        assert got.reason == "no vision model"
        assert got.state.retry_target is JobState.PARSING

    def test_attempts_survive_the_round_trip(self, store):
        job = IngestJob.for_document(doc())
        job.advance(JobState.FETCHING)
        store.put_job(job)
        assert store.get_job(TENANT, job.job_id).attempts["FETCHING"] == 1

    def test_an_unknown_job_is_none_not_an_error(self, store):
        assert store.get_job(TENANT, "job-nope") is None

    def test_another_tenants_job_is_not_visible(self, store):
        """Tenant isolation here is a key prefix, so a wrong tenant must simply miss."""
        job = IngestJob.for_document(doc())
        store.put_job(job)
        assert store.get_job("firm-other", job.job_id) is None


class TestQueryByDocument:
    def test_finds_every_job_for_one_document(self, store):
        """The UI polls by document id: after an upload that is the only id it holds."""
        d = doc()
        first = IngestJob.for_document(d)
        second = IngestJob.for_document(d)
        second.job_id = "job-second"
        store.put_job(first)
        store.put_job(second)

        found = store.jobs_for_document(TENANT, d.document_id)
        assert {j.job_id for j in found} == {first.job_id, "job-second"}

    def test_does_not_return_another_documents_jobs(self, store):
        store.put_job(IngestJob.for_document(doc(content="a" * 64)))
        other = doc(content="b" * 64)
        store.put_job(IngestJob.for_document(other))
        found = store.jobs_for_document(TENANT, other.document_id)
        assert len(found) == 1
        assert found[0].document_id == other.document_id

    def test_jobs_in_state_filters(self, store):
        live = IngestJob.for_document(doc(content="c" * 64))
        live.advance(JobState.FETCHING)
        store.put_job(live)
        store.put_job(IngestJob.for_document(doc(content="d" * 64)))

        assert [j.job_id for j in store.jobs_in_state(TENANT, JobState.FETCHING)] == [live.job_id]


def job(doc_id: str, state: JobState, created_at: str, job_id: str = "") -> IngestJob:
    return IngestJob(
        job_id=job_id or f"job-{doc_id}-{created_at}",
        document_id=doc_id,
        tenant_id=TENANT,
        state=state,
        created_at=created_at,
    )


class TestStateProgress:
    def test_the_happy_path_is_strictly_increasing(self):
        """Derived from `_TRANSITIONS`, so a new state cannot be ordered wrongly by omission."""
        path = [
            JobState.REGISTERED,
            JobState.FETCHING,
            JobState.PARSING,
            JobState.CHUNKING,
            JobState.EXTRACTING,
            JobState.EMBEDDING,
            JobState.GRAPH_STAGED,
            JobState.PENDING_REVIEW,
            JobState.APPROVED,
            JobState.LIVE,
        ]
        assert [s.progress for s in path] == sorted(s.progress for s in path)
        assert len({s.progress for s in path}) == len(path)

    def test_live_is_the_furthest_state(self):
        assert JobState.LIVE.progress == max(s.progress for s in JobState)

    def test_a_failure_outranks_the_phase_before_it_but_not_its_own(self):
        """PARSE_FAILED means parsing was entered and did not finish."""
        assert JobState.FETCHING.progress < JobState.PARSE_FAILED.progress
        assert JobState.PARSE_FAILED.progress < JobState.PARSING.progress

    def test_every_state_is_ordered(self):
        assert {s.progress for s in JobState} and all(isinstance(s.progress, int) for s in JobState)


class TestLatestPerDocument:
    def test_one_row_per_document(self):
        jobs = [
            job("doc-a", JobState.LIVE, "2026-01-01T00:00:00Z"),
            job("doc-a", JobState.LIVE, "2026-01-02T00:00:00Z"),
            job("doc-b", JobState.LIVE, "2026-01-03T00:00:00Z"),
        ]
        assert {j.document_id for j in latest_per_document(jobs)} == {"doc-a", "doc-b"}

    def test_the_furthest_attempt_wins_over_the_newest(self):
        """The bug this exists for: a document whose facts are live displayed as pending
        because a later re-upload stalled at the review gate. Every job for one document ran
        over identical bytes, so the furthest state is the honest answer."""
        jobs = [
            job("doc-a", JobState.LIVE, "2026-01-01T00:00:00Z"),
            job("doc-a", JobState.PENDING_REVIEW, "2026-01-02T00:00:00Z"),
        ]
        assert latest_per_document(jobs)[0].state is JobState.LIVE

    def test_a_later_failure_does_not_retract_a_live_document(self):
        jobs = [
            job("doc-a", JobState.LIVE, "2026-01-01T00:00:00Z"),
            job("doc-a", JobState.PARSE_FAILED, "2026-01-02T00:00:00Z"),
        ]
        assert latest_per_document(jobs)[0].state is JobState.LIVE

    def test_the_newest_wins_when_both_reached_the_same_state(self):
        jobs = [
            job("doc-a", JobState.LIVE, "2026-01-01T00:00:00Z", job_id="old"),
            job("doc-a", JobState.LIVE, "2026-01-02T00:00:00Z", job_id="new"),
        ]
        assert latest_per_document(jobs)[0].job_id == "new"

    def test_a_retry_that_got_further_wins(self):
        jobs = [
            job("doc-a", JobState.PARSE_FAILED, "2026-01-01T00:00:00Z"),
            job("doc-a", JobState.PENDING_REVIEW, "2026-01-02T00:00:00Z"),
        ]
        assert latest_per_document(jobs)[0].state is JobState.PENDING_REVIEW

    def test_rows_come_back_newest_first(self):
        jobs = [
            job("doc-a", JobState.LIVE, "2026-01-01T00:00:00Z"),
            job("doc-b", JobState.LIVE, "2026-01-03T00:00:00Z"),
        ]
        assert [j.document_id for j in latest_per_document(jobs)] == ["doc-b", "doc-a"]


class TestDocumentsByState:
    def test_four_attempts_on_one_document_count_once(self):
        """The dashboard counted job rows, so a document uploaded four times added four to
        the totals and appeared under several states at the same time."""
        jobs = [
            job("doc-a", JobState.LIVE, "2026-01-01T00:00:00Z"),
            job("doc-a", JobState.LIVE, "2026-01-02T00:00:00Z"),
            job("doc-a", JobState.PENDING_REVIEW, "2026-01-03T00:00:00Z"),
            job("doc-a", JobState.PARSE_FAILED, "2026-01-04T00:00:00Z"),
        ]
        assert documents_by_state(jobs) == {"LIVE": 1}

    def test_states_with_no_documents_are_omitted(self):
        assert documents_by_state([job("doc-a", JobState.LIVE, "2026-01-01T00:00:00Z")]) == {
            "LIVE": 1
        }

    def test_documents_in_different_states_are_counted_separately(self):
        jobs = [
            job("doc-a", JobState.LIVE, "2026-01-01T00:00:00Z"),
            job("doc-b", JobState.PENDING_REVIEW, "2026-01-01T00:00:00Z"),
        ]
        assert documents_by_state(jobs) == {"LIVE": 1, "PENDING_REVIEW": 1}


class TestScreeningAcrossAttempts:
    """Visibility is decided over every attempt, not just the row that gets displayed.

    Two attempts on one document can name different matters -- a refile updates each job row
    separately and can fail part-way. Judging the wall from the winning row alone would then
    show a screened document whenever the unfiled attempt happened to win.
    """

    def _ctx(self):
        from src.graph.scope import AuthContext

        return AuthContext(
            user_id="alice",
            tenant_id=TENANT,
            matter_denylist=frozenset({"NTL-2026-0114"}),
        )

    def test_a_screened_matter_on_any_attempt_hides_the_document(self):
        from src.api.routes_documents import _screened

        jobs = [
            job("doc-a", JobState.LIVE, "2026-01-01T00:00:00Z"),
            job("doc-a", JobState.LIVE, "2026-01-02T00:00:00Z"),
        ]
        jobs[1].matter_id = "NTL-2026-0114"
        assert _screened(self._ctx(), jobs, "doc-a") is True

    def test_a_readable_document_is_not_hidden(self):
        from src.api.routes_documents import _screened

        jobs = [job("doc-a", JobState.LIVE, "2026-01-01T00:00:00Z")]
        jobs[0].matter_id = "NTL-2026-0999"
        assert _screened(self._ctx(), jobs, "doc-a") is False

    def test_another_documents_screen_is_not_applied(self):
        from src.api.routes_documents import _screened

        jobs = [
            job("doc-a", JobState.LIVE, "2026-01-01T00:00:00Z"),
            job("doc-b", JobState.LIVE, "2026-01-01T00:00:00Z"),
        ]
        jobs[1].matter_id = "NTL-2026-0114"
        assert _screened(self._ctx(), jobs, "doc-a") is False


class TestKeys:
    def test_tenants_never_share_a_partition(self):
        assert tenant_pk("firm-a") != tenant_pk("firm-b")

    def test_document_index_key_is_tenant_scoped(self):
        """Two firms could hold documents with the same id only if the digest collided,
        but the index key is scoped anyway — a shared partition is one bug from a leak."""
        assert document_gsi_pk("firm-a", "doc-1") != document_gsi_pk("firm-b", "doc-1")


class TestJobTracker:
    def test_advance_persists_without_a_second_call(self):
        """The point of the tracker: no call site can advance a job and forget to save."""
        store = InMemoryJobStore()
        tracker = JobTracker(store)
        job = tracker.open(IngestJob.for_document(doc()))

        tracker.advance(job, JobState.FETCHING)
        assert store.get_job(TENANT, job.job_id).state is JobState.FETCHING

    def test_state_is_written_before_the_work_is_attempted(self):
        """A job recorded as PARSING with no worker is retryable. One still recorded as
        REGISTERED looks like one nobody ever picked up, and is lost."""
        store = InMemoryJobStore()
        tracker = JobTracker(store)
        job = tracker.open(IngestJob.for_document(doc()))
        tracker.advance(job, JobState.FETCHING)

        observed: list[JobState] = []

        def work() -> None:
            observed.append(store.get_job(TENANT, job.job_id).state)
            raise RuntimeError("container died")

        tracker.advance(job, JobState.PARSING)
        with pytest.raises(RuntimeError):
            work()

        assert observed == [JobState.PARSING]
        assert store.get_job(TENANT, job.job_id).state is JobState.PARSING

    def test_fail_records_the_reason(self):
        store = InMemoryJobStore()
        tracker = JobTracker(store)
        job = tracker.open(IngestJob.for_document(doc()))
        tracker.advance(job, JobState.FETCHING)
        tracker.advance(job, JobState.PARSING)

        tracker.fail(job, JobState.PARSE_FAILED, "Bedrock unreachable")
        got = store.get_job(TENANT, job.job_id)
        assert got.state is JobState.PARSE_FAILED
        assert got.reason == "Bedrock unreachable"

    def test_fail_persists_even_from_an_illegal_state(self):
        """A failure handler must not raise a second exception over the first, or the
        reason the job died is lost and it becomes undiagnosable."""
        store = InMemoryJobStore()
        tracker = JobTracker(store)
        job = tracker.open(IngestJob.for_document(doc()))

        tracker.fail(job, JobState.EMBED_FAILED, "vector store down")
        assert store.get_job(TENANT, job.job_id).reason == "vector store down"

    def test_a_stored_failure_knows_where_to_resume(self):
        store = InMemoryJobStore()
        tracker = JobTracker(store)
        job = tracker.open(IngestJob.for_document(doc()))
        tracker.advance(job, JobState.FETCHING)
        tracker.advance(job, JobState.PARSING)
        tracker.fail(job, JobState.PARSE_FAILED, "no vision model")

        resumed = store.get_job(TENANT, job.job_id)
        assert resumed.state.retry_target is JobState.PARSING
        resumed.retry()
        assert resumed.state is JobState.PARSING
