"""Persistence for ingest jobs, so a job survives the process that started it.

Ingestion runs in a background task, and a background task dies with its container —
a deploy or a scale-in mid-ingest takes the work with it. That is tolerable only if the
job's state outlives the process, because `JobState.retry_target` then says exactly
which phase to restart from.

The ordering rule is the whole design: **a state is written before the work it
describes is attempted**, never after. A job recorded as PARSING with no live worker is
a job that can be retried; a job still recorded as REGISTERED while a dead process was
half-way through parsing is indistinguishable from one nobody ever picked up.

    PK = TENANT#{tenant}      SK = JOB#{job_id}
    GSI1PK = TENANT#{tenant}#DOC#{document_id}

The GSI answers "what happened to my upload" — the UI polls by document, because that
is the id the browser has after an upload, and it has no job id until the notification
has been processed.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterable
from typing import Any, Protocol

from src.documents.models import IngestJob, JobState

logger = logging.getLogger(__name__)

#: Must match the index name in `cdk/lib/data-stack.ts`.
GSI1 = "GSI1"

JOB = "JOB#"

#: Jobs are bookkeeping over an S3 object that is still there, so expiring them costs a
#: re-run and nothing else. Long enough that a failure is still diagnosable next week.
JOB_TTL_SECONDS = 30 * 24 * 3600


def latest_per_document(jobs: Iterable[IngestJob]) -> list[IngestJob]:
    """One job per document: the attempt that got furthest, newest breaking a tie.

    Not the newest attempt. A document id is the hash of its bytes, so every job for one
    document ran over identical content — whatever the furthest attempt reached is therefore
    true of the document *now*, and a later re-upload that failed at parsing did not take
    those facts back out of the graph. Showing PENDING_REVIEW for a document whose facts went
    live hours ago understates what the system holds, which is the worse error of the two.

    The failed attempt is not hidden: `/documents/{id}/jobs` and the detail timeline still
    carry every attempt, which is where a stuck ingest is diagnosed.
    """
    best: dict[str, IngestJob] = {}
    for job in jobs:
        seen = best.get(job.document_id)
        if seen is None or (job.state.progress, job.created_at) > (
            seen.state.progress,
            seen.created_at,
        ):
            best[job.document_id] = job
    return sorted(best.values(), key=lambda j: j.created_at, reverse=True)


def documents_by_state(jobs: Iterable[IngestJob]) -> dict[str, int]:
    """How many *documents* stand in each state, for the dashboard.

    Counting job rows instead counted attempts: one document uploaded four times added four
    to the totals and appeared under several states at once, so the dashboard both inflated
    and contradicted itself. States with no documents are omitted, as before.
    """
    counts: dict[str, int] = {}
    for job in latest_per_document(jobs):
        counts[job.state.value] = counts.get(job.state.value, 0) + 1
    return counts


class TableLike(Protocol):
    """The slice of a boto3 DynamoDB `Table` resource this module uses.

    No `scan`, deliberately: a scan on a status poll would degrade with tenant size.
    """

    def put_item(self, **kwargs: Any) -> dict[str, Any]: ...
    def get_item(self, **kwargs: Any) -> dict[str, Any]: ...
    def query(self, **kwargs: Any) -> dict[str, Any]: ...
    def delete_item(self, **kwargs: Any) -> dict[str, Any]: ...


def tenant_pk(tenant_id: str) -> str:
    return f"TENANT#{tenant_id}"


def document_gsi_pk(tenant_id: str, document_id: str) -> str:
    return f"TENANT#{tenant_id}#DOC#{document_id}"


def _item(job: IngestJob) -> dict[str, Any]:
    return {
        "PK": tenant_pk(job.tenant_id),
        "SK": f"{JOB}{job.job_id}",
        "GSI1PK": document_gsi_pk(job.tenant_id, job.document_id),
        "GSI1SK": job.created_at,
        "tenant_id": job.tenant_id,
        "job_id": job.job_id,
        "document_id": job.document_id,
        "state": job.state.value,
        "expires_at": int(time.time()) + JOB_TTL_SECONDS,
        # The model owns its own shape, including the append-only history. Storing one
        # JSON blob keeps a schema change from needing a table migration; nothing
        # queries on the inner fields, only on the keys above.
        "job": job.model_dump_json(),
    }


def _to_job(item: dict[str, Any]) -> IngestJob:
    return IngestJob.model_validate(json.loads(item["job"]))


class InMemoryJobStore:
    """Default store. Real enough for tests and single-process dev, and the reason
    nothing above this depends on DynamoDB being reachable."""

    def __init__(self) -> None:
        self._jobs: dict[tuple[str, str], IngestJob] = {}

    def put_job(self, job: IngestJob) -> None:
        self._jobs[(job.tenant_id, job.job_id)] = job.model_copy(deep=True)

    def get_job(self, tenant_id: str, job_id: str) -> IngestJob | None:
        job = self._jobs.get((tenant_id, job_id))
        return job.model_copy(deep=True) if job is not None else None

    def jobs_for_document(self, tenant_id: str, document_id: str) -> list[IngestJob]:
        return sorted(
            (
                j.model_copy(deep=True)
                for j in self._jobs.values()
                if j.tenant_id == tenant_id and j.document_id == document_id
            ),
            key=lambda j: j.created_at,
        )

    def jobs_in_state(self, tenant_id: str, state: JobState) -> list[IngestJob]:
        return [
            j.model_copy(deep=True)
            for j in self._jobs.values()
            if j.tenant_id == tenant_id and j.state is state
        ]

    def jobs_for_tenant(self, tenant_id: str) -> list[IngestJob]:
        return sorted(
            (j.model_copy(deep=True) for j in self._jobs.values() if j.tenant_id == tenant_id),
            key=lambda j: j.created_at,
            reverse=True,
        )

    def drop_job(self, tenant_id: str, job_id: str) -> bool:
        return self._jobs.pop((tenant_id, job_id), None) is not None

    def drop_tenant(self, tenant_id: str) -> int:
        doomed = [k for k in self._jobs if k[0] == tenant_id]
        for key in doomed:
            del self._jobs[key]
        return len(doomed)


class DynamoJobStore:
    """Job state in one DynamoDB table. boto3 is lazy and injectable, so tests need
    no AWS credentials."""

    def __init__(
        self,
        table_name: str = "",
        *,
        table: TableLike | None = None,
        table_factory: Callable[[], TableLike] | None = None,
        index_name: str = GSI1,
    ) -> None:
        self.table_name = table_name
        self.index_name = index_name
        self._table = table
        self._table_factory = table_factory

    @property
    def table(self) -> TableLike:
        if self._table is None:
            factory = self._table_factory
            if factory is None:
                import boto3

                name = self.table_name
                # Resource API, not client: it hands back plain Python values so
                # nothing here has to speak AttributeValue.
                factory = lambda: boto3.resource("dynamodb").Table(name)
            self._table = factory()
        return self._table

    def put_job(self, job: IngestJob) -> None:
        self.table.put_item(Item=_item(job))

    def get_job(self, tenant_id: str, job_id: str) -> IngestJob | None:
        got = self.table.get_item(Key={"PK": tenant_pk(tenant_id), "SK": f"{JOB}{job_id}"})
        item = got.get("Item")
        return _to_job(item) if item else None

    def jobs_for_document(self, tenant_id: str, document_id: str) -> list[IngestJob]:
        got = self.table.query(
            IndexName=self.index_name,
            KeyConditionExpression="GSI1PK = :pk",
            ExpressionAttributeValues={":pk": document_gsi_pk(tenant_id, document_id)},
        )
        return [_to_job(i) for i in got.get("Items", [])]

    def jobs_in_state(self, tenant_id: str, state: JobState) -> list[IngestJob]:
        got = self.table.query(
            KeyConditionExpression="PK = :pk AND begins_with(SK, :sk)",
            # `state` is a DynamoDB reserved word, hence the alias.
            FilterExpression="#s = :state",
            ExpressionAttributeNames={"#s": "state"},
            ExpressionAttributeValues={
                ":pk": tenant_pk(tenant_id),
                ":sk": JOB,
                ":state": state.value,
            },
        )
        return [_to_job(i) for i in got.get("Items", [])]

    def jobs_for_tenant(self, tenant_id: str) -> list[IngestJob]:
        """Every job for a tenant, newest first.

        A query on the partition key, not a scan: jobs are already partitioned by tenant,
        so the document list costs one read regardless of how many tenants share the table.
        """
        got = self.table.query(
            KeyConditionExpression="PK = :pk AND begins_with(SK, :sk)",
            ExpressionAttributeValues={":pk": tenant_pk(tenant_id), ":sk": JOB},
        )
        jobs = [_to_job(i) for i in got.get("Items", [])]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def drop_job(self, tenant_id: str, job_id: str) -> bool:
        """Delete one job row.

        Deleted rather than superseded: a job is bookkeeping about a pipeline run, not a claim
        about the world, so there is no belief here to preserve. What the run *concluded* lives in
        the assertions, which are superseded rather than removed.
        """
        self.table.delete_item(Key={"PK": tenant_pk(tenant_id), "SK": f"{JOB}{job_id}"})
        return True

    def drop_tenant(self, tenant_id: str) -> int:
        """Delete every job row for a tenant.

        Batched deletes rather than a TTL wait: a reset is meant to be observable
        immediately, and `expires_at` is up to 30 days out.
        """
        deleted = 0
        got = self.table.query(
            KeyConditionExpression="PK = :pk AND begins_with(SK, :sk)",
            ExpressionAttributeValues={":pk": tenant_pk(tenant_id), ":sk": JOB},
        )
        for item in got.get("Items", []):
            self.table.delete_item(Key={"PK": item["PK"], "SK": item["SK"]})
            deleted += 1
        return deleted


class JobTracker:
    """Advances a job and persists it in one call.

    Exists so no call site can advance a job in memory and forget to write it — the
    failure that makes a resumable pipeline unresumable. `fail` records the reason
    because a failed state without one is refused by the model.
    """

    def __init__(self, store: InMemoryJobStore | DynamoJobStore) -> None:
        self.store = store

    def open(self, job: IngestJob) -> IngestJob:
        self.store.put_job(job)
        return job

    def advance(self, job: IngestJob, state: JobState, *, reason: str | None = None) -> IngestJob:
        job.advance(state, reason=reason)
        self.store.put_job(job)
        return job

    def fail(self, job: IngestJob, state: JobState, reason: str) -> IngestJob:
        """Record a failure, tolerating a job that cannot legally reach it.

        A failure path must not raise a second exception on top of the first — the
        reason is the only thing that makes the job diagnosable, so it is persisted
        even when the transition is illegal from the current state.
        """
        try:
            job.advance(state, reason=reason)
        except Exception as e:
            logger.warning("could not record %s on %s: %s", state.value, job.job_id, e)
            job.reason = reason
        self.store.put_job(job)
        return job
