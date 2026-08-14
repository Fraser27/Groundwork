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
from collections.abc import Callable
from typing import Any, Protocol

from src.documents.models import IngestJob, JobState

logger = logging.getLogger(__name__)

#: Must match the index name in `cdk/lib/data-stack.ts`.
GSI1 = "GSI1"

JOB = "JOB#"

#: Jobs are bookkeeping over an S3 object that is still there, so expiring them costs a
#: re-run and nothing else. Long enough that a failure is still diagnosable next week.
JOB_TTL_SECONDS = 30 * 24 * 3600


class TableLike(Protocol):
    """The slice of a boto3 DynamoDB `Table` resource this module uses.

    No `scan`, deliberately: a scan on a status poll would degrade with tenant size.
    """

    def put_item(self, **kwargs: Any) -> dict[str, Any]: ...
    def get_item(self, **kwargs: Any) -> dict[str, Any]: ...
    def query(self, **kwargs: Any) -> dict[str, Any]: ...


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
