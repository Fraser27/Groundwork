"""Shapes that move through the unstructured pipeline.

`DocumentMeta` is a pointer into S3 plus the content hash — S3 is the source of
truth, so this is the only authoritative object here. `Chunk` and `IngestJob`
describe a derived index that can be deleted and rebuilt, which is why identity for
both is content-addressed: rebuilding produces the same ids, so a re-run is a no-op
rather than a duplicate.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.graph.assertions import SourceLocator


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_hex(data: str | bytes) -> str:
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()


class JobState(str, Enum):
    """Where an ingest job is, including how it failed.

    Failure states are terminal for the current attempt but recoverable: each maps
    to the phase that must be re-run (`retry_target`). Nothing is lost by retrying
    because S3 still holds the bytes and every derived id is content-addressed.
    """

    REGISTERED = "REGISTERED"
    FETCHING = "FETCHING"
    PARSING = "PARSING"
    CHUNKING = "CHUNKING"
    EXTRACTING = "EXTRACTING"
    EMBEDDING = "EMBEDDING"
    GRAPH_STAGED = "GRAPH_STAGED"
    PENDING_REVIEW = "PENDING_REVIEW"
    APPROVED = "APPROVED"
    LIVE = "LIVE"

    FETCH_FAILED = "FETCH_FAILED"
    PARSE_FAILED = "PARSE_FAILED"
    CHUNK_FAILED = "CHUNK_FAILED"
    EXTRACT_FAILED = "EXTRACT_FAILED"
    EMBED_FAILED = "EMBED_FAILED"
    STAGE_FAILED = "STAGE_FAILED"
    PROMOTE_FAILED = "PROMOTE_FAILED"

    @property
    def is_failed(self) -> bool:
        return self in _RETRY_TARGET

    @property
    def is_terminal(self) -> bool:
        return not _TRANSITIONS[self]

    @property
    def retry_target(self) -> JobState | None:
        return _RETRY_TARGET.get(self)


#: Which phase a failure must restart from. STAGE_FAILED restarts at EXTRACTING
#: rather than at staging: extractions are not persisted between attempts, and
#: re-deriving them is cheap and deterministic.
_RETRY_TARGET: dict[JobState, JobState] = {
    JobState.FETCH_FAILED: JobState.FETCHING,
    JobState.PARSE_FAILED: JobState.PARSING,
    JobState.CHUNK_FAILED: JobState.CHUNKING,
    JobState.EXTRACT_FAILED: JobState.EXTRACTING,
    JobState.EMBED_FAILED: JobState.EMBEDDING,
    JobState.STAGE_FAILED: JobState.EXTRACTING,
    JobState.PROMOTE_FAILED: JobState.APPROVED,
}

_TRANSITIONS: dict[JobState, frozenset[JobState]] = {
    JobState.REGISTERED: frozenset({JobState.FETCHING, JobState.FETCH_FAILED}),
    JobState.FETCHING: frozenset({JobState.PARSING, JobState.FETCH_FAILED}),
    JobState.PARSING: frozenset({JobState.CHUNKING, JobState.PARSE_FAILED}),
    JobState.CHUNKING: frozenset({JobState.EXTRACTING, JobState.CHUNK_FAILED}),
    JobState.EXTRACTING: frozenset({JobState.EMBEDDING, JobState.EXTRACT_FAILED}),
    JobState.EMBEDDING: frozenset(
        {JobState.GRAPH_STAGED, JobState.EMBED_FAILED, JobState.STAGE_FAILED}
    ),
    JobState.GRAPH_STAGED: frozenset({JobState.PENDING_REVIEW, JobState.STAGE_FAILED}),
    JobState.PENDING_REVIEW: frozenset({JobState.APPROVED}),
    JobState.APPROVED: frozenset({JobState.LIVE, JobState.PROMOTE_FAILED}),
    JobState.LIVE: frozenset(),
    **{failed: frozenset({target}) for failed, target in _RETRY_TARGET.items()},
}


class IllegalTransition(ValueError):
    """Raised when a job is moved to a state it cannot reach from its current one."""


class StateChange(BaseModel):
    model_config = ConfigDict(frozen=True)

    state: JobState
    at: str = Field(default_factory=now_iso)
    reason: str | None = None


class DocumentMeta(BaseModel):
    """An immutable object in S3, plus enough to re-fetch and verify it."""

    model_config = ConfigDict(frozen=True)

    tenant_id: str
    bucket: str
    key: str
    filename: str
    media_type: str
    content_sha256: str
    byte_size: int
    uploaded_by: str
    matter_id: str | None = None
    uploaded_at: str = Field(default_factory=now_iso)
    s3_version_id: str | None = None
    page_count: int | None = None
    document_id: str = ""

    @model_validator(mode="after")
    def _derive_id(self) -> DocumentMeta:
        if not self.document_id:
            # Tenant is in the hash: identical bytes uploaded by two firms are two
            # documents, or a shared graph would fuse them.
            digest = sha256_hex(f"{self.tenant_id}:{self.content_sha256}")[:24]
            object.__setattr__(self, "document_id", f"doc-{digest}")
        return self

    @property
    def s3_uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"

    @property
    def entity_id(self) -> str:
        """Graph node id for this document."""
        return f"document:{self.document_id}"


class Chunk(BaseModel):
    """A span of document text with its exact position in the parsed text.

    `char_start`/`char_end` index the whole-document text, not the page and not the
    chunk — a stored assertion has to resolve back to one unambiguous span years
    later. The validator enforces `text == parsed_text[char_start:char_end]` by
    length, which is what stops a well-meaning `.strip()` from silently shifting
    every offset downstream.
    """

    model_config = ConfigDict(frozen=True)

    document_id: str
    tenant_id: str
    ordinal: int
    page: int
    char_start: int
    char_end: int
    text: str
    matter_id: str | None = None
    filename: str | None = None
    """Carried through from `DocumentMeta` so a locator can name the file a reviewer
    has to open. Optional because a chunk is still valid without it."""

    span_sha256: str = ""
    chunk_id: str = ""

    @model_validator(mode="after")
    def _check_offsets(self) -> Chunk:
        if self.char_start < 0:
            raise ValueError(f"char_start must be >= 0, got {self.char_start}")
        if self.char_end - self.char_start != len(self.text):
            raise ValueError(
                f"offsets {self.char_start}..{self.char_end} do not match text of "
                f"length {len(self.text)}, chunking lost or trimmed a span"
            )
        if self.page < 1:
            raise ValueError(f"page is 1-based, got {self.page}")
        if not self.span_sha256:
            object.__setattr__(self, "span_sha256", sha256_hex(self.text))
        if not self.chunk_id:
            object.__setattr__(
                self,
                "chunk_id",
                f"{self.document_id}:p{self.page}:{self.char_start}-{self.char_end}",
            )
        return self

    def to_locator(
        self, char_start: int | None = None, char_end: int | None = None
    ) -> SourceLocator:
        """Locator for this chunk, or for a narrower span inside it.

        The span selects the `quote`, which is the citation itself — file, page and a
        verbatim sentence is what a lawyer would use by hand, and needs no coordinate
        mapping to resolve. The offsets travel too, but only as debug metadata: they
        index the extracted text buffer rather than the PDF.
        """
        start = self.char_start if char_start is None else char_start
        end = self.char_end if char_end is None else char_end
        if not (self.char_start <= start <= end <= self.char_end):
            raise ValueError(f"span {start}..{end} is outside chunk {self.chunk_id}")
        quote = self.text[start - self.char_start : end - self.char_start]
        return SourceLocator(
            document_id=self.document_id,
            filename=self.filename,
            page=self.page,
            chunk_id=self.chunk_id,
            quote=quote,
            span_sha256=sha256_hex(quote),
            char_start=start,
            char_end=end,
        )


class IngestJob(BaseModel):
    """One pass of the pipeline over one document.

    Append-only history: every state a job passed through, with the failure reason
    if it failed there. A job that succeeded on its third attempt should still show
    the first two.
    """

    job_id: str
    document_id: str
    tenant_id: str
    matter_id: str | None = None
    state: JobState = JobState.REGISTERED
    reason: str | None = None
    attempts: dict[str, int] = Field(default_factory=dict)
    history: list[StateChange] = Field(default_factory=list)
    chunk_count: int = 0
    staged_assertion_ids: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)

    @classmethod
    def for_document(cls, doc: DocumentMeta) -> IngestJob:
        job = cls(
            job_id=f"job-{sha256_hex(f'{doc.document_id}:{now_iso()}')[:20]}",
            document_id=doc.document_id,
            tenant_id=doc.tenant_id,
            matter_id=doc.matter_id,
        )
        job.history.append(StateChange(state=JobState.REGISTERED))
        return job

    def advance(self, state: JobState, *, reason: str | None = None) -> IngestJob:
        if state not in _TRANSITIONS[self.state]:
            allowed = sorted(s.value for s in _TRANSITIONS[self.state])
            raise IllegalTransition(f"{self.state.value} -> {state.value} (allowed: {allowed})")
        if state.is_failed and not reason:
            raise IllegalTransition(f"{state.value} requires a reason")
        self.state = state
        self.reason = reason if state.is_failed else None
        self.attempts[state.value] = self.attempts.get(state.value, 0) + 1
        self.updated_at = now_iso()
        self.history.append(StateChange(state=state, at=self.updated_at, reason=reason))
        return self

    def retry(self) -> IngestJob:
        target = self.state.retry_target
        if target is None:
            raise IllegalTransition(f"{self.state.value} is not a recoverable failure")
        return self.advance(target)
