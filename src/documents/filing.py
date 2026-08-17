"""Which matter an ingest files a document under.

A document id is the hash of its bytes, so a re-upload is the *same* document arriving
again. An ingest that named no matter filed its facts under none, which silently un-filed a
document somebody had already filed -- and matter access is allowlist-primary, so an unfiled
fact is readable by everyone in the tenant, including people screened from the matter it
belongs to. Relinking is point-in-time and nothing re-applies it, so every re-ingest was an
access widening nobody asked for.

The rule: an upload that names a matter is believed; one that does not adopts whatever the
document is already filed under.

Explicit wins because a user choosing a matter during upload is expressing intent, and the
other way round would make a deliberate re-filing inexpressible through the only surface
that files documents at all. It is also the only one of the two that *changes* access, so it
is the one that is audited -- adoption restores the filing that was already in force, and
is logged rather than recorded as a change.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Filing:
    """The matter an ingest will use, and how it got there."""

    matter_id: str | None
    requested: str | None
    previous: str | None

    @property
    def adopted(self) -> bool:
        return self.requested is None and self.previous is not None

    @property
    def overrode(self) -> bool:
        """An upload naming a different matter than the document is filed under.

        The access change: allowlist-primary means the old matter's team loses the facts and
        the new matter's team gains them, effected by an upload rather than by a link.
        """
        return (
            self.requested is not None
            and self.previous is not None
            and self.requested != self.previous
        )


def resolve_filing(
    store: Any | None,
    tenant_id: str,
    document_id: str,
    requested: str | None,
) -> Filing:
    previous = _current_filing(store, tenant_id, document_id)
    named = requested or None
    return Filing(matter_id=named or previous, requested=named, previous=previous)


def _current_filing(store: Any | None, tenant_id: str, document_id: str) -> str | None:
    """The matter this document is already filed under: the newest prior job that names one.

    The newest that *names* one, not simply the newest. Nothing files a document to no matter
    deliberately -- `link_documents` only ever moves one into a matter -- so a null on a later
    job is the bug this module undoes rather than a decision to respect, and reading it that
    way also recovers documents already un-filed in production.
    """
    if store is None:
        return None
    try:
        jobs = store.jobs_for_document(tenant_id, document_id)
    except Exception as e:  # noqa: BLE001
        # An unreachable job store must not fail an ingest: the bytes are stored and the facts
        # still land. This degrades to the old behaviour, loudly.
        logger.warning("could not read prior filings of %s: %s", document_id, e)
        return None
    filed = [j for j in jobs if j.matter_id]
    if not filed:
        return None
    return max(filed, key=lambda j: j.created_at).matter_id


def apply_filing(doc: Any, filing: Filing) -> Any:
    """Put the resolved matter back on the `DocumentMeta` storage handed us.

    `put_document` returns a cached record on a re-ingest, and that record carries the matter of
    the *first* upload -- which for a document filed later is None. Believing it would reinstate
    the un-filing this module exists to prevent, one process-lifetime at a time.
    """
    if doc.matter_id == filing.matter_id:
        return doc
    return doc.model_copy(update={"matter_id": filing.matter_id})


def log_filing(filing: Filing, document_id: str, actor: str) -> None:
    if filing.adopted:
        logger.info(
            "%s re-ingested %s naming no matter, keeping %s", actor, document_id, filing.previous
        )
    elif filing.overrode:
        logger.warning(
            "%s re-ingested %s as %s, moving it off %s",
            actor,
            document_id,
            filing.requested,
            filing.previous,
        )


def audit_filing(
    audit: Any | None,
    *,
    tenant_id: str,
    actor: str,
    document_id: str,
    filing: Filing,
    reason: str | None = None,
) -> None:
    """Record an upload that moved a document between matters. Never raises.

    `LINK_DOCUMENTS` rather than a new action: the vocabulary is closed on purpose, and what
    happened is exactly what that action describes -- a document moved between matters. The
    detail shape mirrors `matters._audit_link` so the Audit page renders both the same.

    No assertion ids, because extraction has not run yet: the facts this affects do not exist
    at the moment the decision is taken. `affected` reads 0, which is honest.
    """
    if not filing.overrode:
        return
    if audit is None:
        logger.error("no audit log configured: %s moved %s by re-upload", actor, document_id)
        return

    from src.graph_audit import LINK_DOCUMENTS, GraphEvent

    try:
        audit.append(
            GraphEvent(
                tenant_id=tenant_id,
                actor=actor,
                action=LINK_DOCUMENTS,
                document_id=document_id,
                matter_id=filing.matter_id,
                reason=reason or "re-uploaded naming a different matter",
                detail={
                    "documents": [document_id],
                    "previous_matters": {document_id: filing.previous},
                    "via": "ingest",
                },
            )
        )
    except Exception as e:  # noqa: BLE001
        logger.error("AUDIT FAILED, %s moved %s by re-upload unrecorded: %s", actor, document_id, e)
