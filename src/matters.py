"""Matters as records, and moving documents between them.

A matter exists the moment somebody creates it, before any document is filed. That ordering is the
point: a team is staffed and an ethical screen is raised *before* the first document arrives, and
neither is expressible against a matter that does not exist yet.

Relinking is a property update rather than a rewrite, because `matter_id` is deliberately absent
from the assertion hash. Re-filing a document keeps every assertion id, so nothing forks and no
citation moves.

**Every link and relink is audited.** Moving a document changes *who can read its facts* -- matter
access is allowlist-primary, so a document moved into a matter somebody is not on becomes invisible
to them, and moved out of a screened matter becomes visible. That is an access change effected
through a data operation, so it belongs in the record with who did it and when.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from src.graph import matter_queries as q
from src.graph.scope import AuthContext

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class MatterError(ValueError):
    """A matter could not be created, found, or linked. Message is safe to show a user."""


@dataclass(frozen=True)
class Matter:
    """A matter record. Name only for now, which is all the UI needs to be usable."""

    matter_id: str
    name: str
    created_at: str = ""
    created_by: str = ""
    updated_at: str = ""
    updated_by: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "matter_id": self.matter_id,
            "name": self.name,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "updated_at": self.updated_at,
            "updated_by": self.updated_by,
        }


@dataclass
class LinkReport:
    """What a bulk link moved."""

    matter_id: str
    documents: tuple[str, ...] = ()
    assertions_relinked: int = 0
    previous_matters: dict[str, str | None] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matter_id": self.matter_id,
            "documents": list(self.documents),
            "assertions_relinked": self.assertions_relinked,
            "previous_matters": dict(self.previous_matters),
            "errors": self.errors,
            "at": self.at,
            "note": (
                "Assertion ids are unchanged: the matter is not part of a fact's identity, so "
                "re-filing a document keeps every citation intact. This changes who can read "
                "these facts, so it is recorded on the Audit page."
            ),
        }


def _to_matter(node: dict[str, Any]) -> Matter:
    return Matter(
        matter_id=str(node.get("matter_id", "")),
        name=str(node.get("name") or node.get("matter_id", "")),
        created_at=str(node.get("created_at") or ""),
        created_by=str(node.get("created_by") or ""),
        updated_at=str(node.get("updated_at") or ""),
        updated_by=str(node.get("updated_by") or ""),
    )


class MatterStore:
    """Matter records in the graph."""

    def __init__(self, graph: Any) -> None:
        self.graph = graph

    def create(self, ctx: AuthContext, matter_id: str, name: str) -> Matter:
        """Create or rename a matter.

        An existing id is an update rather than an error: two people setting up the same matter
        should converge on one record instead of one of them being refused.
        """
        matter_id = (matter_id or "").strip()
        if not matter_id:
            raise MatterError("a matter needs a reference, for example NTL-2026-0114")
        if not name or not name.strip():
            raise MatterError("a matter needs a name, so a list of references is readable")

        rows = self.graph.query(
            q.UPSERT_MATTER,
            {
                "tenant_id": ctx.tenant_id,
                "matter_id": matter_id,
                "name": name.strip(),
                "at": _now(),
                "actor": ctx.user_id,
            },
        )
        logger.info("%s created matter %s for %s", ctx.user_id, matter_id, ctx.tenant_id)
        return _to_matter(rows[0]["m"]) if rows else Matter(matter_id=matter_id, name=name.strip())

    def list(self, ctx: AuthContext) -> list[Matter]:
        """Every matter record for the tenant.

        Not filtered by the matter wall here. The caller decides: an administrator managing a
        screen has to see the matter the screen applies to, while a lawyer's matter list does not.
        """
        rows = self.graph.query(q.LIST_MATTERS, {"tenant_id": ctx.tenant_id})
        return [_to_matter(r["m"]) for r in rows]

    def get(self, ctx: AuthContext, matter_id: str) -> Matter | None:
        rows = self.graph.query(q.GET_MATTER, {"tenant_id": ctx.tenant_id, "matter_id": matter_id})
        return _to_matter(rows[0]["m"]) if rows else None

    def exists(self, ctx: AuthContext, matter_id: str) -> bool:
        """Whether a matter is real, for the check an upload depends on.

        This is what stops a typo becoming a matter: without it `NTL-2026-114` and
        `NTL-2026-0114` are two matters and nothing notices, because the list used to be whatever
        the data happened to contain.
        """
        return self.get(ctx, matter_id) is not None

    def delete(self, ctx: AuthContext, matter_id: str) -> None:
        """Remove the record. Its facts keep their `matter_id`, so this loses the name and
        nothing else -- withdrawing the facts is `wipe_matter`, which is a separate, louder act."""
        self.graph.write(q.DELETE_MATTER, {"tenant_id": ctx.tenant_id, "matter_id": matter_id})
        logger.info("%s deleted matter record %s", ctx.user_id, matter_id)


def link_documents(
    services: Any,
    ctx: AuthContext,
    matter_id: str,
    document_ids: list[str],
    *,
    reason: str | None = None,
) -> LinkReport:
    """File documents under a matter, in bulk, and record it.

    Refuses an unknown matter rather than creating one, because a link that silently invents a
    matter is how a typo becomes a permanent second matter that a conflict check then misses.

    Both copies of `matter_id` are written -- node and edge -- because a traversal filters on the
    edge. Updating only the node would leave a fact readable by the old matter's team and
    invisible to the new one, which is a silent access change.
    """
    if not document_ids:
        raise MatterError("no documents were selected")

    store = MatterStore(services.graph)
    if not store.exists(ctx, matter_id):
        raise MatterError(
            f"no matter {matter_id!r}. Create it first: filing documents under a matter that "
            "does not exist is how a typo becomes a second matter nobody queries."
        )
    ctx.assert_can_read_matter(matter_id)

    report = LinkReport(matter_id=matter_id, documents=tuple(document_ids))

    # Recorded before the move: afterwards the old matter is unrecoverable from the data, and
    # "where did this document come from" is exactly what somebody asks later.
    for document_id in document_ids:
        previous = _current_matter(services, ctx, document_id)
        report.previous_matters[document_id] = previous
        if previous is not None:
            ctx.assert_can_read_matter(previous)

    for document_id in document_ids:
        try:
            rows = services.graph.query(
                q.RELINK_DOCUMENT_ASSERTIONS,
                {
                    "tenant_id": ctx.tenant_id,
                    "document_id": document_id,
                    "matter_id": matter_id,
                },
            )
            report.assertions_relinked += int(rows[0]["updated"]) if rows else 0
        except Exception as e:
            # Reported per document rather than raised: a batch where one document fails should
            # move the rest and say which one did not.
            report.errors.append(f"{document_id}: {e}")

    _audit_link(services, ctx, report, reason)
    logger.info(
        "%s linked %d documents to %s, %d assertions relinked",
        ctx.user_id,
        len(document_ids),
        matter_id,
        report.assertions_relinked,
    )
    return report


def _current_matter(services: Any, ctx: AuthContext, document_id: str) -> str | None:
    """The matter a document's facts currently carry, or None."""
    for record in services.review_queue.visible(ctx):
        if record.assertion.source_locator.document_id == document_id:
            return record.assertion.matter_id
    return None


def _audit_link(services: Any, ctx: AuthContext, report: LinkReport, reason: str | None) -> None:
    """Record the link. Loud on failure, because an unrecorded access change is the bad outcome."""
    audit = getattr(services, "graph_audit", None)
    if audit is None:
        report.errors.append("no audit log configured: this link was not recorded")
        return

    from src.graph_audit import LINK_DOCUMENTS, GraphEvent

    try:
        audit.append(
            GraphEvent(
                tenant_id=ctx.tenant_id,
                actor=ctx.user_id,
                action=LINK_DOCUMENTS,
                at=report.at,
                matter_id=report.matter_id,
                reason=reason,
                detail={
                    "documents": list(report.documents),
                    "previous_matters": dict(report.previous_matters),
                    "assertions_relinked": report.assertions_relinked,
                },
            )
        )
    except Exception as e:
        report.errors.append(f"AUDIT FAILED, this link is unrecorded: {e}")
        logger.error("graph audit write failed for a matter link: %s", e)
