"""Create a tenant with its first admin, and delete one back to a reusable id.

**Why this is hard where `documents.wipe` is deliberately soft.** Wiping a document
supersedes its facts rather than removing them, because the firm still exists and an
`as_of` read has to reconstruct what it relied on when advice was given. Deleting a tenant
removes the firm, so there is nobody left with standing to read that history and nothing
for a soft delete to preserve it *for*. A tenant id is also reusable, which makes anything
left behind worse than untidy: the next tenant with that id inherits it.

**Creation is one act on purpose.** A tenant with no users is unreachable -- nobody can
sign in to it -- so creating one without an admin produces a namespace and no way to use
it. The email is therefore mandatory rather than optional.

**Cognito goes first on the way in and on the way out.** On creation it is the only step
whose constraint we do not own: email is the pool-wide username and `custom:tenant_id` is
immutable, so an address already in use cannot be moved and the request must fail before
anything else is written. On deletion the same call goes first for the opposite reason --
while identities still work, a user could sign in and upload into a tenant that is being
swept, racing the delete. Orphaned bytes with no reader is the safer failure.

**Nothing aborts.** Every stage records its failure and the next one runs, because a delete
that stops halfway leaves a tenant that is neither usable nor gone. The report says what
did not happen and the whole operation is idempotent, so the answer to a partial failure is
to run it again.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from src.admin_ops import ResetScope, reset_derived
from src.constants import DEFAULT_ONTOLOGY_PACK
from src.graph.scope import AuthContext, is_valid_tenant_id
from src.tenant_registry import TenantRecord

logger = logging.getLogger(__name__)


class TenantLifecycleError(RuntimeError):
    """The request cannot proceed. Raised before anything is written."""


class TenantExists(TenantLifecycleError):
    """Something already occupies this id."""


@dataclass
class CreateTenantReport:
    tenant_id: str
    admin_email: str
    admin_user_id: str = ""
    ontology_domain: str = DEFAULT_ONTOLOGY_PACK
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "admin_email": self.admin_email,
            "admin_user_id": self.admin_user_id,
            "ontology_domain": self.ontology_domain,
            "note": self.note,
        }


@dataclass
class DeleteTenantReport:
    """What the cascade removed, and what it could not.

    Counts are per store rather than a single total, because "the graph is empty but eleven
    grants survived" and "everything went" must not render the same.
    """

    tenant_id: str
    users_deleted: int = 0
    groups_deleted: int = 0
    assertions_dropped: int = 0
    vectors_dropped: int = 0
    jobs_dropped: int = 0
    grants_dropped: int = 0
    graph_audit_dropped: int = 0
    query_audit_dropped: int = 0
    documents_erased: int = 0
    settings_deleted: bool = False
    tombstoned: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        """Whether the slate is actually clean.

        Keyed off `errors` rather than off any count, because `reset_derived` *reports* a
        store that cannot drop a tenant instead of raising -- so a zero count can mean
        "nothing there" or "nothing removed", and only the error list separates them.
        """
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "complete": self.complete,
            "users_deleted": self.users_deleted,
            "groups_deleted": self.groups_deleted,
            "assertions_dropped": self.assertions_dropped,
            "vectors_dropped": self.vectors_dropped,
            "jobs_dropped": self.jobs_dropped,
            "grants_dropped": self.grants_dropped,
            "graph_audit_dropped": self.graph_audit_dropped,
            "query_audit_dropped": self.query_audit_dropped,
            "documents_erased": self.documents_erased,
            "settings_deleted": self.settings_deleted,
            "tombstoned": self.tombstoned,
            "errors": self.errors,
            "note": (
                "Deleted. Nothing here is recoverable: the documents were erased from S3 "
                "including every version, and both audit logs are gone."
                if self.complete
                else "Incomplete. What remains is listed in errors, and running this again "
                "is safe: every step is idempotent."
            ),
        }


def create_tenant(
    services: Any,
    actor: AuthContext,
    *,
    tenant_id: str,
    admin_email: str,
    name: str = "",
    ontology_domain: str = DEFAULT_ONTOLOGY_PACK,
) -> CreateTenantReport:
    """Create a tenant and invite its first admin. Both, or neither."""
    tenant_id = tenant_id.strip().lower()
    admin_email = admin_email.strip().lower()

    if not is_valid_tenant_id(tenant_id):
        raise TenantLifecycleError(
            f"{tenant_id!r} is not a valid tenant id. Lowercase letters, digits and hyphens, "
            "2 to 63 characters, starting with a letter or digit -- the id reaches S3 prefixes "
            "and a vector index name, so it is validated rather than escaped."
        )
    if not admin_email:
        raise TenantLifecycleError(
            "an admin email is required. A tenant with no users cannot be signed in to, so "
            "creating one without an admin produces a namespace nobody can reach."
        )
    if ontology_domain:
        from src.ontology.loader import available_domains

        if ontology_domain not in available_domains():
            raise TenantLifecycleError(
                f"no ontology pack {ontology_domain!r}; available: {available_domains()}"
            )

    registry = services.tenant_registry
    directory = services.tenant_directory

    # Whether the id is taken is checked before whether the deployment can service the request.
    # Both refuse, but they refuse different things: "this id is occupied" is about the request
    # and stays true everywhere, while a missing store is a deployment fact. Reporting the
    # latter first told a caller their configuration was broken when their id simply collided.
    if registry is not None:
        existing = registry.get(tenant_id)
        if existing is not None:
            raise TenantExists(
                f"tenant {tenant_id!r} already exists"
                if existing.is_live
                else f"tenant {tenant_id!r} was deleted on {existing.deleted_at}. Reusing the "
                "id would give its people a graph built by someone else; choose another."
            )
    if directory is not None and directory.users_for_tenant(tenant_id):
        raise TenantExists(
            f"users are already bound to {tenant_id!r} without a tenant record, so something "
            "occupies this id. Investigate rather than overwrite."
        )

    if registry is None or directory is None:
        raise TenantLifecycleError(
            "tenant creation needs both the registry and the directory, and one is not "
            "configured, so a tenant would exist with no way to bind a user to it"
        )

    admin = services.user_admin
    if admin is None:
        raise TenantLifecycleError(
            "no user pool is configured, so the tenant's admin cannot be invited"
        )

    # Cognito first: the only constraint here we do not own. Nothing is written when it fails.
    from src.user_admin import UserAdminError

    try:
        entry = admin.create_user(
            email=admin_email,
            tenant_id=tenant_id,
            admin_sub=actor.user_id,
            is_admin=True,
        )
    except UserAdminError as e:
        raise TenantLifecycleError(
            f"{e}. An email address is the username for the whole user pool and a user's tenant "
            "is fixed when the account is made, so an existing account cannot be moved into a "
            "new tenant -- invite a different address."
        ) from e

    directory.put_user(entry.user_id, tenant_id, email=entry.email)
    # So the new admin can invite their own people. Idempotent.
    try:
        admin.ensure_owner_group(entry.user_id)
    except UserAdminError as e:
        logger.warning("could not create the ownership group for %s: %s", entry.user_id, e)

    if services.governance_store is not None:
        from src.governance import GovernanceSettings

        settings = GovernanceSettings.from_env()
        settings.ontology_domain = ontology_domain
        settings.updated_by = actor.user_id
        services.governance_store.put(tenant_id, settings)
        services.governance.pop(tenant_id, None)

    registry.put(
        TenantRecord(
            tenant_id=tenant_id,
            name=name or tenant_id,
            ontology_domain=ontology_domain,
            created_at=datetime.now(UTC).isoformat(),
            created_by=actor.user_id,
        )
    )
    logger.info("created tenant %s with admin %s by %s", tenant_id, admin_email, actor.user_id)

    return CreateTenantReport(
        tenant_id=tenant_id,
        admin_email=entry.email,
        admin_user_id=entry.user_id,
        ontology_domain=ontology_domain,
        note=(
            "Cognito has emailed a temporary password. They must change it at first sign-in, "
            f"and their tenant is fixed at {tenant_id} and cannot be changed."
        ),
    )


def delete_tenant(services: Any, actor: AuthContext, tenant_id: str) -> DeleteTenantReport:
    """Remove a tenant and everything belonging to it. Ordered so a failure is recoverable."""
    report = DeleteTenantReport(tenant_id=tenant_id)
    directory = services.tenant_directory
    if directory is None:
        raise TenantLifecycleError(
            "no tenant directory is configured, so this tenant's users cannot be enumerated. "
            "Deleting the data while accounts survive would leave people able to sign in to a "
            "tenant that no longer holds anything."
        )

    ctx = AuthContext(user_id=actor.user_id, tenant_id=tenant_id)

    # ── 0. Inventory, before anything is dropped ────────────────────────────
    #
    # `users_for_tenant` is a GSI query. `user_admin.list_tenant_users` pages the whole pool
    # once and filters in process, so it can return fewer of a tenant's users than exist --
    # and a missed user is an account that still signs in after the delete.
    users = list(directory.users_for_tenant(tenant_id))
    matter_ids: list[str] = []
    if services.graph is not None:
        try:
            from src.matters import MatterStore

            matter_ids = [m.matter_id for m in MatterStore(services.graph).list(ctx)]
        except Exception as e:  # noqa: BLE001
            # Recorded rather than fatal: matters only widen the grant sweep, and refusing to
            # delete because the list is unreadable would leave everything standing.
            report.errors.append(f"could not list matters, grants may be incomplete: {e}")

    # ── 1. Identity first ───────────────────────────────────────────────────
    admin = services.user_admin
    for user in users:
        if not user.email:
            report.errors.append(f"user {user.sub} has no stored email, so it was not deleted")
            continue
        try:
            if admin is not None:
                admin.delete_user(user.email, tenant_id=tenant_id, directory=directory)
            else:
                directory.forget_user(user.sub)
            report.users_deleted += 1
        except Exception as e:  # noqa: BLE001
            report.errors.append(f"user {user.email}: {e}")
        if admin is not None:
            try:
                if admin.delete_owner_group(user.sub):
                    report.groups_deleted += 1
            except Exception as e:  # noqa: BLE001
                report.errors.append(f"ownership group for {user.email}: {e}")

    # ── 2. Settings and per-process caches ──────────────────────────────────
    #
    # Early, so a half-deleted tenant is not still answering under its own governance -- the
    # kill switch and the trust floor both live here.
    if services.governance_store is not None:
        try:
            services.governance_store.delete(tenant_id)
            report.settings_deleted = True
        except Exception as e:  # noqa: BLE001
            report.errors.append(f"governance settings: {e}")
    services.governance.pop(tenant_id, None)
    services.blocked_queries.pop(tenant_id, None)

    # ── 3. Derived data: graph, vectors, jobs, catalog, routing ─────────────
    #
    # Metrics included. They are authored work with no upstream source, which is why
    # `reset_derived` spares them by default -- but the firm that authored them is going.
    try:
        derived = reset_derived(services, ctx, ResetScope(metrics=True))
        report.assertions_dropped = derived.assertions_dropped
        report.vectors_dropped = derived.vectors_dropped
        report.jobs_dropped = derived.jobs_dropped
        report.errors.extend(derived.errors)
    except Exception as e:  # noqa: BLE001
        report.errors.append(f"derived data: {e}")

    # ── 4. Access grants, screens and the access audit ──────────────────────
    store = getattr(services.access, "store", None)
    drop_grants = getattr(store, "drop_tenant", None)
    if drop_grants is None:
        report.errors.append("access store cannot drop a tenant; grants and screens remain")
    else:
        try:
            report.grants_dropped = drop_grants(
                tenant_id, user_ids=[u.sub for u in users], matter_ids=matter_ids
            )
        except Exception as e:  # noqa: BLE001
            report.errors.append(f"access grants: {e}")

    # ── 5. Audit logs ───────────────────────────────────────────────────────
    #
    # After the data, because until this point the audit was the only record of what the
    # delete had done. Both go: the user asked for a clean slate, and a log describing a firm
    # that no longer exists has no reader.
    audits = (("graph audit", services.graph_audit), ("query audit", services.query_audit))
    for label, audit in audits:
        drop = getattr(audit, "drop_tenant", None)
        if audit is None or drop is None:
            continue
        try:
            count = drop(tenant_id)
            if label == "graph audit":
                report.graph_audit_dropped = count
            else:
                report.query_audit_dropped = count
        except Exception as e:  # noqa: BLE001
            report.errors.append(f"{label}: {e}")

    # ── 6. S3 last ──────────────────────────────────────────────────────────
    #
    # The only step nothing reconstructs. Last so that a failure anywhere above leaves a
    # tenant which is still re-deletable and whose graph can still be rebuilt from these
    # bytes; erasing them first would make a later failure permanent.
    from src.documents.storage import storage_from_config

    storage = storage_from_config(services.config)
    if storage is not None:
        try:
            report.documents_erased = storage.drop_tenant(tenant_id)
        except Exception as e:  # noqa: BLE001
            report.errors.append(f"documents: {e}")

    # ── 7. Tombstone ────────────────────────────────────────────────────────
    registry = services.tenant_registry
    if registry is not None:
        try:
            report.tombstoned = registry.tombstone(tenant_id, actor=actor.user_id) is not None
        except Exception as e:  # noqa: BLE001
            report.errors.append(f"tenant record: {e}")

    logger.info(
        "deleted tenant %s by %s: complete=%s errors=%d",
        tenant_id,
        actor.user_id,
        report.complete,
        len(report.errors),
    )
    return report
