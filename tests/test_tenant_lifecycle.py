"""Creating a tenant, and deleting one without touching anyone else's data.

The assertion that matters most here is the boring-looking one: a second tenant is seeded in
every fake, and after a delete it is still whole. Everything else in this file is about
ordering and partial failure, which is what decides whether a half-finished delete leaves a
tenant that can still be recovered or one that cannot.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.access import AccessManager, InMemoryAccessStore, MatterAssignment, MatterScreen
from src.config import AuthConfig, GroundworkConfig
from src.documents.review import InMemoryAssertionStore, ReviewQueue
from src.governance_store import InMemoryGovernanceStore
from src.graph.scope import AuthContext
from src.ontology.loader import load_ontology
from src.tenant_lifecycle import (
    TenantExists,
    TenantLifecycleError,
    create_tenant,
    delete_tenant,
)
from src.tenant_registry import InMemoryTenantRegistry, TenantRecord

HOME = "demo-firm"
TARGET = "demo-clinic"
#: Seeded into every fake and never named in a request. Nothing may touch it.
BYSTANDER = "other-firm"


class FakeUserAdmin:
    def __init__(self, log: list[tuple[str, str]], *, existing_emails: set[str] | None = None):
        self.log = log
        self.existing = existing_emails or set()
        self.deleted: list[str] = []
        self.groups: list[str] = []
        self.list_users_raises = False

    def create_user(self, *, email, tenant_id, admin_sub, is_admin=False):
        from src.user_admin import UserAdminError

        if email in self.existing:
            raise UserAdminError(f"{email} already has an account")
        self.log.append(("cognito.create", tenant_id))
        return SimpleNamespace(user_id=f"sub-{email}", email=email, display_name=email, status="X")

    def ensure_owner_group(self, sub):
        self.log.append(("group.ensure", sub))
        return f"owner-{sub}"

    def delete_user(self, email, *, tenant_id, directory=None):
        self.log.append(("cognito.delete", email))
        self.deleted.append(email)
        if directory is not None:
            directory.forget_user(f"sub-{email}")

    def delete_owner_group(self, sub):
        self.groups.append(sub)
        return True

    def list_tenant_users(self, tenant_id, *, limit=60):
        """The unreliable enumeration source. Raises so a caller reaching for it is caught."""
        if self.list_users_raises:
            raise AssertionError("list_tenant_users pages once and must not drive a cascade")
        return []


class FakeDirectory:
    def __init__(self) -> None:
        self.rows: dict[str, tuple[str, str]] = {}

    def seed(self, sub: str, tenant: str, email: str) -> None:
        self.rows[sub] = (tenant, email)

    def users_for_tenant(self, tenant_id):
        return [
            SimpleNamespace(sub=s, tenant_id=t, email=e)
            for s, (t, e) in self.rows.items()
            if t == tenant_id
        ]

    def put_user(self, sub, tenant_id, email=""):
        self.rows[sub] = (tenant_id, email)

    def forget_user(self, sub):
        self.rows.pop(sub, None)


class FakeStorage:
    def __init__(self, log: list[tuple[str, str]], *, raises: bool = False):
        self.log = log
        self.raises = raises
        self.swept: list[str] = []

    def drop_tenant(self, tenant_id: str) -> int:
        self.log.append(("s3.drop", tenant_id))
        if self.raises:
            raise RuntimeError("access denied")
        self.swept.append(tenant_id)
        return 5


class FakeJobs:
    def __init__(self) -> None:
        self.dropped: list[str] = []

    def drop_tenant(self, tenant_id: str) -> int:
        self.dropped.append(tenant_id)
        return 4


class FakeCatalog:
    def __init__(self) -> None:
        self.cleared: list[str] = []

    def tables(self, tenant_id: str) -> list[int]:
        return [1, 2]

    def clear(self, tenant_id: str) -> None:
        self.cleared.append(tenant_id)


class FakeVectors:
    def __init__(self) -> None:
        self.dropped: list[str] = []

    def drop_tenant(self, tenant_id: str) -> int:
        self.dropped.append(tenant_id)
        return 9


class FakeAudit:
    def __init__(self) -> None:
        self.dropped: list[str] = []

    def drop_tenant(self, tenant_id: str) -> int:
        self.dropped.append(tenant_id)
        return 3


def _services(monkeypatch, *, storage_raises: bool = False, existing_emails=None) -> Any:
    log: list[tuple[str, str]] = []
    directory = FakeDirectory()
    # The bystander has a user, grants and a screen. None of it is named in any request.
    directory.seed("sub-bystander@x.example", BYSTANDER, "bystander@x.example")

    access_store = InMemoryAccessStore()
    for tenant in (TARGET, BYSTANDER):
        access_store.put_assignment(
            MatterAssignment(
                tenant_id=tenant,
                user_id=f"sub-{tenant}@x.example",
                matter_id="M-1",
                granted_by="admin",
                granted_at="2026-01-01T00:00:00Z",
            )
        )
        access_store.put_screen(
            MatterScreen(
                tenant_id=tenant,
                user_id=f"sub-{tenant}@x.example",
                matter_id="M-2",
                reason="conflict",
                screened_by="admin",
                screened_at="2026-01-01T00:00:00Z",
            )
        )

    registry = InMemoryTenantRegistry()
    registry.put(TenantRecord(tenant_id=BYSTANDER, name="Other", created_by="someone"))

    governance = InMemoryGovernanceStore()
    storage = FakeStorage(log, raises=storage_raises)
    monkeypatch.setattr("src.documents.storage.storage_from_config", lambda cfg: storage)

    services = SimpleNamespace(
        config=GroundworkConfig(environment="local", auth=AuthConfig(home_tenant=HOME)),
        ontology=load_ontology("legal"),
        tenant_registry=registry,
        tenant_directory=directory,
        user_admin=FakeUserAdmin(log, existing_emails=existing_emails),
        governance_store=governance,
        governance={},
        blocked_queries={},
        graph=None,
        access=AccessManager(access_store),
        review_queue=ReviewQueue(InMemoryAssertionStore()),
        graph_audit=FakeAudit(),
        query_audit=FakeAudit(),
        embedder=FakeVectors(),
        job_store=FakeJobs(),
        catalog=FakeCatalog(),
        router_indexer=None,
    )
    services.log = log
    services.storage = storage
    services.access_store = access_store
    return services


@pytest.fixture
def actor() -> AuthContext:
    return AuthContext(user_id="home-admin", tenant_id=HOME)


class TestCreatingATenant:
    def test_the_admin_is_invited_and_the_tenant_is_recorded(self, monkeypatch, actor):
        services = _services(monkeypatch)
        report = create_tenant(
            services,
            actor,
            tenant_id=TARGET,
            admin_email="Doc@Clinic.example",
            ontology_domain="healthcare",
        )
        assert report.tenant_id == TARGET
        # Lowercased, because an address is a username and case would fork the account.
        assert report.admin_email == "doc@clinic.example"
        assert services.tenant_registry.get(TARGET).ontology_domain == "healthcare"
        assert services.governance_store.get(TARGET).ontology_domain == "healthcare"
        assert [u.email for u in services.tenant_directory.users_for_tenant(TARGET)] == [
            "doc@clinic.example"
        ]

    def test_the_pack_the_tenant_asked_for_is_what_governs_it(self, monkeypatch, actor):
        """The reason the domain is settable at creation: it decides which closed vocabulary
        every later write is validated against."""
        services = _services(monkeypatch)
        create_tenant(
            services, actor, tenant_id=TARGET, admin_email="a@b.example", ontology_domain="fintech"
        )
        assert services.governance_store.get(TARGET).ontology_domain == "fintech"

    def test_an_unknown_pack_is_refused(self, monkeypatch, actor):
        services = _services(monkeypatch)
        with pytest.raises(TenantLifecycleError, match="no ontology pack"):
            create_tenant(
                services, actor, tenant_id=TARGET, admin_email="a@b.example", ontology_domain="zzz"
            )
        assert services.tenant_registry.get(TARGET) is None

    def test_an_invalid_id_is_refused_before_anything_is_written(self, monkeypatch, actor):
        services = _services(monkeypatch)
        with pytest.raises(TenantLifecycleError, match="not a valid tenant id"):
            create_tenant(services, actor, tenant_id="Demo Clinic", admin_email="a@b.example")
        assert services.log == []

    def test_an_existing_email_leaves_nothing_behind(self, monkeypatch, actor):
        """Cognito goes first precisely so this failure writes nothing. An address is the
        username for the whole pool and a user's tenant is immutable, so the account cannot be
        moved -- and a registry row for a tenant nobody can sign in to would be worse."""
        services = _services(monkeypatch, existing_emails={"taken@x.example"})
        with pytest.raises(TenantLifecycleError, match="already has an account"):
            create_tenant(services, actor, tenant_id=TARGET, admin_email="taken@x.example")

        assert services.tenant_registry.get(TARGET) is None
        assert services.tenant_directory.users_for_tenant(TARGET) == []
        # Empty, not a pack name: no pack was ever chosen for this tenant, which is the whole
        # claim. A name here would mean the settings write had happened after all.
        assert services.governance_store.get(TARGET).ontology_domain == ""
        assert TARGET not in [t.tenant_id for t in services.tenant_registry.list()]

    def test_a_taken_id_is_refused(self, monkeypatch, actor):
        services = _services(monkeypatch)
        with pytest.raises(TenantExists, match="already exists"):
            create_tenant(services, actor, tenant_id=BYSTANDER, admin_email="a@b.example")

    def test_a_deleted_id_is_not_silently_reusable(self, monkeypatch, actor):
        """The tombstone's whole purpose. Reusing an id would hand a new firm's people a graph
        somebody else built."""
        services = _services(monkeypatch)
        services.tenant_registry.tombstone(BYSTANDER, actor="someone")
        with pytest.raises(TenantExists, match="was deleted"):
            create_tenant(services, actor, tenant_id=BYSTANDER, admin_email="a@b.example")

    def test_users_bound_without_a_record_block_creation(self, monkeypatch, actor):
        """A tenant that exists only as bindings is the pre-registry state. Overwriting it would
        put a new admin into somebody else's data."""
        services = _services(monkeypatch)
        services.tenant_directory.seed("sub-ghost", TARGET, "ghost@x.example")
        with pytest.raises(TenantExists, match="without a tenant record"):
            create_tenant(services, actor, tenant_id=TARGET, admin_email="a@b.example")


class TestDeletingATenant:
    def _create(self, services, actor, tenant=TARGET):
        return create_tenant(
            services, actor, tenant_id=tenant, admin_email=f"admin@{tenant}.example"
        )

    def test_everything_belonging_to_the_tenant_goes(self, monkeypatch, actor):
        services = _services(monkeypatch)
        self._create(services, actor)
        report = delete_tenant(services, actor, TARGET)

        assert report.complete, report.errors
        assert report.users_deleted == 1
        assert report.documents_erased == 5
        assert report.grants_dropped > 0
        assert report.graph_audit_dropped == 3
        assert report.query_audit_dropped == 3
        assert report.settings_deleted
        assert report.tombstoned
        assert services.tenant_directory.users_for_tenant(TARGET) == []

    def test_another_tenant_is_untouched(self, monkeypatch, actor):
        """The assertion this file exists for."""
        services = _services(monkeypatch)
        self._create(services, actor)
        delete_tenant(services, actor, TARGET)

        assert [u.email for u in services.tenant_directory.users_for_tenant(BYSTANDER)] == [
            "bystander@x.example"
        ]
        assert services.access_store.assignments_for(BYSTANDER, f"sub-{BYSTANDER}@x.example")
        assert services.access_store.screens_for(BYSTANDER, f"sub-{BYSTANDER}@x.example")
        assert services.tenant_registry.get(BYSTANDER).is_live
        assert services.storage.swept == [TARGET]
        assert services.graph_audit.dropped == [TARGET]
        assert BYSTANDER not in services.user_admin.deleted

    def test_identity_goes_before_data_and_s3_goes_last(self, monkeypatch, actor):
        """Identity first, so nobody can sign in and upload into a tenant being swept. S3 last,
        because it is the only step nothing reconstructs: a failure above it leaves a tenant
        whose graph can still be rebuilt from those bytes."""
        services = _services(monkeypatch)
        self._create(services, actor)
        services.log.clear()
        delete_tenant(services, actor, TARGET)

        steps = [name for name, _ in services.log]
        assert steps.index("cognito.delete") < steps.index("s3.drop")
        assert steps[-1] == "s3.drop"

    def test_the_enumeration_source_is_the_directory_not_cognito(self, monkeypatch, actor):
        """`list_tenant_users` pages the pool once and filters in process, so it can return
        fewer of a tenant's users than exist -- and a missed user still signs in."""
        services = _services(monkeypatch)
        self._create(services, actor)
        services.user_admin.list_users_raises = True
        report = delete_tenant(services, actor, TARGET)
        assert report.users_deleted == 1
        assert report.complete, report.errors

    def test_a_failed_sweep_reports_incomplete_and_is_re_runnable(self, monkeypatch, actor):
        """The bytes are the one unrecoverable thing, so a failure there must not read as
        success -- and everything before it has already happened, so a retry must be safe."""
        services = _services(monkeypatch, storage_raises=True)
        self._create(services, actor)
        report = delete_tenant(services, actor, TARGET)

        assert report.complete is False
        assert any("documents" in e for e in report.errors)
        assert "not recoverable" not in report.to_dict()["note"]
        # The users are already gone, so a retry must not fail on their absence.
        assert services.tenant_directory.users_for_tenant(TARGET) == []
        again = delete_tenant(services, actor, TARGET)
        assert again.users_deleted == 0

    def test_settings_are_dropped_so_a_stale_kill_switch_cannot_answer(self, monkeypatch, actor):
        services = _services(monkeypatch)
        self._create(services, actor)
        services.governance[TARGET] = services.governance_store.get(TARGET)
        services.blocked_queries[TARGET] = [{"question": "x"}]
        delete_tenant(services, actor, TARGET)
        assert TARGET not in services.governance
        assert TARGET not in services.blocked_queries

    def test_deleting_without_a_directory_is_refused(self, monkeypatch, actor):
        """A tenant whose users cannot be enumerated must not be deleted: the data would go and
        the accounts would keep working."""
        services = _services(monkeypatch)
        services.tenant_directory = None
        with pytest.raises(TenantLifecycleError, match="cannot be enumerated"):
            delete_tenant(services, actor, TARGET)
        assert services.storage.swept == []
