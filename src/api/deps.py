"""Shared FastAPI dependencies and the application container.

Everything mutable lives on one `Services` object rather than in module globals, so
tests construct a container and the app under test is the app that ships. The
alternative — module-level singletons initialised in a lifespan hook — makes
`TestClient` order-dependent in ways that are painful to debug.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path as FsPath
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Path, Query, status

from src.access import AccessManager, AccessStore, InMemoryAccessStore
from src.access_dynamo import DynamoAccessStore
from src.auth import Authenticator, AuthError, Grants, bearer_from_header
from src.config import LexGraphConfig, load_config
from src.documents.embed import Embedder, InMemoryVectorStore
from src.documents.job_store import DynamoJobStore, InMemoryJobStore
from src.documents.parse import VisionParser
from src.documents.review import InMemoryAssertionStore, ReviewQueue
from src.documents.runner import IngestLimiter
from src.governance import GovernanceSettings
from src.graph.scope import AuthContext, ScopeViolation
from src.metrics.loader import load_metrics
from src.metrics.models import StaticCatalog
from src.ontology.loader import Ontology, load_ontology
from src.query.graph_reader import GraphReader
from src.query.metric_matcher import MetricMatcher
from src.query.resolver import Resolver
from src.query.vector_search import VectorSearch
from src.tenant_directory import StaticTenantDirectory, TenantDirectory

logger = logging.getLogger(__name__)

#: Default metric pack. YAML rather than the database at this stage.
METRICS_PATH = FsPath(__file__).resolve().parents[2] / "sample" / "metrics.yaml"


@dataclass
class Services:
    """Everything a route handler might need."""

    config: LexGraphConfig
    authenticator: Authenticator
    ontology: Ontology
    review_queue: ReviewQueue
    access: AccessManager
    """Assignments, screens and the audit trail. The same instance `Authenticator`
    resolves through, so a screen raised over the API bites the next request."""

    governance: dict[str, GovernanceSettings] = field(default_factory=dict)
    """Per-tenant overrides. Absent means env defaults."""

    graph: object | None = None
    """None when no graph is reachable. Routes that need it return 503 rather than
    failing at import time, so the API still starts and /health can say why."""

    graph_reader: GraphReader | None = None
    metric_matcher: MetricMatcher | None = None

    embedder: Embedder | None = None
    """None disables the vector path. Retrieval then degrades to graph traversal
    rather than erroring — a partial answer beats an outage."""

    parser: VisionParser | None = None
    """None means an upload is stored but not transcribed. Degrading rather than
    refusing is deliberate: the bytes are the record, so a document uploaded without a
    vision model is re-parseable, whereas one refused at the door is gone."""

    job_store: InMemoryJobStore | DynamoJobStore = field(default_factory=InMemoryJobStore)
    """Where ingest job state lives. In-memory locally; DynamoDB when JOB_TABLE is set,
    which is what makes a job outlive the container that started it."""

    ingest_limiter: IngestLimiter | None = None
    """Bounds concurrent ingests. Shared across requests, so it is built once with the
    process rather than per call."""

    def settings_for(self, tenant_id: str) -> GovernanceSettings:
        if tenant_id not in self.governance:
            self.governance[tenant_id] = GovernanceSettings.from_env()
        return self.governance[tenant_id]

    def build_resolver(self) -> Resolver:
        """A resolver wired to whatever is actually available.

        Constructed per request because a `Resolver` records blocked queries and that
        state should not be shared, but the collaborators it wraps are long-lived.
        """
        return Resolver(
            metric_matcher=self.metric_matcher,
            graph_reader=self.graph_reader,
            vector_search=VectorSearch(self.embedder) if self.embedder else None,
            sql_generator=None,
            firewall=None,
        )


def _load_metric_matcher(cfg: LexGraphConfig) -> MetricMatcher | None:
    """Load the sample metric pack if it is present.

    Metrics live in YAML rather than the database at this stage, so a missing pack is
    normal and disables tier 1 rather than failing startup.
    """
    path = FsPath(cfg.metrics_file) if getattr(cfg, "metrics_file", "") else METRICS_PATH
    if not path.exists():
        return None
    try:
        result = load_metrics(path)
    except Exception as e:
        logger.warning("metric pack %s failed to load: %s", path, e)
        return None
    if not result.metrics:
        return None

    catalog = StaticCatalog(tables={})
    logger.info("loaded %d governed metrics from %s", len(result.metrics), path)
    return MetricMatcher(result.metrics, catalog)


def _build_parser(cfg: LexGraphConfig) -> VisionParser | None:
    """The page reader, or None when there is no way to reach a vision model.

    boto3's absence is checked here because it is knowable at boot; a missing credential
    is not, so the client itself is built on first use. Either way the ingest route
    degrades to store-without-transcribing rather than failing the upload.
    """
    try:
        import boto3
    except ImportError:
        logger.warning("boto3 unavailable — uploads will be stored but not transcribed")
        return None

    region = cfg.models.region
    return VisionParser(
        model_id=cfg.documents.ocr_model,
        bedrock_factory=lambda: boto3.client("bedrock-runtime", region_name=region),
        dpi=cfg.documents.ocr_dpi,
        max_pages=cfg.documents.max_pages,
        batch_size=cfg.documents.page_batch_size,
        max_concurrency=cfg.documents.page_concurrency,
    )


def _build_access_store(cfg: LexGraphConfig) -> AccessStore:
    """DynamoDB when a table is configured, in-memory otherwise.

    Degrading locally is the same pattern as the embedder, with one difference worth
    naming: an in-memory access store is still closed by default, so the fallback loses
    persistence rather than enforcement.
    """
    if not cfg.tables.grants:
        return InMemoryAccessStore()
    logger.info("matter access backed by DynamoDB table %s", cfg.tables.grants)
    return DynamoAccessStore(cfg.tables.grants)


def _build_tenant_directory(
    cfg: LexGraphConfig,
) -> TenantDirectory | StaticTenantDirectory | None:
    """Where a user's tenant comes from when the access token cannot carry it.

    None in local dev with the auth bypass, where the tenant is configured directly. In a
    deployed environment TENANT_TABLE is set and this is the only source of the binding.
    """
    if not cfg.tables.tenants:
        return None
    logger.info("tenant bindings read from DynamoDB table %s", cfg.tables.tenants)
    return TenantDirectory(cfg.tables.tenants)


def _build_job_store(cfg: LexGraphConfig) -> InMemoryJobStore | DynamoJobStore:
    """DynamoDB when a table is configured, in-memory otherwise.

    The fallback is not equivalent: an in-memory job store dies with the process, so a
    background ingest interrupted by a deploy cannot be found again. That is fine for
    local dev and is why JOB_TABLE is set in every deployed environment.
    """
    if not cfg.tables.jobs:
        return InMemoryJobStore()
    logger.info("ingest jobs backed by DynamoDB table %s", cfg.tables.jobs)
    return DynamoJobStore(cfg.tables.jobs)


def build_services(config: LexGraphConfig | None = None) -> Services:
    cfg = config or load_config()
    store = InMemoryAssertionStore()
    queue = ReviewQueue(store)
    access = AccessManager(_build_access_store(cfg))

    embedder: Embedder | None = None
    if cfg.vector.enabled:
        # Only constructed when an endpoint is configured; Bedrock is reached lazily
        # so a missing credential surfaces on first use rather than at boot.
        embedder = Embedder(
            InMemoryVectorStore(),
            model_id=cfg.vector.embedding_model,
            dimensions=cfg.vector.embedding_dimensions,
        )

    return Services(
        config=cfg,
        authenticator=Authenticator(cfg, access, _build_tenant_directory(cfg)),
        ontology=load_ontology(cfg.ontology_pack),
        review_queue=queue,
        access=access,
        graph_reader=GraphReader(queue),
        metric_matcher=_load_metric_matcher(cfg),
        embedder=embedder,
        parser=_build_parser(cfg),
        job_store=_build_job_store(cfg),
        ingest_limiter=IngestLimiter(cfg.documents.max_concurrent_ingests),
    )


_services: Services | None = None


def set_services(services: Services) -> None:
    global _services
    _services = services


def get_services() -> Services:
    if _services is None:
        raise RuntimeError("services not initialised — call set_services() first")
    return _services


ServicesDep = Annotated[Services, Depends(get_services)]


async def get_principal(
    services: ServicesDep,
    authorization: Annotated[str | None, Header()] = None,
    include_suggestions: Annotated[bool, Query()] = False,
) -> tuple[AuthContext, Grants]:
    """Authenticate, and build the scope every graph read is filtered by.

    `include_suggestions` is a query parameter because it is a *view* choice, not a
    permission — `scope.edge_scope` still refuses PREDICTED unless it is set, so the
    parameter can only narrow trust, never widen it.
    """
    try:
        return services.authenticator.authenticate(
            bearer_from_header(authorization), include_suggestions=include_suggestions
        )
    except AuthError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(e)) from e


PrincipalDep = Annotated[tuple[AuthContext, Grants], Depends(get_principal)]


async def get_tenant_scope(
    services: ServicesDep,
    principal: PrincipalDep,
    tenant: Annotated[str, Path()],
) -> tuple[AuthContext, Grants]:
    """Validate the path tenant against the token.

    404 rather than 403 on mismatch: confirming another tenant exists is itself a
    leak, and "no such tenant, as far as you are concerned" is the honest answer.
    """
    ctx, grants = principal
    try:
        services.authenticator.assert_tenant_matches(ctx, tenant)
    except AuthError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e)) from e
    return ctx, grants


TenantDep = Annotated[tuple[AuthContext, Grants], Depends(get_tenant_scope)]


def require_reviewer(principal: tuple[AuthContext, Grants]) -> None:
    _, grants = principal
    if not grants.can_review:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "approving or rejecting an extracted claim requires a reviewer role",
        )


def require_admin(principal: tuple[AuthContext, Grants]) -> None:
    _, grants = principal
    if not grants.is_platform_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "requires platform-admin")


def scope_violation_to_http(e: ScopeViolation) -> HTTPException | ScopeViolation:
    """What to raise so a scope violation reaches the client correctly — never a 500.

    A screen is handed back unchanged so `app.py`'s handler builds the 403 with the matter,
    reason and contact on it. Flattening it to an `HTTPException` here would need a second
    copy of that body, and the two would drift. Everything else becomes 404, keeping a
    cross-tenant probe unable to distinguish refusal from absence.
    """
    if e.is_screen:
        return e
    return HTTPException(status.HTTP_404_NOT_FOUND, str(e))
