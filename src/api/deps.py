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
from typing import Annotated, Any

from fastapi import Depends, Header, HTTPException, Path, Query, status

from src.access import AccessManager, AccessStore, InMemoryAccessStore
from src.access_dynamo import DynamoAccessStore
from src.auth import Authenticator, AuthError, Grants, bearer_from_header
from src.config import LexGraphConfig, load_config
from src.discovery.catalog_store import CatalogStore
from src.documents.embed import Embedder, InMemoryVectorStore
from src.documents.job_store import DynamoJobStore, InMemoryJobStore
from src.documents.parse import VisionParser
from src.documents.review import InMemoryAssertionStore, ReviewQueue
from src.documents.runner import IngestLimiter
from src.governance import GovernanceSettings
from src.governance_store import GovernanceStore, InMemoryGovernanceStore
from src.graph.scope import AuthContext, ScopeViolation
from src.graph_audit import GraphAudit, InMemoryGraphAudit
from src.metrics.loader import load_metrics
from src.metrics.models import StaticCatalog
from src.ontology.loader import Ontology, load_ontology
from src.query.graph_reader import GraphReader
from src.query.metric_matcher import MetricMatcher
from src.query.resolver import Resolver
from src.query.vector_search import VectorSearch
from src.query_audit import InMemoryQueryAudit, QueryAudit
from src.tenant_directory import StaticTenantDirectory, TenantDirectory
from src.user_admin import UserAdmin

logger = logging.getLogger(__name__)

#: Example metrics for a fictional firm, used only to seed a demo. The real store is the
#: How many refused questions to keep per tenant. Enough to see a pattern worth writing a
#: metric for, bounded so a firm asking thousands cannot exhaust the task's memory.
MAX_BLOCKED_PER_TENANT = 200

#: graph, so this is not a configuration point and there is deliberately no env var for it.
EXAMPLE_METRICS_PATH = FsPath(__file__).resolve().parents[2] / "sample" / "metrics.yaml"


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
    """Per-request cache in front of `governance_store`. Absent means read it."""

    governance_store: Any | None = None
    """Where settings live. DynamoDB when a tenant table is configured, memory otherwise.
    Without this an Admin change was lost on the next deploy, which for the ungoverned-query
    kill switch means a firm believing questions are refused while they are being answered."""

    graph_audit: Any | None = None
    """Append-only record of who changed what the system believes: a reviewer overriding a model,
    an administrator wiping a document. Distinct from the access audit, which answers who could
    *read* what."""

    query_audit: Any | None = None
    """Append-only record of what was asked and on what basis. The read side of the same claim:
    without it a partner cannot say which advice rested on a fact that later turned out wrong."""

    blocked_queries: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    """Refused questions per tenant, newest last.

    Here rather than on the `Resolver` because a resolver is built per request and thrown away,
    so its record of refusals died with it and the admin surface could only ever show an empty
    list. A refusal is the signal the kill switch exists to produce: a question people keep
    asking is a governed metric waiting to be written.

    In process and capped, deliberately. This is a backlog to read, not an audit record -- the
    audit trail is the log line -- so losing it on a deploy costs a hint rather than evidence,
    and a firm that asks ten thousand refused questions must not exhaust the task's memory."""

    graph: object | None = None
    """None when no graph is reachable. Routes that need it return 503 rather than
    failing at import time, so the API still starts and /health can say why."""

    graph_reader: GraphReader | None = None
    metric_matcher: MetricMatcher | None = None
    """Injected by tests. In a running system the matcher is built per request from the
    tenant's approved metrics in the graph, so this stays None."""

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

    tenant_directory: TenantDirectory | StaticTenantDirectory | None = None
    """Which tenant a subject belongs to. The same instance `Authenticator` reads through,
    so a binding written by the admin API is visible to the next request."""

    user_admin: UserAdmin | None = None
    """Creating and listing users. None without a user pool, which makes the admin routes
    answer 503 rather than pretending to work."""

    catalog: CatalogStore = field(default_factory=CatalogStore)
    """What the last Glue scan found. A cache over Glue, not a source of truth — losing it
    costs a re-scan, which is why it is in-memory."""

    routing_index: Any | None = None
    """Where the tier router's semantic descriptions live. None without a vector endpoint, in
    which case tier selection falls back to keyword matching rather than erroring."""

    router_indexer: Any | None = None
    """Builds those descriptions. Per-process rather than per-request: it holds no request state
    and the graph and metric store it reads through are the shared ones."""

    def settings_for(self, tenant_id: str) -> GovernanceSettings:
        if tenant_id not in self.governance:
            store = self.governance_store
            self.governance[tenant_id] = (
                store.get(tenant_id) if store is not None else GovernanceSettings.from_env()
            )
        return self.governance[tenant_id]

    def record_blocked(self, tenant_id: str, entry: dict[str, Any]) -> None:
        """Remember a refusal for the Governance screen."""
        log = self.blocked_queries.setdefault(tenant_id, [])
        log.append(entry)
        if len(log) > MAX_BLOCKED_PER_TENANT:
            # Oldest first: a recent refusal is the one an administrator can still act on.
            del log[: len(log) - MAX_BLOCKED_PER_TENANT]

    def record_question(self, event: Any) -> bool:
        """Record an answered question. False means it was not recorded.

        Never raises. An audit write that fails must not turn a good answer into a 500 — the
        caller already has the answer, and the log line below is the fallback record.
        """
        audit = self.query_audit
        if audit is None:
            return False
        try:
            audit.append(event)
            return True
        except Exception:
            logger.exception("question not recorded in the query audit: %r", event.question)
            return False

    def save_settings(self, tenant_id: str, settings: GovernanceSettings) -> GovernanceSettings:
        """Persist a governance change and refresh the in-process copy.

        Both, in that order: writing without updating the cache would leave this task serving
        the old settings for the cache's lifetime, so an administrator would toggle something
        and watch it appear not to apply.
        """
        if self.governance_store is not None:
            settings = self.governance_store.put(tenant_id, settings)
        self.governance[tenant_id] = settings
        return settings

    def build_resolver(self, tenant_id: str = "") -> Resolver:
        """A resolver wired to whatever is actually available.

        Constructed per request because a `Resolver` records blocked queries and that state
        should not be shared. The metric matcher is also per request, and per *tenant*: it
        reads this firm's approved metrics from the graph, so a metric approved a minute ago
        answers the next question without a restart.
        """
        # An injected matcher wins. Tests set it directly, and honouring it keeps them
        # testing tier-1 behaviour rather than graph plumbing. In a running system it is
        # None, so the tenant's approved metrics are read from the graph.
        matcher = self.metric_matcher or (
            build_metric_matcher(self, tenant_id) if tenant_id else None
        )
        return Resolver(
            metric_matcher=matcher,
            graph_reader=self.graph_reader,
            vector_search=VectorSearch(self.embedder) if self.embedder else None,
            sql_generator=None,
            firewall=None,
            router=self.build_tier_router(),
        )

    def build_tier_router(self) -> Any | None:
        """The router, or None when there is nothing for it to search.

        None rather than a disabled router: the resolver treats absence as "try every permitted
        tier in order", which is precisely the behaviour a deployment without a vector store
        should have, so there is nothing to special-case downstream.
        """
        if self.routing_index is None or self.embedder is None:
            return None

        from src.query.router import TierRouter

        return TierRouter(
            routing_index=self.routing_index,
            # The same store the passage lane reads, so a routing score for tier 3 and the
            # retrieval that follows it cannot disagree about what is in the index.
            chunk_search=getattr(self.embedder, "store", None),
            embedder=self.embedder,
        )


def load_example_pack() -> list[Any]:
    """The shipped example metrics, for seeding a demo.

    Not the metric store. Metrics live in the graph with version history, authored through
    the app, because that is what makes "what did this definition mean when it answered"
    answerable. This file is examples for a fictional firm, and `POST /metrics/seed` loads
    them as drafts for someone to look at.
    """
    if not EXAMPLE_METRICS_PATH.exists():
        return []
    try:
        return load_metrics(EXAMPLE_METRICS_PATH).metrics
    except Exception as e:
        logger.warning("example metric pack failed to load: %s", e)
        return []


def build_metric_matcher(services: Services, tenant_id: str) -> MetricMatcher | None:
    """What tier 1 matches a question against: this tenant's **approved** metrics.

    Built per request rather than at startup, because a metric approved a minute ago has to
    be able to answer the next question. A missing graph disables tier 1 rather than
    falling back to the example pack: answering a firm's question with a demo metric about
    a fictional firm's invoices would be worse than declining to answer.
    """
    if services.graph is None:
        return None
    try:
        from src.metrics.graph_store import GraphMetricStore

        metrics = GraphMetricStore(services.graph).list_metrics(tenant_id, approved_only=True)
    except Exception as e:
        logger.warning("could not load metrics for %s: %s", tenant_id, e)
        return None
    if not metrics:
        return None

    tables = {}
    try:
        from src.metrics.models import TableSchema

        for table in services.catalog.tables(tenant_id):
            tables[table.full_name] = TableSchema(
                full_name=table.full_name,
                columns={c.name: c.data_type for c in table.columns},
                primary_keys=frozenset(c.name for c in table.columns if c.is_primary_key),
            )
    except Exception as e:
        logger.debug("no schema catalog available: %s", e)

    return MetricMatcher(
        metrics, StaticCatalog(tables=tables), executor=build_athena_executor(services, tenant_id)
    )


def build_athena_executor(services: Services, tenant_id: str) -> Any | None:
    """What actually runs a compiled metric, or None when there is nowhere to run it.

    Nothing constructed one for the life of this project, so `MetricMatch.run` logged "no executor
    wired" and returned None on every call: a governed metric compiled correctly and never returned
    a figure. Tier 1 has been half a tier.

    The firewall's allowlist is the tenant's own catalogued tables, so a compiled metric naming a
    table this tenant has not scanned is refused before it reaches Athena. That is defence in
    depth rather than the primary control -- a metric's SQL is compiled from a definition a human
    approved, not written by a model -- but it is the same firewall that will check generated SQL,
    and having it live on the deterministic path first is deliberate.
    """
    structured = getattr(services.config, "structured", None)
    bucket = getattr(structured, "athena_results_bucket", "")
    if not bucket:
        return None

    from src.executors.athena import AthenaConfig, AthenaExecutor
    from src.query.firewall import SQLFirewall

    try:
        allowed = {t.full_name for t in services.catalog.tables(tenant_id)}
    except Exception as e:
        logger.debug("no catalog for the firewall allowlist: %s", e)
        allowed = set()

    return AthenaExecutor(
        AthenaConfig(
            workgroup=getattr(structured, "athena_workgroup", "") or "primary",
            # Under the prefix the 14-day lifecycle rule covers, so results expire and the Iceberg
            # warehouse sharing the same bucket does not.
            output_location=f"s3://{bucket}/athena-results/",
            # The graph's region rather than a separate setting: one deployment, one region, and a
            # second knob to keep in step would only ever be wrong.
            region=getattr(services.config.graph, "region", "") or "",
        ),
        SQLFirewall(allowed_tables=allowed),
    )


def build_router_indexer(services: Services) -> Any | None:
    """The indexer, wired to whatever graph is reachable now.

    The graph connects in the lifespan hook, *after* `build_services`, so the stored indexer is
    built without one and its graph-backed layers are attached here — the same reason
    `build_metric_matcher` runs per request rather than at startup.
    """
    indexer = services.router_indexer
    if indexer is None or services.graph is None:
        return indexer
    indexer.graph = services.graph
    if indexer.metric_store is None:
        from src.metrics.graph_store import GraphMetricStore

        indexer.metric_store = GraphMetricStore(services.graph)
    return indexer


def _build_parser(cfg: LexGraphConfig) -> VisionParser | None:
    """The page reader, or None when there is no way to reach a vision model.

    boto3's absence is checked here because it is knowable at boot; a missing credential
    is not, so the client itself is built on first use. Either way the ingest route
    degrades to store-without-transcribing rather than failing the upload.
    """
    try:
        import boto3
    except ImportError:
        logger.warning("boto3 unavailable, uploads will be stored but not transcribed")
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


def _build_graph_audit(cfg: LexGraphConfig) -> object:
    """The grants table, which already holds the compliance artifact and is already RETAIN.

    Falls back to memory rather than to nothing: a wipe that cannot be recorded still reports the
    failure, and `wipe` says so in its report, so an unaudited deletion is never silent.
    """
    if not cfg.tables.grants:
        logger.info("graph audit held in process (no grants table configured)")
        return InMemoryGraphAudit()
    logger.info("graph audit backed by DynamoDB table %s", cfg.tables.grants)
    return GraphAudit(cfg.tables.grants)


def _build_query_audit(cfg: LexGraphConfig) -> object:
    """The same table as the graph log, in its own partition.

    Falls back to memory rather than to nothing, for the same reason: a question that cannot be
    recorded is still answered, and losing the record on a deploy beats refusing to answer.
    """
    if not cfg.tables.grants:
        logger.info("query audit held in process (no grants table configured)")
        return InMemoryQueryAudit()
    logger.info("query audit backed by DynamoDB table %s", cfg.tables.grants)
    return QueryAudit(cfg.tables.grants)


def _build_governance_store(cfg: LexGraphConfig) -> object:
    """DynamoDB when a tenant table is configured, memory otherwise.

    Shares the tenant table rather than adding one: settings are a single small item per tenant
    read by key, which is what that table already is.
    """
    if not cfg.tables.tenants:
        logger.info("governance settings held in process (no tenant table configured)")
        return InMemoryGovernanceStore()
    logger.info("governance settings backed by DynamoDB table %s", cfg.tables.tenants)
    return GovernanceStore(cfg.tables.tenants)


def _build_vector_store(cfg: LexGraphConfig) -> object:
    """OpenSearch when an endpoint is configured, memory otherwise.

    This used to build the in-memory store unconditionally, so a deployed system with a
    provisioned collection still held its embeddings in one task's memory and lost them on every
    deploy. Falls back rather than raising if the client cannot be built: search degrading to
    keyword-only is worse than vector search but far better than an API that will not start.
    """
    if not cfg.vector.endpoint:
        # Reachable: `vector.enabled` is the caller's gate, but a helper that silently returns
        # an OpenSearch client pointed at "" would fail on first search rather than here.
        return InMemoryVectorStore()
    try:
        from src.documents.opensearch_store import OpenSearchVectorStore

        store = OpenSearchVectorStore(
            endpoint=cfg.vector.endpoint,
            region=cfg.vector.region,
            dimensions=cfg.vector.embedding_dimensions,
        )
        logger.info("vectors backed by OpenSearch at %s", cfg.vector.endpoint)
        return store
    except Exception as e:
        logger.warning("OpenSearch unavailable (%s), vectors held in process", e)
        return InMemoryVectorStore()


def _build_routing_index(cfg: LexGraphConfig) -> Any | None:
    """The tier router's index, or None with no endpoint.

    None rather than an in-memory stand-in, unlike `_build_vector_store`. A routing index that
    lived in one task's memory would be empty for whichever task served the next question, so the
    router would see no hits and quietly decide nothing looked relevant -- worse than having no
    router, because the fallback is at least honest about it.
    """
    if not cfg.vector.enabled:
        return None
    try:
        from src.query.router_index import RoutingIndex

        index = RoutingIndex(
            endpoint=cfg.vector.endpoint,
            region=cfg.vector.region,
            dimensions=cfg.vector.embedding_dimensions,
        )
        logger.info("tier routing index backed by OpenSearch at %s", cfg.vector.endpoint)
        return index
    except Exception as e:
        logger.warning("routing index unavailable (%s), tier selection stays keyword-based", e)
        return None


def build_services(config: LexGraphConfig | None = None) -> Services:
    cfg = config or load_config()
    store = InMemoryAssertionStore()
    queue = ReviewQueue(store)
    access = AccessManager(_build_access_store(cfg))
    # One instance, shared: the authenticator reads bindings the admin API writes, so a
    # newly invited user can sign in without waiting for a cache in a second copy.
    tenants = _build_tenant_directory(cfg)

    governance_store = _build_governance_store(cfg)
    graph_audit = _build_graph_audit(cfg)
    query_audit = _build_query_audit(cfg)

    embedder: Embedder | None = None
    if cfg.vector.enabled:
        # Only constructed when an endpoint is configured; Bedrock is reached lazily
        # so a missing credential surfaces on first use rather than at boot.
        embedder = Embedder(
            _build_vector_store(cfg),
            model_id=cfg.vector.embedding_model,
            dimensions=cfg.vector.embedding_dimensions,
        )

    ontology = load_ontology(cfg.ontology_pack)
    # Built here rather than by the field default, because the router indexer reads through the
    # same instance and a second copy would index tables nobody scanned.
    catalog = CatalogStore()

    routing_index = _build_routing_index(cfg)
    router_indexer = None
    if routing_index is not None and embedder is not None:
        from src.query.router_indexer import RouterIndexer

        # No graph and no metric store yet: both are attached by `build_router_indexer` once the
        # lifespan hook has connected. The catalog store exists from the start.
        router_indexer = RouterIndexer(
            routing_index,
            embedder=embedder,
            ontology=ontology,
            catalog=catalog,
        )

    return Services(
        config=cfg,
        authenticator=Authenticator(cfg, access, tenants),
        ontology=ontology,
        review_queue=queue,
        access=access,
        graph_reader=GraphReader(queue),
        embedder=embedder,
        catalog=catalog,
        routing_index=routing_index,
        router_indexer=router_indexer,
        parser=_build_parser(cfg),
        job_store=_build_job_store(cfg),
        ingest_limiter=IngestLimiter(cfg.documents.max_concurrent_ingests),
        tenant_directory=tenants,
        governance_store=governance_store,
        graph_audit=graph_audit,
        query_audit=query_audit,
        user_admin=(
            UserAdmin(cfg.auth.user_pool_id, region=cfg.auth.region)
            if cfg.auth.user_pool_id
            else None
        ),
    )


_services: Services | None = None


def set_services(services: Services) -> None:
    global _services
    _services = services


def get_services() -> Services:
    if _services is None:
        raise RuntimeError("services not initialised, call set_services() first")
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
