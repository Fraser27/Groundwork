"""Shared FastAPI dependencies and the application container.

Everything mutable lives on one `Services` object rather than in module globals, so
tests construct a container and the app under test is the app that ships. The
alternative — module-level singletons initialised in a lifespan hook — makes
`TestClient` order-dependent in ways that are painful to debug.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path as FsPath
from typing import Annotated, Any

from fastapi import Depends, Header, HTTPException, Path, Query, status

from src.access import AccessManager, AccessStore, InMemoryAccessStore
from src.access_dynamo import DynamoAccessStore
from src.auth import Authenticator, AuthError, Grants, bearer_from_header
from src.config import GroundworkConfig, load_config
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

#: How many refused questions to keep per tenant. Enough to see a pattern worth writing a
#: metric for, bounded so a firm asking thousands cannot exhaust the task's memory.
MAX_BLOCKED_PER_TENANT = 200

#: Example metrics, one file per ontology pack, used only to seed a demo. The real store is the
#: graph, so this is not a configuration point and there is deliberately no env var for it.
#:
#: Per pack because a metric names real tables. Seeding `legal_ops.invoices` into a retail
#: deployment gives a reader six approved-looking definitions that compile to SQL against tables
#: their catalog does not have -- which is worse than an empty Metrics page, because it looks like
#: the semantic layer is already populated.
EXAMPLE_METRICS_DIR = FsPath(__file__).resolve().parents[2] / "sample" / "metrics"

#: How long to wait before trying the graph again after a failed connect. Long enough that a
#: genuinely down Neptune is not probed once per request, short enough that a task which lost the
#: race against a starting cluster heals on its own rather than waiting for an operator.
GRAPH_RECONNECT_COOLDOWN_SECONDS = 30.0


@dataclass
class Services:
    """Everything a route handler might need."""

    config: GroundworkConfig
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

    tenant_registry: Any | None = None
    """Which tenants exist. Separate from `tenant_directory`, which answers who belongs to one:
    a tenant with no users still exists, and is exactly the state a new one is in."""

    user_admin: UserAdmin | None = None
    """Creating and listing users. None without a user pool, which makes the admin routes
    answer 503 rather than pretending to work."""

    catalog: CatalogStore = field(default_factory=CatalogStore)
    """What the last Glue scan found. A cache over Glue, not a source of truth — losing it
    costs a re-scan, which is why it is in-memory.

    Read through `catalog_reader()` or `enriched_catalog()`, never directly: those reload it from
    the graph, and this store starts empty in every new process."""

    catalog_confirmed: dict[str, bool] = field(default_factory=dict)
    """Per tenant: did the durable copy answer when the catalog cache was last reloaded?

    Absent means it has not been tried. False is not the same as an empty catalog and must not be
    reported as one: "nobody has scanned" is a claim only something authoritative can support, and
    an empty cache on its own supports "this process has not been told"."""

    routing_index: Any | None = None
    """Where the tier router's semantic descriptions live. None without a vector endpoint, in
    which case tier selection falls back to keyword matching rather than erroring."""

    router_indexer: Any | None = None
    """Builds those descriptions. Per-process rather than per-request: it holds no request state
    and the graph and metric store it reads through are the shared ones."""

    enrichment_runs: dict[str, Any] = field(default_factory=dict)
    """The catalog enrichment run in flight per tenant, so a page can poll it.

    In process, like `blocked_queries`, and it dies with the container. That is the same limit
    `BackgroundTasks` already has, and it is better than a durable record stuck at RUNNING with no
    worker behind it. The work already staged survives regardless, because each table is staged as
    it completes."""

    def settings_for(self, tenant_id: str) -> GovernanceSettings:
        if tenant_id not in self.governance:
            store = self.governance_store
            settings = store.get(tenant_id) if store is not None else GovernanceSettings.from_env()
            # A tenant that has never chosen a pack inherits the one this process booted with,
            # rather than `GovernanceSettings`' own default. Two independent defaults meant
            # `GROUNDWORK_ONTOLOGY_PACK` could be honoured at boot and then silently overridden per
            # tenant, so the vocabulary a write was validated against was not the one the logs
            # and `/health` reported.
            if not settings.ontology_domain:
                settings = replace(settings, ontology_domain=self.ontology.domain)
            self.governance[tenant_id] = settings
        return self.governance[tenant_id]

    def ontology_for(self, tenant_id: str) -> Ontology:
        """The pack this tenant is governed by, which is not necessarily the one loaded at boot.

        `ontology_domain` was a per-tenant setting that nothing read: Admin persisted it, the
        vocabulary table re-rendered from `/ontology/{domain}`, and every write kept validating
        against the process-wide pack. A control that reports success and changes nothing is worse
        than no control, because the closed vocabulary is what the graph's defensibility rests on.

        Falls back to `self.ontology` when the tenant names no pack or names one that will not
        load. A missing pack must not take the tenant's writes down with it -- the boot pack is a
        working vocabulary, and the alternative is a 500 on every ingest.
        """
        domain = self.settings_for(tenant_id).ontology_domain
        if not domain or domain == self.ontology.domain:
            return self.ontology
        try:
            return load_ontology(domain)
        except FileNotFoundError:
            logger.warning(
                "tenant %s names ontology pack %r, which does not exist; using %r",
                tenant_id,
                domain,
                self.ontology.domain,
            )
            return self.ontology

    def catalog_graph_store(self) -> Any | None:
        """Catalog nodes and their descriptions in the graph, or None with no graph."""
        if self.graph is None:
            return None
        from src.discovery.graph_store import CatalogGraphStore

        return CatalogGraphStore(self.graph)

    def hydrate_catalog(self, tenant_id: str) -> bool:
        """Reload this tenant's catalog cache from the graph if it has not been. True if the
        durable copy answered.

        Called on read rather than at boot: the MCP sidecar has no lifespan, and a tenant that
        never asks costs nothing.

        False for both a missing graph and an unreachable one, and `catalog_confirmed` is what
        keeps those apart from a cache that is empty because nothing was ever scanned.

        Neither a failed read nor a missing graph is marked hydrated: it costs one refused read per
        page load while the graph is down, and the alternative is a task that reports "cannot say"
        forever after the graph comes back. That applies to no-graph too now that `connect_graph`
        retries -- marking it would pin an empty catalog in place across the reconnect, and
        `catalog_reader` is what the column allowlist is built from.

        A *successful* empty read is not permanent either. `CatalogStore._settle` expires it, which
        is what lets this process notice a scan run in the sibling one.
        """
        store = self.catalog_graph_store()
        if store is None:
            self.catalog_confirmed[tenant_id] = False
            return False
        if self.catalog.is_hydrated(tenant_id):
            return self.catalog_confirmed.get(tenant_id, False)
        from src.discovery.catalog_hydrate import hydrate_once

        ctx = AuthContext(user_id="catalog", tenant_id=tenant_id)
        try:
            hydrate_once(self.catalog, store, ctx)
        except Exception as exc:  # noqa: BLE001
            logger.warning("catalog not reloaded from the graph for %s: %s", tenant_id, exc)
            self.catalog_confirmed[tenant_id] = False
            return False
        self.catalog_confirmed[tenant_id] = True
        return True

    def catalog_reader(self) -> Any:
        """The raw catalog, reloaded from the graph on first read of each tenant.

        Two callers want the scanned shape without the description overlay: the firewall's
        allowlist and the metric compiler's schema. A description cannot change which tables exist
        or what type a column is, and a graph outage must not be able to shrink the allowlist.
        """
        return _HydratingCatalog(self)

    def catalog_synonyms(self) -> Any | None:
        """Approved synonyms per table `full_name` for a tenant, or None with no graph.

        Keyed by `full_name` and not by graph id, because `sql_generation` must not learn the id
        format: `catalog_overlay` establishes that ids are built, never parsed.

        Memoised per provider, and a provider lives as long as the resolver that holds it, so one
        request. `Planner` runs two lanes that each ask, which would otherwise be two graph reads
        for one answer, while a cache outliving the request would hide a synonym just approved.
        """
        store = self.catalog_graph_store()
        if store is None:
            return None
        from src.discovery.glue_scanner import table_node_id

        cache: dict[str, dict[str, list[str]]] = {}

        def synonyms_for(ctx: AuthContext) -> dict[str, list[str]]:
            hit = cache.get(ctx.tenant_id)
            if hit is not None:
                return hit
            by_subject = store.approved_synonyms(ctx)
            found = {}
            for table in self.catalog_reader().tables(ctx.tenant_id):
                names = by_subject.get(table_node_id(table.source_id, table.full_name))
                if names:
                    found[table.full_name] = list(names)
            cache[ctx.tenant_id] = found
            return found

        return synonyms_for

    def enriched_catalog(self) -> Any:
        """The catalog with approved descriptions layered on.

        Every read path uses this rather than `self.catalog`, so the schema the SQL generator is
        given and the schema the Tables page shows are the same text.
        """
        reader = self.catalog_reader()
        store = self.catalog_graph_store()
        if store is None:
            return reader
        from src.discovery.catalog_overlay import EnrichedCatalog

        # Tenant scope with no matter filter, which is correct here and worth stating because it is
        # the one place this module builds a context rather than receiving one. A description
        # describes a column, not a case: `enrich_tables` sets no `matter_id`, so these assertions
        # are tenant-level and a matter allowlist would filter on a property none of them carry.
        return EnrichedCatalog(
            reader,
            store,
            lambda tenant_id: AuthContext(user_id="catalog", tenant_id=tenant_id),
        )

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
            catalog=self.enriched_catalog(),
            sql_lane=self.build_sql_lane(tenant_id) if tenant_id else None,
            router=self.build_tier_router(),
            synonyms_for=self.catalog_synonyms(),
        )

    def build_planner(self, tenant_id: str = "", *, synthesise: bool = True) -> Any:
        """A planner over the same lanes `build_resolver` uses.

        Here rather than in the route because two callers now build one: `/query/compose` and the
        `compose` MCP tool. Two copies of this wiring would drift in the direction that matters
        least visibly -- one caller getting a `sql_lane` and the other not means the same question
        is governed differently depending on whether a person or an agent asked it.

        `synthesise` is a parameter rather than a fixed choice because the two callers want
        opposite defaults: a person reading the page wants prose over the parts, while an agent
        *is* the writer and a second model's paragraph would be an ungoverned layer it then writes
        over, with nobody able to say which of them added a claim.
        """
        from src.query.planner import Planner

        matcher = self.metric_matcher or (
            build_metric_matcher(self, tenant_id) if tenant_id else None
        )
        return Planner(
            metric_matcher=matcher,
            graph_reader=self.graph_reader,
            vector_search=VectorSearch(self.embedder) if self.embedder else None,
            catalog=self.enriched_catalog(),
            synthesiser=build_synthesiser(self) if synthesise else None,
            # Recorded, not obeyed: `ROUTER_NARROWS_LANES` is False, so every permitted lane still
            # runs. The router's decision is part of the trace rather than a filter on it.
            router=self.build_tier_router(),
            sql_lane=self.build_sql_lane(tenant_id) if tenant_id else None,
            question_splitter=self.build_question_splitter(tenant_id),
            synonyms_for=self.catalog_synonyms(),
        )

    def build_question_splitter(self, tenant_id: str = "") -> Any | None:
        """Splits a compound question, or None with no model configured.

        `query_model`, the same model that writes SQL: splitting is a structural read of the
        question rather than prose, so it belongs with the query-side setting a firm may pay more
        for, not with the summariser. Planner-only -- `/query` answers from one tier, and half a
        question answered by one tier is not an answer.
        """
        model_id = self.settings_for(tenant_id).query_model
        if not model_id:
            return None

        from src.query.decompose import QuestionSplitter

        return QuestionSplitter(model_id=model_id)

    def build_sql_lane(self, tenant_id: str) -> Any | None:
        """Model-written SQL over this tenant's catalogued schema, or None with no model.

        Per tenant, because the firewall's allowlist is built per request from the tables the prompt
        was offered -- see `sql_generation`. The same instance goes to `Resolver` and `Planner`, so
        `/query` and `/query/compose` cannot disagree about whether a question got generated SQL.

        `query_model` rather than `synthesis_model`: the two are separately settable precisely so a
        firm can pay for a stronger model writing a query than writing a paragraph over it, and this
        is the setting whose help text says so.
        """
        model_id = self.settings_for(tenant_id).query_model
        if not model_id:
            return None

        from src.query.sql_generation import SqlGenerator, SqlLane

        return SqlLane(
            generator=SqlGenerator(model_id=model_id),
            executor_factory=lambda offered: build_athena_executor(
                self, tenant_id, allowed_tables=offered, generated=True
            ),
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


class _HydratingCatalog:
    """`CatalogStore` that reloads a tenant from the graph before the first read.

    A wrapper at the seam rather than a `hydrate_once` call in each route, because a route added
    later cannot forget one it never had to write. Both processes serving this container reach the
    catalog through `Services`, so both get it.

    Everything else falls through to the real store, including the writes: `record_scan` marks the
    tenant hydrated itself, so a scan is not undone by a later reload.
    """

    def __init__(self, services: Services) -> None:
        self._services = services

    def tables(self, tenant_id: str) -> Any:
        self._services.hydrate_catalog(tenant_id)
        return self._services.catalog.tables(tenant_id)

    def table(self, tenant_id: str, full_name: str) -> Any:
        self._services.hydrate_catalog(tenant_id)
        return self._services.catalog.table(tenant_id, full_name)

    def sources(self, tenant_id: str) -> Any:
        self._services.hydrate_catalog(tenant_id)
        return self._services.catalog.sources(tenant_id)

    def with_sources(self, tenant_id: str, full_name: str) -> Any:
        self._services.hydrate_catalog(tenant_id)
        return self._services.catalog.with_sources(tenant_id, full_name)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._services.catalog, name)


def load_example_pack(domain: str = "legal") -> list[Any]:
    """The shipped example metrics for one ontology pack, for seeding a demo.

    Not the metric store. Metrics live in the graph with version history, authored through
    the app, because that is what makes "what did this definition mean when it answered"
    answerable. These files are examples for a fictional company, and `POST /metrics/seed`
    loads them as drafts for someone to look at.

    Empty for a pack with no file, which is the honest answer: a metric names a table, so
    there is nothing domain-neutral to offer instead.
    """
    path = EXAMPLE_METRICS_DIR / f"{domain}.yaml"
    if not path.exists():
        return []
    try:
        return load_metrics(path).metrics
    except Exception as e:
        logger.warning("example metric pack %s failed to load: %s", domain, e)
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

        for table in services.catalog_reader().tables(tenant_id):
            tables[table.full_name] = TableSchema(
                full_name=table.full_name,
                columns={c.name: c.data_type for c in table.columns},
                primary_keys=frozenset(c.name for c in table.columns if c.is_primary_key),
            )
    except Exception as e:
        logger.debug("no schema catalog available: %s", e)

    return MetricMatcher(
        metrics,
        StaticCatalog(tables=tables),
        executor=build_athena_executor(services, tenant_id),
        # The router again, as a *candidate* source for when no metric word matches the question.
        # None where there is no vector store, which is keyword-only matching -- the behaviour
        # before this existed.
        candidate_source=services.build_tier_router(),
    )


def build_athena_executor(
    services: Services,
    tenant_id: str,
    *,
    allowed_tables: set[str] | None = None,
    generated: bool = False,
) -> Any | None:
    """What actually runs a compiled metric, or None when there is nowhere to run it.

    Nothing constructed one for the life of this project, so `MetricMatch.run` logged "no executor
    wired" and returned None on every call: a governed metric compiled correctly and never returned
    a figure. Tier 1 has been half a tier.

    The firewall's allowlist is the tenant's own catalogued tables, so a compiled metric naming a
    table this tenant has not scanned is refused before it reaches Athena. That is defence in
    depth rather than the primary control -- a metric's SQL is compiled from a definition a human
    approved, not written by a model.

    `generated=True` narrows it to the tables actually offered to the prompt and turns on the
    aggregate and limit rules. Those are the difference between the two paths: an approved metric
    was read by a human who could see what it exposed, and a generated query was not.
    """
    structured = getattr(services.config, "structured", None)
    bucket = getattr(structured, "athena_results_bucket", "")
    if not bucket:
        return None

    from src.executors.athena import AthenaConfig, AthenaExecutor
    from src.query.firewall import SQLFirewall

    if allowed_tables is not None:
        allowed = {t for t in allowed_tables if t}
    else:
        try:
            allowed = {t.full_name for t in services.catalog_reader().tables(tenant_id)}
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
        SQLFirewall(
            allowed_tables=allowed,
            require_aggregate=generated,
            require_limit=generated,
        ),
    )


def build_router_indexer(services: Services) -> Any | None:
    """The indexer, wired to whatever graph is reachable now.

    The graph connects in the lifespan hook, *after* `build_services`, so the stored indexer is
    built without one and its graph-backed layers are attached here — the same reason
    `build_metric_matcher` runs per request rather than at startup.
    """
    indexer = services.router_indexer
    if indexer is None:
        return None
    # The enriched catalog, not the raw store: an approved description is most of what makes a table
    # findable, and the raw store carries only whatever comment Glue happened to hold. Swapped only
    # when it is this container's own store, so an injected catalog stays the one that was injected.
    if indexer.catalog is services.catalog:
        indexer.catalog = services.enriched_catalog()
    if services.graph is None:
        return indexer
    indexer.graph = services.graph
    if indexer.metric_store is None:
        from src.metrics.graph_store import GraphMetricStore

        indexer.metric_store = GraphMetricStore(services.graph)
    return indexer


def _build_parser(cfg: GroundworkConfig) -> VisionParser | None:
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


def _build_access_store(cfg: GroundworkConfig) -> AccessStore:
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
    cfg: GroundworkConfig,
) -> TenantDirectory | StaticTenantDirectory | None:
    """Where a user's tenant comes from when the access token cannot carry it.

    None in local dev with the auth bypass, where the tenant is configured directly. In a
    deployed environment TENANT_TABLE is set and this is the only source of the binding.
    """
    if not cfg.tables.tenants:
        return None
    logger.info("tenant bindings read from DynamoDB table %s", cfg.tables.tenants)
    return TenantDirectory(cfg.tables.tenants)


def _build_tenant_registry(cfg: GroundworkConfig) -> Any:
    """Which tenants exist. In-memory without a table, so local development still lists them."""
    from src.tenant_registry import InMemoryTenantRegistry, TenantRegistry

    if not cfg.tables.tenants:
        return InMemoryTenantRegistry()
    return TenantRegistry(cfg.tables.tenants)


def _build_job_store(cfg: GroundworkConfig) -> InMemoryJobStore | DynamoJobStore:
    """DynamoDB when a table is configured, in-memory otherwise.

    The fallback is not equivalent: an in-memory job store dies with the process, so a
    background ingest interrupted by a deploy cannot be found again. That is fine for
    local dev and is why JOB_TABLE is set in every deployed environment.
    """
    if not cfg.tables.jobs:
        return InMemoryJobStore()
    logger.info("ingest jobs backed by DynamoDB table %s", cfg.tables.jobs)
    return DynamoJobStore(cfg.tables.jobs)


def _build_graph_audit(cfg: GroundworkConfig) -> object:
    """The grants table, which already holds the compliance artifact and is already RETAIN.

    Falls back to memory rather than to nothing: a wipe that cannot be recorded still reports the
    failure, and `wipe` says so in its report, so an unaudited deletion is never silent.
    """
    if not cfg.tables.grants:
        logger.info("graph audit held in process (no grants table configured)")
        return InMemoryGraphAudit()
    logger.info("graph audit backed by DynamoDB table %s", cfg.tables.grants)
    return GraphAudit(cfg.tables.grants)


def _build_query_audit(cfg: GroundworkConfig) -> object:
    """The same table as the graph log, in its own partition.

    Falls back to memory rather than to nothing, for the same reason: a question that cannot be
    recorded is still answered, and losing the record on a deploy beats refusing to answer.
    """
    if not cfg.tables.grants:
        logger.info("query audit held in process (no grants table configured)")
        return InMemoryQueryAudit()
    logger.info("query audit backed by DynamoDB table %s", cfg.tables.grants)
    return QueryAudit(cfg.tables.grants)


def _build_governance_store(cfg: GroundworkConfig) -> object:
    """DynamoDB when a tenant table is configured, memory otherwise.

    Shares the tenant table rather than adding one: settings are a single small item per tenant
    read by key, which is what that table already is.
    """
    if not cfg.tables.tenants:
        logger.info("governance settings held in process (no tenant table configured)")
        return InMemoryGovernanceStore()
    logger.info("governance settings backed by DynamoDB table %s", cfg.tables.tenants)
    return GovernanceStore(cfg.tables.tenants)


def _build_vector_store(cfg: GroundworkConfig) -> object:
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


def _build_routing_index(cfg: GroundworkConfig) -> Any | None:
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


def build_services(config: GroundworkConfig | None = None) -> Services:
    cfg = config or load_config()
    ontology = load_ontology(cfg.ontology_pack)
    store = InMemoryAssertionStore()
    # The pack decides which predicates approval lifts into the answerable band, so the queue
    # needs it. Without one every approval is treated as governing, and an approved MENTIONS
    # would outrank the ADVERSE_TO it exists to stop outranking.
    queue = ReviewQueue(
        store,
        governing_predicates=ontology.governing_predicates,
        canonical_entity_id=ontology.canonical_entity_id,
        entity_blocking_keys=ontology.entity_blocking_keys,
    )
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

    # Built here rather than by the field default, because the router indexer reads through the
    # same instance and a second copy would index tables nobody scanned.
    catalog = CatalogStore()

    routing_index = _build_routing_index(cfg)
    router_indexer = None
    if routing_index is not None and embedder is not None:
        from src.query.router_indexer import RouterIndexer

        # No graph and no metric store yet: both are attached by `build_router_indexer` once the
        # lifespan hook has connected, and it swaps this raw store for the enriched catalog at the
        # same time. The raw store is what an indexer built without that call still reads.
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
        graph_reader=GraphReader(queue, ontology=ontology),
        embedder=embedder,
        catalog=catalog,
        routing_index=routing_index,
        router_indexer=router_indexer,
        parser=_build_parser(cfg),
        job_store=_build_job_store(cfg),
        ingest_limiter=IngestLimiter(cfg.documents.max_concurrent_ingests),
        tenant_directory=tenants,
        tenant_registry=_build_tenant_registry(cfg),
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
    # The one place both processes pass through per request, so it is where a graph that came up
    # after startup gets noticed. Costs an attribute check once connected, and is cooldown-gated
    # while it is not.
    if _services.graph is None:
        reconnect_if_due(_services)
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


def build_synthesiser(services: Services) -> Any | None:
    """The synthesis model, or None when the deployment has no Bedrock access.

    None rather than raising: without a model the parts and their citations are still the answer,
    and refusing the question outright would trade a complete result for no result.
    """
    model_id = getattr(getattr(services, "config", None), "models", None)
    model_id = getattr(model_id, "synthesis_model", "")
    if not model_id:
        return None

    from src.query.synthesis import Synthesiser

    return Synthesiser(model_id=model_id)


def drain_blocked(services: Any, tenant_id: str, blocked: list[Any]) -> None:
    """Move a request's refusals onto `Services`, which outlives it.

    A resolver and a planner are both built per request and discarded, so their lists died with
    them and the Governance screen could only ever show an empty backlog. A refusal is the signal
    the kill switch exists to produce: a question people keep asking is a metric waiting to be
    written.
    """
    for entry in blocked:
        services.record_blocked(
            tenant_id,
            {
                "question": entry.question,
                "user_id": entry.user_id,
                "reason": entry.reason,
                "at": entry.at,
            },
        )


#: Reentrant so `reconnect_if_due` can hold it across its own call to `connect_graph`.
_graph_connect_lock = threading.RLock()
_graph_retry_after = 0.0


def reconnect_if_due(services: Services) -> bool:
    """Retry a graph that failed to connect, at most once per cooldown. True if connected.

    Separate from `connect_graph` because the cooldown must not gate an explicit call. A lifespan
    hook or an operator asking to connect means *now*; only the per-request path is opportunistic,
    and that is the one that would otherwise probe a down Neptune once per request.
    """
    global _graph_retry_after

    if services.graph is not None:
        return True
    if time.monotonic() < _graph_retry_after:
        return False
    # Whoever holds it is already retrying on everyone's behalf, so no request waits on another's.
    if not _graph_connect_lock.acquire(blocking=False):
        return False
    try:
        return connect_graph(services)
    finally:
        if services.graph is None:
            _graph_retry_after = time.monotonic() + GRAPH_RECONNECT_COOLDOWN_SECONDS
        _graph_connect_lock.release()


def connect_graph(services: Services) -> bool:
    """Connect the graph and put the assertion store on it. True when the graph is reachable.

    Here rather than in `build_services` because the client does not exist until something
    decides to connect, and connecting lazily is what lets `/health` report a graph that is
    down instead of the process failing to start.

    Here rather than in the REST app's lifespan because **two** processes serve this container:
    the API and the MCP sidecar. While this lived only in the lifespan, the MCP server built a
    container, never connected, and served every tool from an empty `InMemoryAssertionStore` --
    so `search_assertions` returned zero rows for a tenant holding 56 facts, and reported it as
    an empty result rather than an error. A read path silently answering "nothing" is the exact
    failure this codebase treats as worse than a crash.

    Idempotent: an already-connected container is left alone.

    Failing is not terminal, see `reconnect_if_due`. This used to run once per process and give
    up, so a task that started a few minutes before Neptune finished provisioning stayed degraded
    for its whole life: metric authoring returned 503 long after the cluster was healthy, and only
    a redeploy cleared it.
    """
    if services.graph is not None:
        return True

    with _graph_connect_lock:
        # Another thread may have connected while this one waited.
        if services.graph is not None:
            return True
        return _connect_graph_once(services)


def _connect_graph_once(services: Services) -> bool:
    cfg = services.config
    try:
        from src.graph.client import GraphClient
        from src.graph.schema import init_schema

        client = GraphClient(
            cfg.graph.uri,
            cfg.graph.user,
            cfg.graph.password,
            iam_auth=cfg.graph.iam_auth,
            region=cfg.graph.region,
        )
        if not client.verify_connectivity():
            logger.warning("graph unreachable at %s, degraded mode", cfg.graph.uri)
            return False

        init_schema(client, is_neptune=cfg.graph.iam_auth)
        services.graph = client

        # Until this swap, an approval wrote to a dict and was lost on the next deploy while
        # the UI reported success.
        from src.graph.assertion_store import GraphAssertionStore

        services.review_queue.store = GraphAssertionStore(graph=client)
        logger.info("graph connected, assertions persisted to the graph")
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("graph init failed (%s), degraded mode", e)
        return False


def require_home_admin(services: Services, principal: tuple[AuthContext, Grants]) -> AuthContext:
    """Gate the routes where the tenant is an argument rather than the caller's own.

    Every other admin route reads the tenant from the token, so there is nothing to tamper
    with. Creating and deleting tenants cannot work that way -- the tenant being created does
    not exist yet, and the one being deleted is not the caller's -- so authority has to come
    from somewhere else. Being a platform-admin is not enough: that is a role within a firm,
    and one firm's admin must not reach another's data.

    The message names neither the caller's tenant nor the configured one. A refused caller
    learning which tenant *would* qualify is a probe answered.
    """
    ctx, grants = principal
    home = services.config.auth.home_tenant
    if not home:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "no operator tenant is configured (AUTH_HOME_TENANT unset), so creating and "
            "deleting tenants is closed",
        )
    if not grants.is_platform_admin or ctx.tenant_id != home:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, "requires platform-admin of the operator tenant"
        )
    return ctx


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
