"""Configuration, loaded from the environment.

Environment-only, with no config file. In development the graph is Neo4j in Docker
and in deployment it is Neptune behind IAM auth; the CDK app-stack sets the same
variable names as `.env.example`, so the difference between the two is entirely in
values rather than in code paths.

One thing is deliberately absent: there is no setting that widens tenant scope.
`tenant_id` comes from a verified JWT at request time and never from configuration,
because a config-supplied tenant would be a cross-firm privilege breach one typo
away.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from src.constants import (
    APP_PORT,
    DEFAULT_AWS_REGION,
    DEFAULT_CHUNK_CHARS,
    DEFAULT_CHUNK_OVERLAP_CHARS,
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EXTRACTION_MODEL,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_OCR_MODEL,
    DEFAULT_ONTOLOGY_PACK,
    DEFAULT_SYNTHESIS_MODEL,
    MAX_CONCURRENT_INGESTS,
    MAX_PAGE_CONCURRENCY,
    MAX_QUERY_ROWS,
    PAGE_BATCH_SIZE,
)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw else default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class GraphConfig:
    """Neptune in AWS, Neo4j locally. Both speak openCypher over Bolt."""

    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = ""
    """Empty in deployment: with `iam_auth` the handshake is SigV4-signed and there
    is no password to leak or rotate. Only local Neo4j needs one."""

    iam_auth: bool = False
    region: str = DEFAULT_AWS_REGION
    max_connection_pool_size: int = 50


@dataclass
class VectorConfig:
    """OpenSearch Serverless, or nothing.

    `endpoint` empty disables vector retrieval rather than failing: keyword search
    over verbatim text still works, and a degraded read path beats no read path.
    """

    endpoint: str = ""
    collection: str = ""
    region: str = DEFAULT_AWS_REGION
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS
    min_score: float = 0.6
    top_k: int = 20

    @property
    def enabled(self) -> bool:
        return bool(self.endpoint)


@dataclass
class DocumentConfig:
    bucket: str = ""
    kms_key_id: str = ""
    chunk_chars: int = DEFAULT_CHUNK_CHARS
    chunk_overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS
    ocr_model: str = DEFAULT_OCR_MODEL
    """Vision model that transcribes pages. Separate from the extraction model on
    purpose: transcription is mechanical and a cheap model does it well, whereas
    judging what a passage means is not. Editable in Admin."""

    ocr_dpi: int = 150
    max_pages: int = 400

    page_batch_size: int = PAGE_BATCH_SIZE
    """Pages per batch. The unit of progress reporting and of a confined failure."""

    page_concurrency: int = MAX_PAGE_CONCURRENCY
    """In-flight vision calls per document. The throughput knob: sequential, a
    400-page bundle takes tens of minutes; unbounded, it is throttled."""

    max_concurrent_ingests: int = MAX_CONCURRENT_INGESTS
    """Documents ingesting at once in one process. Without this a bulk upload
    multiplies by `page_concurrency` into Bedrock throttling and surprise cost."""


@dataclass
class StructuredConfig:
    """Glue and Athena. Rows are queried in place; only metadata enters the graph."""

    glue_databases: list[str] = field(default_factory=list)
    athena_workgroup: str = "primary"
    athena_results_bucket: str = ""
    max_query_rows: int = MAX_QUERY_ROWS


@dataclass
class AuthConfig:
    user_pool_id: str = ""
    client_id: str = ""
    issuer_url: str = ""
    policy_store_id: str = ""
    region: str = DEFAULT_AWS_REGION

    dev_bypass_tenant: str = ""
    """Local development only: a tenant_id to assume when no JWT is present.

    Refused outright unless `LexGraphConfig.environment == "local"`, because a
    production deployment that honoured this would accept unauthenticated requests
    as a real tenant. See `LexGraphConfig.validate`."""

    home_tenant: str = ""
    """The one tenant whose platform-admins may create and delete other tenants.

    Every other admin route takes its tenant from the caller's own token, so a firm's admin has
    no parameter to tamper with. The platform routes are the deliberate exception -- the tenant
    is an argument there -- so something has to say who may pass it, and "any platform-admin"
    would let one customer delete another.

    Empty closes those routes. Closed by default: an unset operator tenant is a
    misconfiguration, and the safe reading of it is that nobody qualifies."""


@dataclass
class ModelConfig:
    extraction_model: str = DEFAULT_EXTRACTION_MODEL
    synthesis_model: str = DEFAULT_SYNTHESIS_MODEL
    region: str = DEFAULT_AWS_REGION


@dataclass
class TableConfig:
    tenants: str = ""
    jobs: str = ""
    grants: str = ""


@dataclass
class LexGraphConfig:
    environment: str = "local"
    port: int = APP_PORT
    log_level: str = "INFO"

    ontology_pack: str = DEFAULT_ONTOLOGY_PACK

    graph: GraphConfig = field(default_factory=GraphConfig)
    vector: VectorConfig = field(default_factory=VectorConfig)
    documents: DocumentConfig = field(default_factory=DocumentConfig)
    structured: StructuredConfig = field(default_factory=StructuredConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    tables: TableConfig = field(default_factory=TableConfig)

    internal_api_secret: str = ""
    """Shared secret the S3-notification Lambda presents to the internal ingest
    endpoint. Empty means that endpoint refuses every call: there is no user on that
    path, so being reachable only from inside the VPC is not by itself authorization."""

    mcp_url: str = ""
    """Where the retrieval agent reaches the MCP tools.

    A separate process, not this one. The MCP tool bodies are `async def` with no `await`
    inside, so their graph and Athena calls block the event loop -- an agent awaiting its own
    worker would starve the loop that has to serve it. In deployment this is a sidecar
    container on `127.0.0.1`; locally it is the `mcp` compose service.

    Empty disables the Retrieval routes, which answer 503 rather than pretending to work."""

    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    """Retrieval trust floor. Narrow it freely; widening it below the default lets
    low-confidence model output shape answers."""

    require_review_for_model_extractions: bool = True
    """The review gate. Present as a flag so it appears in an audit of what the
    system can be configured to do — and `validate` refuses to let it be off
    outside local development. Turning it off means an LLM's opinion is written
    straight into the graph as a finding."""

    def validate(self) -> None:
        """Refuse to start in a configuration that cannot be defended.

        These are startup failures rather than warnings on purpose: both of them
        are silent when wrong. Nothing in a running system looks different because
        the review gate is off — until someone asks why the graph asserted it.
        """
        is_local = self.environment == "local"

        if self.auth.dev_bypass_tenant and not is_local:
            raise ValueError(
                "AUTH_DEV_BYPASS_TENANT is set outside local development. This would "
                "serve unauthenticated requests as a real tenant."
            )

        if not self.require_review_for_model_extractions and not is_local:
            raise ValueError(
                "REQUIRE_REVIEW_FOR_MODEL_EXTRACTIONS is off outside local development. "
                "EXTRACTED_MODEL assertions would be auto-asserted, which makes an "
                "LLM's opinion indistinguishable from a parsed fact."
            )

        # Checked at startup rather than at the route, because the failure is silent: a typo
        # here matches no caller's tenant, so the platform routes 403 for everyone and look
        # like a permissions problem rather than a misconfiguration.
        if self.auth.home_tenant:
            from src.graph.scope import is_valid_tenant_id

            if not is_valid_tenant_id(self.auth.home_tenant):
                raise ValueError(
                    f"AUTH_HOME_TENANT {self.auth.home_tenant!r} is not a valid tenant id, so "
                    "it can never equal a caller's tenant and the platform routes would "
                    "refuse everyone."
                )

        if not is_local and not self.auth.issuer_url:
            raise ValueError(
                "COGNITO_ISSUER_URL is required outside local development, tenant_id "
                "comes from a verified token and there is no other source for it."
            )

        if not self.graph.iam_auth and not self.graph.password and not is_local:
            raise ValueError("GRAPH_PASSWORD is required when GRAPH_IAM_AUTH is off")


def load_config() -> LexGraphConfig:
    region = _env("AWS_DEFAULT_REGION") or _env("AWS_REGION") or DEFAULT_AWS_REGION

    cfg = LexGraphConfig(
        environment=_env("ENVIRONMENT", "local"),
        port=_env_int("PORT", APP_PORT),
        log_level=_env("LOG_LEVEL", "INFO"),
        ontology_pack=_env("ONTOLOGY_PACK", DEFAULT_ONTOLOGY_PACK),
        graph=GraphConfig(
            uri=_env("GRAPH_URI", "bolt://localhost:7687"),
            user=_env("GRAPH_USER", "neo4j"),
            password=_env("GRAPH_PASSWORD"),
            iam_auth=_env_bool("GRAPH_IAM_AUTH"),
            region=_env("GRAPH_REGION", region),
            max_connection_pool_size=_env_int("GRAPH_MAX_POOL_SIZE", 50),
        ),
        vector=VectorConfig(
            endpoint=_env("VECTOR_ENDPOINT"),
            collection=_env("VECTOR_COLLECTION"),
            region=_env("VECTOR_REGION", region),
            embedding_model=_env("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            embedding_dimensions=_env_int("EMBEDDING_DIMENSIONS", DEFAULT_EMBEDDING_DIMENSIONS),
            min_score=_env_float("VECTOR_MIN_SCORE", 0.6),
            top_k=_env_int("VECTOR_TOP_K", 20),
        ),
        documents=DocumentConfig(
            bucket=_env("DOCUMENT_BUCKET"),
            kms_key_id=_env("DOCUMENT_KMS_KEY_ID"),
            chunk_chars=_env_int("CHUNK_CHARS", DEFAULT_CHUNK_CHARS),
            chunk_overlap_chars=_env_int("CHUNK_OVERLAP_CHARS", DEFAULT_CHUNK_OVERLAP_CHARS),
            ocr_model=_env("OCR_MODEL", DEFAULT_OCR_MODEL),
            ocr_dpi=_env_int("OCR_DPI", 150),
            max_pages=_env_int("MAX_DOCUMENT_PAGES", 400),
            page_batch_size=_env_int("PAGE_BATCH_SIZE", PAGE_BATCH_SIZE),
            page_concurrency=_env_int("PAGE_CONCURRENCY", MAX_PAGE_CONCURRENCY),
            max_concurrent_ingests=_env_int("MAX_CONCURRENT_INGESTS", MAX_CONCURRENT_INGESTS),
        ),
        structured=StructuredConfig(
            glue_databases=[d.strip() for d in _env("GLUE_DATABASES").split(",") if d.strip()],
            athena_workgroup=_env("ATHENA_WORKGROUP", "primary"),
            athena_results_bucket=_env("ATHENA_RESULTS_BUCKET"),
            max_query_rows=_env_int("MAX_QUERY_ROWS", MAX_QUERY_ROWS),
        ),
        auth=AuthConfig(
            user_pool_id=_env("COGNITO_USER_POOL_ID"),
            client_id=_env("COGNITO_CLIENT_ID"),
            issuer_url=_env("COGNITO_ISSUER_URL"),
            policy_store_id=_env("POLICY_STORE_ID"),
            region=_env("COGNITO_REGION", region),
            dev_bypass_tenant=_env("AUTH_DEV_BYPASS_TENANT"),
            home_tenant=_env("AUTH_HOME_TENANT"),
        ),
        models=ModelConfig(
            extraction_model=_env("EXTRACTION_MODEL", DEFAULT_EXTRACTION_MODEL),
            synthesis_model=_env("SYNTHESIS_MODEL", DEFAULT_SYNTHESIS_MODEL),
            region=_env("BEDROCK_REGION", region),
        ),
        tables=TableConfig(
            tenants=_env("TENANT_TABLE"),
            jobs=_env("JOB_TABLE"),
            grants=_env("GRANT_TABLE"),
        ),
        internal_api_secret=_env("INTERNAL_API_SECRET"),
        mcp_url=_env("MCP_URL"),
        min_confidence=_env_float("MIN_CONFIDENCE", DEFAULT_MIN_CONFIDENCE),
        require_review_for_model_extractions=_env_bool(
            "REQUIRE_REVIEW_FOR_MODEL_EXTRACTIONS", True
        ),
    )

    cfg.validate()
    return cfg
