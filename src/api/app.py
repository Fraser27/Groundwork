"""FastAPI application.

`docker-compose.yml` points `APP_MODULE` here. Nothing in this module knows how to
answer a question — it wires dependencies and maps the domain's exceptions onto HTTP
status codes.

One mapping is a deliberate policy rather than a convention: a `ScopeViolation` for an
in-tenant ethical screen becomes **403, naming the matter, the reason and a contact**;
every other scope violation stays **404**. A screen is something the firm documented and
acknowledged, so hiding it behind "not found" misleads the person it was agreed with. A
cross-tenant refusal is a confidentiality boundary between firms and stays silent.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api import (
    routes_access,
    routes_catalog,
    routes_documents,
    routes_governance,
    routes_metrics,
    routes_query,
    routes_review,
    routes_tenants,
)
from src.api.deps import Services, build_services, get_services, set_services
from src.auth import AuthError
from src.config import LexGraphConfig
from src.graph.assertions import AssertionError_
from src.graph.scope import ScopeViolation
from src.ontology.loader import load_ontology

logger = logging.getLogger("lexgraph")


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    services: Services = get_services()
    cfg = services.config
    _configure_logging(cfg.log_level)

    logger.info(
        "starting lexgraph env=%s ontology=%s graph=%s",
        cfg.environment,
        cfg.ontology_pack,
        cfg.graph.uri,
    )
    if services.authenticator.dev_mode:
        logger.warning(
            "DEV AUTH BYPASS ACTIVE, all requests served as tenant %r",
            cfg.auth.dev_bypass_tenant,
        )

    # Connect lazily: a graph that is not up yet must not stop the API from starting,
    # or /health can never report why it is down.
    if services.graph is None:
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
            if client.verify_connectivity():
                init_schema(client, is_neptune=cfg.graph.iam_auth)
                services.graph = client

                # Swap the assertion store onto the graph now that one is reachable. Done
                # here rather than in `build_services` because the client does not exist
                # until this point, and connecting lazily is what lets /health report a
                # graph that is down instead of the API failing to start.
                #
                # Until this line existed, an approval wrote to a dict and was lost on the
                # next deploy while the UI reported success.
                from src.graph.assertion_store import GraphAssertionStore

                services.review_queue.store = GraphAssertionStore(graph=client)
                logger.info("graph connected, assertions persisted to the graph")
            else:
                logger.warning("graph unreachable at %s, degraded mode", cfg.graph.uri)
        except Exception as e:
            logger.warning("graph init failed (%s), degraded mode", e)

    yield

    if services.graph is not None:
        try:
            services.graph.close()
        except Exception:
            logger.debug("graph close failed", exc_info=True)


def create_app(config: LexGraphConfig | None = None) -> FastAPI:
    services = build_services(config)
    set_services(services)

    app = FastAPI(
        title="LexGraph",
        description=(
            "A governed semantic layer over structured and unstructured data. Every "
            "fact carries its provenance."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    origins = ["http://localhost:5173", "http://localhost:3000"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins if services.config.environment == "local" else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(ScopeViolation)
    async def _scope(_: Request, exc: ScopeViolation) -> JSONResponse:
        # Structured rather than a bare detail string, so the UI renders the contact as a
        # contact instead of parsing it back out of a sentence.
        if exc.is_screen:
            return JSONResponse(
                status_code=403,
                content={
                    "detail": str(exc),
                    # `is not None`, not truthiness: `AccessDecision.__bool__` is False for
                    # every refusal, so a plain `if` blanks the field it is meant to name.
                    "decision": exc.decision.value if exc.decision is not None else None,
                    "matter_id": exc.matter_id,
                    "reason": exc.reason,
                    "contact": exc.contact,
                },
            )
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(AuthError)
    async def _auth(_: Request, exc: AuthError) -> JSONResponse:
        return JSONResponse(status_code=401, content={"detail": str(exc)})

    @app.exception_handler(AssertionError_)
    async def _assertion(_: Request, exc: AssertionError_) -> JSONResponse:
        # A contract violation is the caller's fault, and the message explains which
        # invariant was broken — worth passing through rather than flattening.
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/health", tags=["ops"])
    async def health() -> dict[str, Any]:
        svc = get_services()
        graph_ok = svc.graph is not None and svc.graph.verify_connectivity()
        return {
            "status": "ok" if graph_ok else "degraded",
            "environment": svc.config.environment,
            "graph": "connected" if graph_ok else "disconnected",
            "vector": "enabled" if svc.config.vector.enabled else "disabled",
            "ontology": svc.ontology.domain,
            "dev_auth_bypass": svc.authenticator.dev_mode,
            # So the UI knows whether to offer the platform screen. Not a secret: the routes
            # enforce the guard server-side regardless, and hiding a nav entry nobody can use
            # is a courtesy rather than a control.
            "home_tenant": svc.config.auth.home_tenant,
        }

    @app.get("/api/health", tags=["ops"])
    async def api_health() -> dict[str, Any]:
        """The UI calls /api/health; ops probes call /health."""
        return await health()

    for module in (
        routes_review,
        routes_governance,
        routes_access,
        routes_catalog,
        routes_documents,
        routes_metrics,
        routes_query,
        routes_tenants,
    ):
        app.include_router(module.router, prefix="/api")

    return app


app = create_app()

__all__ = ["app", "create_app", "load_ontology"]
