"""Neo4j driver wrapper, for local Neo4j and for Neptune over Bolt.

Deliberately thin. The one addition over a bare driver is `read_scoped`, which
takes a `ScopedQuery` and refuses to run a read that has not been through
`src.graph.scope`. That is the mechanism behind "an unscoped query is not
expressible" — callers outside this package have no other way in.

Neptune needs two things a local Neo4j does not, and getting either wrong produces the
*same* unhelpful error, `closed with incomplete handshake response`:

**TLS.** Neptune only speaks Bolt over TLS. A plaintext connection is closed during the
handshake, before authentication is even attempted.

**A SigV4 signature instead of a password.** There is no password. The driver sends a
signed `GET /opencypher` request, JSON-encoded, as the basic-auth credential.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from neo4j import Auth, GraphDatabase
from neo4j.auth_management import AuthManagers

from src.graph.scope import ScopedQuery

logger = logging.getLogger(__name__)

#: The IAM service name Neptune signs against. Not `neptune`.
_NEPTUNE_SERVICE = "neptune-db"

#: Neptune ignores the principal, but the Bolt handshake requires one.
_DUMMY_USERNAME = "username"

#: Engine 1.2.0.0+ expects the `/opencypher` path to be signed. Signing `/` instead
#: fails with an error that blames the signature rather than the path.
_SIGNED_PATH = "/opencypher"

#: A pooled connection authenticates once, when it is created, and a SigV4 signature is
#: valid for about five minutes. Recycling connections below that bound means every
#: handshake uses a freshly signed token instead of replaying an expired one.
NEPTUNE_MAX_CONNECTION_LIFETIME_SECONDS = 180


def neptune_auth(uri: str, region: str) -> Auth:
    """A SigV4-signed Bolt auth token for Neptune.

    Credentials are resolved on every call, not cached: the task role's credentials
    rotate, and the signature itself expires within minutes.
    """
    import boto3
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest

    credentials = boto3.Session().get_credentials()
    if credentials is None:
        raise RuntimeError("no AWS credentials available to sign a Neptune Bolt handshake")

    # The signature covers host and path, not the scheme, and botocore will not sign a
    # `bolt://` URL — so it is signed as https against the same host.
    host = uri.split("://", 1)[-1].rstrip("/")
    request = AWSRequest(method="GET", url=f"https://{host}{_SIGNED_PATH}")
    request.headers.add_header("Host", host)
    SigV4Auth(credentials.get_frozen_credentials(), _NEPTUNE_SERVICE, region).add_auth(request)

    signed = {
        header: request.headers[header]
        for header in ("Authorization", "X-Amz-Date", "X-Amz-Security-Token", "Host")
        # The security token is absent for long-lived credentials, present for a role.
        if header in request.headers
    }
    signed["HttpMethod"] = request.method
    return Auth("basic", _DUMMY_USERNAME, json.dumps(signed), "realm")


class GraphClient:
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        *,
        iam_auth: bool = False,
        region: str = "us-east-1",
    ) -> None:
        if iam_auth:
            self._driver = GraphDatabase.driver(
                uri,
                # An AuthManager, not a bare token or a plain lambda. The driver wraps
                # anything that is not an AuthManager in `AuthManagers.static`, which
                # would sign once at startup and then replay that signature forever —
                # working for about five minutes and failing with "Signature expired"
                # thereafter. `basic` re-invokes the provider when the server rejects
                # the credential, which is exactly what an expired signature does.
                auth=AuthManagers.basic(lambda: neptune_auth(uri, region)),
                encrypted=True,
                max_connection_lifetime=NEPTUNE_MAX_CONNECTION_LIFETIME_SECONDS,
            )
            logger.info("Connected to Neptune at %s (IAM auth, TLS)", uri)
        else:
            self._driver = GraphDatabase.driver(uri, auth=(user, password))
            logger.info("Connected to Neo4j at %s", uri)

    def close(self) -> None:
        self._driver.close()

    def query(self, cypher: str, params: dict | None = None) -> list[dict]:
        """Unscoped read. Admin/catalog use only — never for tenant graph data."""
        with self._driver.session() as session:
            return [r.data() for r in session.run(cypher, params or {})]

    def read_scoped(
        self, cypher_template: str, scope: ScopedQuery, params: dict[str, Any] | None = None
    ) -> list[dict]:
        """Run a read whose WHERE clause came from `src.graph.scope`.

        `cypher_template` must contain the literal token ``{scope}`` where the
        scoping predicate belongs. Refusing to interpolate anything else is what
        stops a caller from quietly dropping the tenant filter.
        """
        if "{scope}" not in cypher_template:
            raise ValueError(
                "scoped reads must contain a {scope} placeholder — see src/graph/scope.py"
            )
        cypher = cypher_template.replace("{scope}", scope.where)
        merged = {**scope.params, **(params or {})}
        with self._driver.session() as session:
            return [r.data() for r in session.run(cypher, merged)]

    def write(self, cypher: str, params: dict | None = None) -> None:
        with self._driver.session() as session:
            session.execute_write(lambda tx: tx.run(cypher, params or {}))

    def write_batch(self, cypher: str, batch: list[dict]) -> None:
        with self._driver.session() as session:
            session.execute_write(
                lambda tx: tx.run(f"UNWIND $batch AS item {cypher}", {"batch": batch})
            )

    def verify_connectivity(self) -> bool:
        try:
            self._driver.verify_connectivity()
            return True
        except Exception as e:
            logger.error("Neo4j connectivity check failed: %s", e)
            return False
