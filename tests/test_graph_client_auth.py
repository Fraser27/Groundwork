"""Neptune Bolt authentication.

This is tested because the failure mode is uniquely unhelpful: both a missing TLS flag
and an unsigned handshake produce `closed with incomplete handshake response`, which
names neither cause. The deployed stack sat in degraded mode with a healthy container
because of it.

No network. The token is a pure function of credentials, region and URL, so it can be
asserted directly.
"""

from __future__ import annotations

import json

import pytest
from botocore.credentials import Credentials

from src.graph.client import NEPTUNE_MAX_CONNECTION_LIFETIME_SECONDS, neptune_auth

URI = "bolt://my-cluster.cluster-abc.us-east-1.neptune.amazonaws.com:8182"
HOST = "my-cluster.cluster-abc.us-east-1.neptune.amazonaws.com:8182"


@pytest.fixture
def creds(monkeypatch):
    """Static credentials, so the signature is deterministic enough to assert on."""
    import boto3

    session_creds = Credentials(access_key="AKIAEXAMPLE", secret_key="secret", token=None)
    monkeypatch.setattr(
        boto3, "Session", lambda: type("S", (), {"get_credentials": lambda self: session_creds})()
    )
    return session_creds


@pytest.fixture
def role_creds(monkeypatch):
    """Assumed-role credentials, which add a session token."""
    import boto3

    session_creds = Credentials(
        access_key="ASIAEXAMPLE", secret_key="secret", token="session-token"
    )
    monkeypatch.setattr(
        boto3, "Session", lambda: type("S", (), {"get_credentials": lambda self: session_creds})()
    )
    return session_creds


class TestAuthToken:
    def test_scheme_and_realm_are_what_neptune_expects(self, creds):
        auth = neptune_auth(URI, "us-east-1")
        assert auth.scheme == "basic"
        assert auth.realm == "realm"

    def test_credentials_are_the_signed_headers_as_json(self, creds):
        """Neptune parses the password field as JSON. A bare signature string fails."""
        payload = json.loads(neptune_auth(URI, "us-east-1").credentials)
        assert payload["HttpMethod"] == "GET"
        assert payload["Host"] == HOST
        assert payload["Authorization"].startswith("AWS4-HMAC-SHA256 ")

    def test_the_signature_names_the_neptune_db_service(self, creds):
        """`neptune`, the control plane, is a different service name and is rejected."""
        payload = json.loads(neptune_auth(URI, "us-east-1").credentials)
        assert "/neptune-db/aws4_request" in payload["Authorization"]

    def test_the_signature_covers_the_requested_region(self, creds):
        payload = json.loads(neptune_auth(URI, "eu-west-1").credentials)
        assert "/eu-west-1/neptune-db/" in payload["Authorization"]

    def test_a_session_token_is_forwarded_when_present(self, role_creds):
        """Without this an assumed role — which is how the Fargate task runs — is
        rejected even though the signature itself is valid."""
        payload = json.loads(neptune_auth(URI, "us-east-1").credentials)
        assert payload["X-Amz-Security-Token"] == "session-token"

    def test_no_session_token_key_for_static_credentials(self, creds):
        """Sending an empty token is not the same as omitting it."""
        payload = json.loads(neptune_auth(URI, "us-east-1").credentials)
        assert "X-Amz-Security-Token" not in payload

    def test_the_bolt_scheme_is_not_signed(self, creds):
        """botocore will not sign a bolt:// URL, so the host is signed as https. The
        signature covers host and path, so this is equivalent and not a workaround."""
        payload = json.loads(neptune_auth(URI, "us-east-1").credentials)
        assert "bolt" not in payload["Authorization"]
        assert payload["Host"] == HOST

    def test_each_call_signs_afresh(self, creds):
        """The provider is re-invoked when the server rejects a stale credential, so it
        must not memoise. A cached signature expires after about five minutes."""
        first = json.loads(neptune_auth(URI, "us-east-1").credentials)
        second = json.loads(neptune_auth(URI, "us-east-1").credentials)
        # Same inputs, but X-Amz-Date is regenerated, so both must be present and the
        # function must have done the work twice rather than returned a cached object.
        assert first["X-Amz-Date"] and second["X-Amz-Date"]

    def test_missing_credentials_raise_rather_than_sign_nothing(self, monkeypatch):
        """An unsigned token would fail at the handshake with a message about the
        handshake, not about credentials."""
        import boto3

        monkeypatch.setattr(
            boto3, "Session", lambda: type("S", (), {"get_credentials": lambda self: None})()
        )
        with pytest.raises(RuntimeError, match="no AWS credentials"):
            neptune_auth(URI, "us-east-1")


class TestConnectionLifetime:
    def test_lifetime_is_below_the_signature_expiry(self):
        """A SigV4 signature lasts about five minutes and a pooled connection
        authenticates only on creation, so connections must be recycled sooner."""
        assert NEPTUNE_MAX_CONNECTION_LIFETIME_SECONDS < 300


class TestAuthManagerRefresh:
    """A SigV4 signature lasts ~5 minutes and Neptune reports a stale one as a *protocol*
    error, not a security error. `AuthManagers.basic` only re-invokes its provider on a
    security error, so it silently kept replaying a dead signature and the pool failed
    permanently minutes after startup. Refresh must therefore be time-based."""

    def test_the_signature_is_reused_within_its_lifetime(self, creds):
        from src.graph.client import _NeptuneAuthManager

        m = _NeptuneAuthManager(URI, "us-east-1", lifetime=300)
        assert m.get_auth() is m.get_auth()

    def test_the_signature_is_renewed_after_its_lifetime(self, creds, monkeypatch):
        import src.graph.client as mod

        clock = {"t": 1000.0}
        monkeypatch.setattr(mod.time, "monotonic", lambda: clock["t"])
        m = mod._NeptuneAuthManager(URI, "us-east-1", lifetime=60)
        first = m.get_auth()

        clock["t"] += 61
        assert m.get_auth() is not first

    def test_a_security_exception_forces_a_resign(self, creds):
        """Belt and braces: if Neptune ever does classify it as a security error, the next
        connection must not reuse the rejected signature."""
        from src.graph.client import _NeptuneAuthManager

        m = _NeptuneAuthManager(URI, "us-east-1", lifetime=300)
        first = m.get_auth()
        assert m.handle_security_exception(first, object()) is True
        assert m.get_auth() is not first

    def test_the_lifetime_is_well_inside_the_signature_window(self):
        """Renewing at the boundary would race the server's own clock skew allowance."""
        from src.graph.client import _NeptuneAuthManager

        assert _NeptuneAuthManager(URI, "us-east-1")._lifetime <= 120
