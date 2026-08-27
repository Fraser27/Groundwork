"""MCP tool tests.

Driven through a real MCP client session over the ASGI app rather than by calling the
tool functions directly. That is deliberate: the properties worth protecting here are
boundary properties — the token is read from the transport, the tier reaches the caller,
and an error message does not leak. Calling the functions in-process would test none of
those, because the transport is where the identity comes from.

The security assertions are the reason this file exists. An agent must never see more
than the lawyer driving it, so tenant isolation and ethical walls are tested through the
tools, not just through the scope module.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import anyio
import httpx
import pytest
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from src.api.deps import get_services
from src.auth import AuthError
from src.config import AuthConfig, GraphConfig, GroundworkConfig
from src.graph.assertions import EpistemicClass, SourceLocator, build_assertion
from src.graph.scope import AuthContext
from src.mcp.server import build_server, create_app

TENANT_A = "firm-acme"
TENANT_B = "firm-beta"

#: Named like a real matter rather than "M-WALLED", so an assertion about this id reaching an
#: agent is about a screen being disclosed rather than a substring appearing anywhere.
WALLED_MATTER = "M-2291"

SCREEN_REASON = "acted for the opposing party in 2024"
SCREEN_CONTACT = "risk@firm.com"

TOKEN_A = "token-for-tenant-a"
TOKEN_B = "token-for-tenant-b"
TOKEN_WALLED = "token-for-walled-user"

_CLAIMS: dict[str, dict[str, Any]] = {
    TOKEN_A: {
        "sub": "alice@acme.com",
        "custom:tenant_id": TENANT_A,
        "cognito:groups": ["reviewer"],
    },
    TOKEN_B: {"sub": "bob@beta.com", "custom:tenant_id": TENANT_B, "cognito:groups": ["reviewer"]},
    TOKEN_WALLED: {"sub": "carol@acme.com", "custom:tenant_id": TENANT_A},
}

#: Who `_stage` assigns to a new matter, so an allowlist-primary posture does not refuse
#: before the behaviour under test gets a chance to.
_USERS_BY_TENANT: dict[str, tuple[str, ...]] = {
    TENANT_A: ("alice@acme.com", "carol@acme.com"),
    TENANT_B: ("bob@beta.com",),
}


class _FakeVerifier:
    """Stands in for Cognito. Signature verification is `src/auth.py`'s job and is tested
    there; what matters here is that whatever tenant the token carries is the tenant the
    tools scope to."""

    def verify(self, token: str) -> dict[str, Any]:
        claims = _CLAIMS.get(token)
        if claims is None:
            raise AuthError("token verification failed")
        return claims


def _config(environment: str = "local", *, dev_bypass: str = "") -> GroundworkConfig:
    cfg = GroundworkConfig(
        # Pinned rather than defaulted: this file asserts the legal pack's rules and
        # vocabulary, so it must not follow a change of default pack.
        ontology_pack="legal",
        environment=environment,
        auth=AuthConfig(dev_bypass_tenant=dev_bypass, issuer_url="https://issuer.test/pool"),
        graph=GraphConfig(uri="bolt://127.0.0.1:1", user="none", password="none"),
    )
    cfg.validate()
    return cfg


def _stage(
    tenant_id: str,
    *,
    matter_id: str | None,
    confidence: float = 0.9,
    klass: EpistemicClass = EpistemicClass.DECLARED,
) -> str:
    """Stage one assertion, returning its id.

    DECLARED by default because `build_assertion` auto-asserts it, which is what the
    governed read paths (`ask` tier 2, `graph_neighbourhood`) will admit. Tests about the
    review gate pass EXTRACTED_MODEL to get a PENDING one.

    Assigns every tenant user to the matter as a side effect. Access is allowlist-primary,
    so without an assignment nothing is readable and a test about screens would pass for
    the wrong reason — the screen has to be what refuses, not the missing assignment.
    """
    services = get_services()
    if matter_id is not None:
        for user_id in _USERS_BY_TENANT[tenant_id]:
            services.authenticator.access.assign(tenant_id, user_id, matter_id, actor="owner@firm")
    ctx = AuthContext(user_id="ingest", tenant_id=tenant_id)
    a = build_assertion(
        tenant_id=tenant_id,
        subject_id="document:d1",
        predicate="CONCERNS_TOPIC",
        object_id=f"topic:{matter_id or 'none'}",
        epistemic_class=klass,
        method="cms:export@v1" if klass is EpistemicClass.DECLARED else "llm:claude-sonnet-5",
        confidence=confidence,
        source_locator=SourceLocator(
            document_id="doc-1", filename="skeleton.pdf", page=7, quote="the parties agree"
        ),
        matter_id=matter_id,
    )
    services.review_queue.stage(ctx, [a])
    return a.assertion_id


def _install(config: GroundworkConfig) -> None:
    """Install a service container, with Cognito stubbed and one screened user.

    Reaching into `_verifier` is the smallest possible seam: `Authenticator` builds it from
    config, and replacing it leaves every other path — claim extraction, access resolution,
    `AuthContext` construction — exactly as it ships. The screen is raised through
    `AccessManager` for the same reason: the reason and contact an agent is shown have to
    come through the real path, not be handed to the context directly.
    """
    create_app(config)
    services = get_services()

    # Metrics live in the graph now and the matcher is built per tenant from approved
    # definitions, so nothing loads at startup. These tests are about the MCP boundary,
    # not about where a definition is stored, so the example pack is injected directly.
    from src.api.deps import load_example_pack
    from src.metrics.models import StaticCatalog
    from src.query.metric_matcher import MetricMatcher

    pack = load_example_pack()
    if pack:
        services.metric_matcher = MetricMatcher(pack, StaticCatalog(tables={}))

    services.authenticator._verifier = _FakeVerifier()
    services.authenticator.access.screen(
        TENANT_A,
        "carol@acme.com",
        WALLED_MATTER,
        actor="risk@firm.com",
        reason=SCREEN_REASON,
        contact=SCREEN_CONTACT,
    )


@pytest.fixture(autouse=True)
def services() -> None:
    _install(_config())


def _session(token: str | None, work: Callable[[ClientSession], Awaitable[Any]]) -> Any:
    """Run `work` against a fresh MCP session over the ASGI app.

    A new app per session because a `FastMCP`'s session manager may only be run once, so
    reusing one across calls fails on the second lifespan. Services are global and outlive
    the app, which is exactly how the deployed process behaves.
    """
    app = build_server().streamable_http_app()

    async def go() -> Any:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://mcp", headers=headers
        )
        async with (
            app.router.lifespan_context(app),
            client as http,
            streamable_http_client("http://mcp/mcp", http_client=http) as (read, write, _),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            return await work(session)

    # anyio.run rather than an async test, so the suite needs no asyncio plugin.
    return anyio.run(go)


def _call(token: str | None, tool: str, args: dict[str, Any] | None = None) -> Any:
    return _session(token, lambda s: s.call_tool(tool, args or {}))


def _list_tools() -> list[Any]:
    async def work(session: ClientSession) -> list[Any]:
        return (await session.list_tools()).tools

    return _session(TOKEN_A, work)


def _text(result: Any) -> str:
    return " ".join(c.text for c in result.content if getattr(c, "text", None))


class TestToolListing:
    def test_every_tool_is_exposed(self):
        assert {t.name for t in _list_tools()} == {
            "ask",
            "compose",
            "list_metrics",
            "describe_ontology",
            "search_assertions",
            "get_provenance",
            "graph_neighbourhood",
        }

    def test_every_tool_states_its_trust_semantics(self):
        """An agent choosing between `ask` and `search_assertions` has only the description
        to go on, so "how much should I trust this" has to be in it."""
        for tool in _list_tools():
            assert "TRUST:" in (tool.description or ""), tool.name

    def test_every_parameter_tells_the_model_what_to_pass(self):
        """A parameter with no description is a parameter a model guesses at.

        `Args:` in a docstring does not reach the schema -- FastMCP reads the docstring for the
        tool description and nothing else -- so everything documented there was invisible. A
        cheap model then passed an entity id to `get_provenance`, which correctly refused it,
        and the refusal looked like a broken tool rather than a missing description.
        """
        undocumented = [
            f"{tool.name}.{name}"
            for tool in _list_tools()
            for name, spec in ((tool.inputSchema or {}).get("properties") or {}).items()
            if not spec.get("description")
        ]
        assert undocumented == []

    def test_the_two_id_shaped_arguments_say_they_are_different(self):
        """The mistake that started this. `get_provenance` takes a hex assertion id and
        `graph_neighbourhood` takes a `kind:slug` entity id, and nothing in the names says so."""
        by_name = {t.name: t.inputSchema["properties"] for t in _list_tools()}
        provenance = by_name["get_provenance"]["assertion_id"]["description"]
        neighbourhood = by_name["graph_neighbourhood"]["node_id"]["description"]

        assert "NOT an entity id" in provenance
        assert "graph_neighbourhood" in provenance
        assert "kind:slug" in neighbourhood
        assert "get_provenance" in neighbourhood

    def test_governed_and_raw_tools_read_differently(self):
        """The distinction an agent has to be able to draw without reading the source."""
        by_name = {t.name: " ".join((t.description or "").split()) for t in _list_tools()}
        assert "deterministic SQL with no model in the path" in by_name["list_metrics"]
        assert "claims still waiting for a human" in by_name["search_assertions"]


class TestCompose:
    """The lane-level tool. `ask` returns the first tier that could answer; this returns what
    every permitted lane found, kept apart, which is what an agent needs to show a trace."""

    def test_returns_the_composed_vocabulary_not_the_thin_one(self):
        """The reason this tool exists. `ask` has no lanes_run, no lanes_skipped and no parts, so
        an agent limited to it cannot say where the system looked or what it declined to merge."""
        body = _call(
            TOKEN_A, "compose", {"question": "show me fees billed by month"}
        ).structuredContent

        for field in ("parts", "lanes_run", "lanes_skipped", "governance", "fully_deterministic"):
            assert field in body, field
        # Not a renamed `ask`: the single-tier shape must not leak into this one, or a caller
        # would read one confidence for a result that deliberately has several.
        for absent in ("tier_name", "governed", "assertions_used"):
            assert absent not in body, absent

    def test_a_metric_question_stays_deterministic(self):
        body = _call(
            TOKEN_A, "compose", {"question": "show me fees billed by month", "execute": False}
        ).structuredContent

        assert body["lanes_run"] == ["metric"]
        assert body["governance"] == "governed"
        assert body["fully_deterministic"] is True
        [part] = body["parts"]
        assert (part["lane"], part["provenance"], part["tier"]) == ("metric", "deterministic", 1)
        assert "SELECT" in part["sql"]

    def test_the_floor_it_was_compared_against_is_reported(self):
        """A confidence means nothing without the threshold it was judged by, and the floor is
        per tenant, so an agent cannot assume the default."""
        body = _call(TOKEN_A, "compose", {"question": "anything"}).structuredContent
        assert body["min_confidence"] == 0.8

    def test_no_prose_unless_asked_for(self):
        """Off by default here and on for the web UI. The agent is the writer, and a second
        model's paragraph underneath its own would be an ungoverned layer nobody can attribute."""
        body = _call(
            TOKEN_A, "compose", {"question": "show me fees billed by month"}
        ).structuredContent
        assert body["synthesis"] is None

    def test_does_not_cross_tenants(self):
        _stage(TENANT_B, matter_id="M-B")
        body = _call(TOKEN_A, "compose", {"question": "which topics are covered"}).structuredContent
        ids = [i for part in body["parts"] for i in part["assertion_ids"]]
        assert ids == []

    def test_a_rendered_answer_keeps_the_structured_one(self):
        """Additive, never a substitution. Returning prose instead of the data would let a client
        show a governance label with none of the parts behind it."""
        body = _call(
            TOKEN_A,
            "compose",
            {"question": "show me fees billed by month", "response_format": "markdown"},
        ).structuredContent

        assert "governed" in body["formatted"]
        assert body["format"] == "markdown"
        # Everything a `data` caller relies on is still here.
        for field in ("parts", "lanes_run", "lanes_skipped", "governance", "min_confidence"):
            assert field in body, field

    def test_the_default_adds_no_rendering(self):
        """So an existing client sees a byte-identical response."""
        body = _call(
            TOKEN_A, "compose", {"question": "show me fees billed by month"}
        ).structuredContent
        assert "formatted" not in body
        assert "format" not in body

    def test_an_unsupported_format_is_refused_rather_than_ignored(self):
        """Silently returning `data` for a format the caller asked for would make the caller
        think it had prose to show."""
        result = _call(TOKEN_A, "compose", {"question": "x", "response_format": "yaml"})
        assert result.isError
        assert "response_format must be one of" in _text(result)


class TestAsk:
    def test_reports_the_tier_that_answered(self):
        body = _call(TOKEN_A, "ask", {"question": "show me fees billed by month"}).structuredContent
        assert body["tier_name"] == "GOVERNED_METRIC"
        assert body["governed"] is True
        assert "SELECT" in body["sql"]

    def test_reports_the_floor_it_applied(self):
        """`compose` sent this and `ask` did not, so a reader of an `ask` result had to assume a
        floor. A trace rendering "nothing cleared the floor of 0.8" from a default it invented is
        claiming a control the server may never have exercised."""
        body = _call(TOKEN_A, "ask", {"question": "show me fees billed by month"}).structuredContent
        assert "min_confidence" in body
        assert isinstance(body["min_confidence"], int | float)

    def test_carries_what_a_trace_needs(self):
        """Retrieval renders the same trace for `ask` as for `compose`, so the fields the trace
        reads must be present on both. Without them the page silently falls back to raw JSON, which
        is the bug this pins."""
        body = _call(TOKEN_A, "ask", {"question": "show me fees billed by month"}).structuredContent
        for field in ("tier", "tier_name", "governed", "blocks", "citations", "assertions_used"):
            assert field in body, field

    def test_returns_assertions_used(self):
        """The audit trail an agent's answer rests on. Without these ids, a tool answer is
        less defensible than the same answer given by a human."""
        _stage(TENANT_A, matter_id="M-1")
        body = _call(
            TOKEN_A, "ask", {"question": "which topics does document d1 concern"}
        ).structuredContent
        assert body["tier_name"] == "GRAPH_TRAVERSAL"
        assert body["assertions_used"]

    def test_every_assertion_used_resolves_to_provenance(self):
        _stage(TENANT_A, matter_id="M-1")
        body = _call(
            TOKEN_A, "ask", {"question": "which topics does document d1 concern"}
        ).structuredContent
        for aid in body["assertions_used"]:
            prov = _call(TOKEN_A, "get_provenance", {"assertion_id": aid}).structuredContent
            assert prov["assertion"]["source"]["quote"]

    def test_tiers_attempted_are_named_not_numbered(self):
        body = _call(TOKEN_A, "ask", {"question": "zzzq unrelated gibberish"}).structuredContent
        assert all(isinstance(t, str) for t in body["tiers_attempted"])

    def test_unanswerable_question_says_so(self):
        body = _call(TOKEN_A, "ask", {"question": "zzzq unrelated gibberish"}).structuredContent
        assert body["answer"] is None
        assert body["warnings"]


class TestAuthentication:
    def test_missing_token_is_refused(self):
        """No dev bypass in this config, so an unauthenticated call has no identity to
        borrow — and an agent without a user has no business in the graph."""
        result = _call(None, "list_metrics")
        assert result.isError
        assert "bearer" in _text(result).lower()

    def test_unverifiable_token_is_refused(self):
        result = _call("forged", "list_metrics")
        assert result.isError

    def test_identity_cannot_be_passed_as_an_argument(self):
        """There is no tenant or user parameter on any tool. A caller that can name its own
        identity has none."""
        for tool in _list_tools():
            properties = (tool.inputSchema or {}).get("properties", {})
            assert not {"tenant", "tenant_id", "user_id", "as_user"} & set(properties), tool.name


class TestTenantIsolation:
    def test_search_does_not_cross_tenants(self):
        _stage(TENANT_B, matter_id="M-B")
        body = _call(TOKEN_A, "search_assertions").structuredContent
        assert body["assertions"] == []

    def test_each_tenant_sees_only_its_own(self):
        _stage(TENANT_A, matter_id="M-A")
        _stage(TENANT_B, matter_id="M-B")
        a_matters = {
            r["matter_id"]
            for r in _call(TOKEN_A, "search_assertions").structuredContent["assertions"]
        }
        b_matters = {
            r["matter_id"]
            for r in _call(TOKEN_B, "search_assertions").structuredContent["assertions"]
        }
        assert a_matters == {"M-A"}
        assert b_matters == {"M-B"}

    def test_provenance_of_another_tenants_assertion_is_unreachable(self):
        """A valid assertion id is not a capability. Tenancy is a property filter, so this
        is the test that the filter is actually applied on the read."""
        foreign = _stage(TENANT_B, matter_id="M-B")
        result = _call(TOKEN_A, "get_provenance", {"assertion_id": foreign})
        assert result.isError
        assert TENANT_B not in _text(result)

    def test_neighbourhood_does_not_cross_tenants(self):
        _stage(TENANT_B, matter_id="M-B")
        body = _call(TOKEN_A, "graph_neighbourhood", {"node_id": "document:d1"}).structuredContent
        assert body["edges"] == []

    def test_cross_tenant_refusal_stays_silent(self):
        """Must not regress alongside the screen disclosure. A screen inside one firm is
        documented and named; a boundary between two firms is confidentiality, agreed with
        nobody, so it says nothing — not the matter, not the tenant, not that a screen
        exists."""
        foreign = _stage(TENANT_B, matter_id="M-B")
        message = _text(_call(TOKEN_A, "get_provenance", {"assertion_id": foreign}))
        assert TENANT_B not in message
        assert "M-B" not in message
        assert "screened" not in message.lower()
        assert message == _text(
            _call(TOKEN_A, "get_provenance", {"assertion_id": "no-such-assertion"})
        )


class TestEthicalWalls:
    """A screen inside one firm is disclosed to the agent, named, with a contact.

    The agent is the only thing standing between a filtered result and a lawyer reading it
    as a clean one, so it is told enough to say "this was not a search of everything".
    """

    def test_walled_assertion_is_still_refused(self):
        """Disclosure is not access. The screen still refuses the read."""
        walled = _stage(TENANT_A, matter_id=WALLED_MATTER)
        result = _call(TOKEN_WALLED, "get_provenance", {"assertion_id": walled})
        assert result.isError
        assert "source" not in str(result.structuredContent)

    def test_refusal_names_the_matter_the_reason_and_the_contact(self):
        walled = _stage(TENANT_A, matter_id=WALLED_MATTER)
        message = _text(_call(TOKEN_WALLED, "get_provenance", {"assertion_id": walled}))
        assert WALLED_MATTER in message
        assert SCREEN_REASON in message
        assert SCREEN_CONTACT in message

    def test_screened_reads_differently_from_absent(self):
        """The reversal. These used to be byte-identical, which is what let an agent report
        "nothing found" for a matter that exists and matched."""
        walled = _stage(TENANT_A, matter_id=WALLED_MATTER)
        walled_msg = _text(_call(TOKEN_WALLED, "get_provenance", {"assertion_id": walled}))
        absent_msg = _text(
            _call(TOKEN_WALLED, "get_provenance", {"assertion_id": "no-such-assertion"})
        )
        assert walled_msg != absent_msg
        assert "screened" in walled_msg.lower()

    def test_an_unknown_id_still_reveals_nothing(self):
        """Only screens are disclosed. An id that does not exist must not become a way to
        enumerate the ones that do."""
        message = _text(
            _call(TOKEN_WALLED, "get_provenance", {"assertion_id": "no-such-assertion"})
        )
        assert "screened" not in message.lower()
        assert WALLED_MATTER not in message

    def test_walled_matter_is_absent_from_search_results(self):
        """Still filtered out of the rows — `withheld_matters` is what reports it."""
        _stage(TENANT_A, matter_id=WALLED_MATTER)
        _stage(TENANT_A, matter_id="M-OPEN")
        body = _call(TOKEN_WALLED, "search_assertions").structuredContent
        assert {r["matter_id"] for r in body["assertions"]} == {"M-OPEN"}

    def test_search_names_what_it_withheld(self):
        """The counterweight to silent filtering, and the change: named, not counted. A count
        does not tell the lawyer which client to go and ask about."""
        _stage(TENANT_A, matter_id=WALLED_MATTER)
        body = _call(TOKEN_WALLED, "search_assertions").structuredContent
        assert body["withheld_matter_count"] == 1
        assert body["withheld_matters"] == [
            {
                "matter_id": WALLED_MATTER,
                "reason": SCREEN_REASON,
                "contact": SCREEN_CONTACT,
            }
        ]

    def test_filtering_by_a_screened_matter_says_it_is_screened(self):
        """It used to look exactly like a matter holding nothing. That is the reading that
        turns a screen into a false negative on a conflict check."""
        _stage(TENANT_A, matter_id=WALLED_MATTER)
        walled = _call(
            TOKEN_WALLED, "search_assertions", {"matter_id": WALLED_MATTER}
        ).structuredContent
        unknown = _call(
            TOKEN_WALLED, "search_assertions", {"matter_id": "M-9999"}
        ).structuredContent
        assert walled["assertions"] == []
        assert walled["withheld_matters"]
        assert unknown["withheld_matters"] != [] or walled != unknown

    def test_an_unwalled_user_is_told_nothing_is_withheld(self):
        body = _call(TOKEN_A, "search_assertions").structuredContent
        assert body["withheld_matter_count"] == 0
        assert body["withheld_matters"] == []

    def test_the_tool_description_tells_an_agent_to_relay_a_screen(self):
        """An agent will not read this file. If the instruction to pass the screen on is not
        in the description, the disclosure stops at the tool boundary."""
        search = next(t for t in _list_tools() if t.name == "search_assertions")
        assert "withheld_matters" in (search.description or "")

    def test_an_unwalled_user_reaches_the_same_assertion(self):
        """Confirms the previous tests fail for the right reason — the assertion exists and
        is reachable, just not by the walled user."""
        walled = _stage(TENANT_A, matter_id=WALLED_MATTER)
        result = _call(TOKEN_A, "get_provenance", {"assertion_id": walled})
        assert not result.isError
        assert result.structuredContent["assertion"]["matter_id"] == WALLED_MATTER


class TestDevBypass:
    def test_config_refuses_the_bypass_outside_local(self):
        """First of two gates. `GroundworkConfig.validate` will not let the process start."""
        with pytest.raises(ValueError, match="AUTH_DEV_BYPASS_TENANT"):
            _config("production", dev_bypass="dev-tenant")

    def test_bypass_is_not_honoured_outside_local(self):
        """Second gate, independent of the first: even if a config reached the runtime with
        the bypass set, `_dev_context` re-checks the environment. Either alone would do;
        this is the one that would still hold if config loading ever changed.

        Set after `validate()` on purpose — that is the only way to reach the state this
        test is about.
        """
        cfg = _config("production")
        cfg.auth.dev_bypass_tenant = "dev-tenant"
        _install(cfg)

        result = _call(None, "list_metrics")
        assert result.isError
        assert "bearer" in _text(result).lower()

    def test_bypass_serves_an_unauthenticated_call_locally(self):
        """The bypass has to actually work, or local development invents its own auth."""
        _install(_config("local", dev_bypass="dev-tenant"))
        body = _call(None, "list_metrics").structuredContent
        assert body["metrics"]


class TestSearchAssertions:
    def test_rejects_an_unknown_review_state(self):
        result = _call(TOKEN_A, "search_assertions", {"review_state": "MAYBE"})
        assert result.isError
        assert "PENDING" in _text(result)

    def test_rejects_an_unknown_epistemic_class(self):
        result = _call(TOKEN_A, "search_assertions", {"epistemic_class": "GUESSED"})
        assert result.isError

    def test_flags_rows_below_the_confidence_floor(self):
        """A row below the floor is not shaping any answer, and an agent repeating it as a
        finding is the failure the floor exists to prevent."""
        _stage(
            TENANT_A,
            matter_id="M-1",
            confidence=0.4,
            klass=EpistemicClass.EXTRACTED_MODEL,
        )
        row = _call(TOKEN_A, "search_assertions").structuredContent["assertions"][0]
        assert row["below_floor"] is True
        assert row["review_state"] == "PENDING"

    def test_limit_is_capped(self):
        for i in range(3):
            _stage(TENANT_A, matter_id=f"M-{i}")
        body = _call(TOKEN_A, "search_assertions", {"limit": 10_000}).structuredContent
        assert len(body["assertions"]) <= 200


class TestProvenance:
    def test_returns_file_page_and_quote(self):
        """File, page, quote — what a lawyer would use to check it by hand."""
        aid = _stage(TENANT_A, matter_id="M-1")
        source = _call(TOKEN_A, "get_provenance", {"assertion_id": aid}).structuredContent[
            "assertion"
        ]["source"]
        assert (source["filename"], source["page"]) == ("skeleton.pdf", 7)
        assert source["quote"] == "the parties agree"

    def test_explanation_avoids_jargon(self):
        aid = _stage(TENANT_A, matter_id="M-1", klass=EpistemicClass.EXTRACTED_MODEL)
        explanation = _call(TOKEN_A, "get_provenance", {"assertion_id": aid}).structuredContent[
            "explanation"
        ]
        assert "AI model" in explanation
        assert "epistemic" not in explanation.lower()

    def test_inference_unwinds_into_a_proof_tree(self):
        premise = _stage(TENANT_A, matter_id="M-1")
        services = get_services()
        ctx = AuthContext(user_id="ingest", tenant_id=TENANT_A)
        inferred = build_assertion(
            tenant_id=TENANT_A,
            subject_id="matter:M-1",
            predicate="ADVERSE_TO",
            object_id="party:beta",
            epistemic_class=EpistemicClass.INFERRED,
            method="rule:conflict_check@v2",
            confidence=0.85,
            source_locator=SourceLocator(source_id="reasoner", table="assertions"),
            premises=(premise,),
            premise_confidences=(0.9,),
            rule_id="conflict_check",
            rule_version="v2",
            matter_id="M-1",
        )
        services.review_queue.stage(ctx, [inferred])

        body = _call(
            TOKEN_A, "get_provenance", {"assertion_id": inferred.assertion_id}
        ).structuredContent
        assert body["rule_id"] == "conflict_check"
        assert [p["assertion_id"] for p in body["premises"]] == [premise]
        assert body["premises"][0]["visible"] is True

    def test_screened_premise_is_reported_as_a_named_gap(self):
        """A partial proof tree that looks complete is worse than one that admits a gap, and a
        gap that names the screen is actionable rather than a dead end."""
        walled_premise = _stage(TENANT_A, matter_id=WALLED_MATTER)
        services = get_services()
        ctx = AuthContext(user_id="ingest", tenant_id=TENANT_A)
        inferred = build_assertion(
            tenant_id=TENANT_A,
            subject_id="matter:M-OPEN",
            predicate="ADVERSE_TO",
            object_id="party:beta",
            epistemic_class=EpistemicClass.INFERRED,
            method="rule:conflict_check@v2",
            confidence=0.85,
            source_locator=SourceLocator(source_id="reasoner", table="assertions"),
            premises=(walled_premise,),
            premise_confidences=(0.9,),
            rule_id="conflict_check",
            rule_version="v2",
        )
        services.review_queue.stage(ctx, [inferred])

        body = _call(
            TOKEN_WALLED, "get_provenance", {"assertion_id": inferred.assertion_id}
        ).structuredContent
        gap = body["premises"][0]
        assert gap["assertion_id"] == walled_premise
        assert gap["visible"] is False
        assert gap["screened"] is True
        assert gap["matter_id"] == WALLED_MATTER
        assert gap["contact"] == SCREEN_CONTACT
        # Still withheld: a named gap is not a readable premise.
        assert "source" not in gap


class TestDescribeOntology:
    def test_splits_governing_from_descriptive(self):
        body = _call(TOKEN_A, "describe_ontology").structuredContent
        assert any(p["id"] == "ADVERSE_TO" for p in body["governing_predicates"])
        assert any(p["id"] == "CONCERNS_TOPIC" for p in body["descriptive_predicates"])

    def test_rules_carry_their_method_string(self):
        """The `method` a rule's INFERRED assertions will carry, so a proof tree can be
        traced back to the rule version that produced it."""
        rules = _call(TOKEN_A, "describe_ontology").structuredContent["rules"]
        assert all("@" in r["method"] for r in rules)


class TestGraphNeighbourhood:
    def test_returns_edges_around_a_node(self):
        _stage(TENANT_A, matter_id="M-1")
        body = _call(TOKEN_A, "graph_neighbourhood", {"node_id": "document:d1"}).structuredContent
        assert body["edges"]
        assert body["confidence_floor"] == 0.8

    def test_depth_is_bounded(self):
        result = _call(TOKEN_A, "graph_neighbourhood", {"node_id": "document:d1", "depth": 9})
        assert result.isError

    def test_the_source_document_reaches_what_was_extracted_from_it(self):
        """Third caller of `expand`, and it must agree with `/query` and `/query/compose`. An agent
        handed a `document_id` from `get_provenance` gets the facts that document produced, not an
        empty neighbourhood."""
        _stage(TENANT_A, matter_id="M-1")
        result = _call(TOKEN_A, "graph_neighbourhood", {"node_id": "document:doc-1"})
        assert [e["predicate"] for e in result.structuredContent["edges"]] == ["CONCERNS_TOPIC"]

    def test_below_floor_edges_are_excluded(self):
        """`graph_neighbourhood` is the defensible view; `search_assertions` is the raw one.
        A weak claim appearing here would blur that distinction."""
        _stage(TENANT_A, matter_id="M-1", confidence=0.4)
        body = _call(TOKEN_A, "graph_neighbourhood", {"node_id": "document:d1"}).structuredContent
        assert body["edges"] == []


class TestKillSwitch:
    def test_an_unanswerable_question_is_not_an_error(self):
        """An agent must be able to tell "nobody could answer that" from "you may not ask that".
        The switch refuses model-written SQL, and there is none to refuse until tier 3 generates
        it, so this comes back as an empty answer rather than a tool error."""
        get_services().settings_for(TENANT_A).block_ungoverned_queries = True
        result = _call(TOKEN_A, "ask", {"question": "zzzq unrelated gibberish"})
        assert not result.isError

    def test_governed_metric_is_unaffected(self):
        get_services().settings_for(TENANT_A).block_ungoverned_queries = True
        body = _call(TOKEN_A, "ask", {"question": "show me fees billed by month"}).structuredContent
        assert body["tier_name"] == "GOVERNED_METRIC"
