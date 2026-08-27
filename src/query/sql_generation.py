"""SQL a model writes, over a catalog it was handed.

Modelled on `synthesis.py`, for the same reason: a generator that could reach the catalog itself
would be one refactor away from widening its own allowlist. So `tables` is a **required keyword**
and there is no default -- no call path can generate SQL without naming the schema it may use.

The prompt is not the control. Everything it asks for is also enforced at the AST by
`SQLFirewall`, with `require_aggregate` and `require_limit` on, and the allowlist built from *these
tables* rather than the tenant's whole catalog. So "the model named a table it was not offered" is
unexecutable rather than discouraged, and `SELECT * FROM matters` is refused by the firewall even
though `matters` is a legitimate table. A prompt instruction is a request.

**Tenant scoping here is the allowlist and nothing else.** The demo warehouse has no tenant
column -- `sample/glue/load_firm_metadata.sh` creates `clients`, `matters` and `time_entries` for a
single firm -- so there is nothing for an injected predicate to filter on, and one added anyway
would either fail to parse or match no rows, which is worse than absent because it would look like
isolation. The allowlist comes from `catalog.tables(tenant_id)`, so it is per-request and
per-tenant, and that is honest for a single-firm warehouse and must not be read as multi-tenant
isolation. Cross-tenant isolation for ad-hoc SQL needs a tenant column in the warehouse or a
per-tenant Glue database. It is also the reason the aggregate rule is not optional: rows from
Athena carry no matter id, so `blocks.py` cannot screen them, and an aggregate is the only shape
that cannot return one walled matter's row.

The gap that remains, stated because it must surface rather than hide: **the firewall validates
tables, not columns.** A hallucinated column reaches Athena and errors there. That error has to be
shown beside the SQL, never turned into an empty result -- an empty result reads as "no data",
which is exactly the silent failure `scope.py` exists to prevent.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from src.query.firewall import DEFAULT_ROW_LIMIT

logger = logging.getLogger(__name__)

MAX_TOKENS = 1000
"""A single query. A model needing more than this is not writing SQL."""

MAX_TABLES = 20
"""Tables offered in one prompt. Beyond this the schema crowds out the question, and a catalog
scan takes every visible database -- so an unrelated `nyc-yellow_taxi` should not be able to push
`groundwork_legal.matters` out of the window."""

SYSTEM_PROMPT = f"""You write a single Trino SQL query for a lawyer's question, against the \
tables given to you. You are handed the schema; you cannot look anything up.

Absolute rules, all of them enforced after you by a firewall that will reject the query:
- Use ONLY the tables listed. Naming any other table is rejected outright, so do not guess at a \
table that "should" exist.
- Use ONLY the columns listed under each table. A column that is not there does not exist.
- The query MUST aggregate: it needs COUNT, SUM, AVG, MIN, MAX or a GROUP BY. A query returning \
individual rows is rejected, so answer with totals, counts or per-group figures.
- The query MUST end with a LIMIT of {DEFAULT_ROW_LIMIT} or fewer.
- One SELECT statement. No INSERT, UPDATE, DELETE, CREATE or anything else that writes.

Return the SQL and nothing else: no explanation, no commentary, no markdown fence. If the tables \
given cannot answer the question, return exactly NO_QUERY -- that is a useful answer, and a query \
over the wrong column is not."""


class BedrockLike(Protocol):
    def converse(self, **kwargs: Any) -> dict[str, Any]: ...


#: What the model says when the schema cannot answer the question. Honoured rather than treated as
#: a failure: a wrong query over the wrong column is worse than declining.
NO_QUERY = "NO_QUERY"

_FENCE = re.compile(r"^```(?:sql)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class GeneratedSQL:
    sql: str
    tables_offered: tuple[str, ...]
    """What the prompt was given, which is also what the firewall's allowlist was built from.
    Recorded on the result so an audit row can answer "what could this query have touched"."""

    model_id: str


def _column(column: Any) -> str:
    """One column line. The markers matter to a query planner: a partition column is how a scan
    stays cheap, and a primary key is how a join is written correctly."""
    parts = [f"{getattr(column, 'name', '')} {getattr(column, 'data_type', '')}".strip()]
    if getattr(column, "is_primary_key", False):
        parts.append("PK")
    if getattr(column, "is_partition", False):
        parts.append("PARTITION")
    if description := getattr(column, "description", ""):
        parts.append(f"-- {description}")
    return "  " + " ".join(parts)


def _offered(tables: Sequence[Any]) -> Sequence[Any]:
    """The window, and a warning naming what fell outside it.

    A silent cap reads as "covered everything", so a question the dropped tables would have
    answered looks like a question the schema could not answer.
    """
    if len(tables) > MAX_TABLES:
        logger.warning(
            "schema window full: offering %d of %d matched tables, %d dropped (%s)",
            MAX_TABLES,
            len(tables),
            len(tables) - MAX_TABLES,
            ", ".join(str(getattr(t, "full_name", "")) for t in tables[MAX_TABLES:]),
        )
    return tables[:MAX_TABLES]


def build_prompt(question: str, *, tables: Sequence[Any]) -> str:
    """The user turn: the question, and the schema it may use. Nothing else is retrievable."""
    blocks = []
    for table in tables[:MAX_TABLES]:
        header = getattr(table, "full_name", "")
        if description := getattr(table, "description", ""):
            header = f"{header}  -- {description}"
        columns = [_column(c) for c in getattr(table, "columns", ())]
        blocks.append("\n".join([f"TABLE {header}", *columns]))
    return json.dumps(
        {
            "question": question,
            "schema": "\n\n".join(blocks),
            "dialect": "trino",
            "row_limit": DEFAULT_ROW_LIMIT,
        },
        indent=2,
        default=str,
    )


@dataclass
class SqlGenerator:
    """Writes one query over the schema it is handed. Injectable client, so tests need no AWS."""

    model_id: str
    bedrock: BedrockLike | None = None
    bedrock_factory: Callable[[], BedrockLike] | None = None
    max_tokens: int = MAX_TOKENS
    temperature: float | None = None
    """None by default. Sonnet 5 -- the configured default -- answers `ValidationException:
    temperature is deprecated for this model`, which silently failed every summary this deployment
    attempted before `synthesis.py` stopped sending it. Do not set this without checking the model
    accepts one."""

    @property
    def client(self) -> BedrockLike:
        if self.bedrock is None:
            factory = self.bedrock_factory
            if factory is None:
                import boto3

                def factory() -> BedrockLike:
                    return boto3.client("bedrock-runtime")

            self.bedrock = factory()
        return self.bedrock

    @property
    def method(self) -> str:
        return f"llm:{self.model_id}"

    def generate(self, question: str, *, tables: Sequence[Any]) -> GeneratedSQL | None:
        """A query, or None when there is nothing to write one over and nothing to say.

        None rather than raising, on every failure path: the SQL lane is one lane of an answer that
        also carries passages and graph facts, and a Bedrock outage must not take those down. The
        reason reaches the log, and the caller reports the lane as skipped.
        """
        if not tables:
            return None
        offered = _offered(tables)

        # Converse, not InvokeModel: the model is admin-selectable across Nova and Anthropic,
        # and their native request bodies are mutually invalid.
        inference: dict[str, Any] = {"maxTokens": self.max_tokens}
        if self.temperature is not None:
            inference["temperature"] = self.temperature

        try:
            response = self.client.converse(
                modelId=self.model_id,
                system=[{"text": SYSTEM_PROMPT}],
                messages=[
                    {"role": "user", "content": [{"text": build_prompt(question, tables=offered)}]}
                ],
                inferenceConfig=inference,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("SQL generation could not reach %s: %s", self.model_id, e)
            return None

        blocks = response.get("output", {}).get("message", {}).get("content") or []
        text = "".join(part.get("text", "") for part in blocks).strip()
        sql = _FENCE.sub("", text).strip().rstrip(";").strip()
        if not sql or sql.upper().startswith(NO_QUERY):
            logger.info("SQL generation declined: the offered schema does not answer the question")
            return None

        return GeneratedSQL(
            sql=sql,
            tables_offered=tuple(str(getattr(t, "full_name", "")) for t in offered),
            model_id=self.model_id,
        )


#: Word characters. `full_name` is `groundwork_legal.matters`, which `str.split()` leaves as a single
#: token -- so a question asking about "matters" never matched its own table, and the catalog lane
#: only ever fired on a description word. Splitting on the punctuation is the fix.
_WORDS = re.compile(r"[^a-z0-9]+")


def _terms(text: str) -> set[str]:
    """Words worth matching on, singularised.

    The trailing `s` goes because a lawyer asks about "matters" and the table is `matter` as often
    as the reverse, and a schema lane that misses on plurality would look like a schema lane that
    found nothing -- which is the failure this repo keeps trying to make impossible.
    """
    words = {w for w in _WORDS.split(text.lower()) if len(w) > 3}
    return words | {w.rstrip("s") for w in words}


def _match(terms: set[str], table: Any, synonyms: Sequence[str]) -> tuple[int, int, int]:
    """How strongly a question reaches one table, split by where the words were found.

    Split rather than summed so the ordering is defensible: a question naming the table beats one
    that only reached it through an approved synonym, whatever the counts.
    """
    return (
        len(terms & _terms(str(getattr(table, "name", "")))),
        len(terms & _terms(str(getattr(table, "description", "")))),
        len(terms & _terms(" ".join(synonyms))) if synonyms else 0,
    )


def relevant_tables(
    question: str,
    tables: Sequence[Any],
    *,
    synonyms: Mapping[str, Sequence[str]] | None = None,
) -> list[Any]:
    """Which catalogued tables a question is about, by word overlap on name, description and any
    approved synonym.

    Crude, and shared with the planner's catalog lane on purpose: the schema a reader is shown and
    the schema the generator was given must be the same list, or the trace explains a query that
    was written over something else. It is also what keeps a Glue scan's unrelated databases out of
    the prompt -- the scan takes every visible database, so a real account holds tables from four
    other projects, and a legal question shares no words with any of them.

    Matched on `name` and `description` rather than `full_name`: the database name is the same for
    every table in it, so `groundwork_legal` would match any question mentioning legal work and offer
    the model everything.

    `synonyms` is what closes the gap word overlap alone leaves open: "how many items did shoppers
    send back" shares nothing with `returns  -- return requests and refund decisions`, so the lane
    returned no candidates and generated no SQL at all -- a silent empty. An approved `HAS_SYNONYM`
    claim is a human saying those are the same thing, so it counts for matching exactly as a name or
    description word does. Keyed by `full_name`, never by graph entity id: ids are built and never
    parsed, and this module must not learn their format.

    Ordered strongest first, so `MAX_TABLES` drops the weakest matches rather than whichever the
    catalog happened to list last. Ties keep catalog order.
    """
    terms = _terms(question)
    if not terms:
        return []
    scored = []
    for t in tables:
        score = _match(terms, t, (synonyms or {}).get(str(getattr(t, "full_name", "")), ()))
        if any(score):
            scored.append((score, t))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [t for _, t in scored]


@dataclass(frozen=True)
class SqlLaneResult:
    """What the SQL lane produced, including the ways it failed.

    `error` is carried rather than swallowed because a hallucinated column errors at Athena, and
    reporting that as no rows would read as "no data" -- the silent empty this whole path exists to
    make impossible. A result with an error and no rows is still a result.
    """

    generated: GeneratedSQL
    rows: dict[str, Any] | None = None
    error: str | None = None
    error_code: str | None = None
    """`blocked` when the firewall refused it, otherwise whatever Athena said."""


@dataclass
class SqlLane:
    """Generate, screen and run. The one path both `/query` and `/query/compose` take.

    Two endpoints disagreeing about whether a question got model-written SQL would make
    `governed` mean different things depending on which was asked, which is a bug class this repo
    has hit repeatedly. So the orchestration lives here rather than twice.
    """

    generator: Any
    executor_factory: Callable[[set[str]], Any | None] | None = None
    """Built with the allowlist for *this* request: the offered tables and nothing else. None, or a
    factory returning None, generates the SQL and does not run it -- which is still the reviewable
    artefact, the same way a compiled metric with no executor is."""

    def run(
        self,
        question: str,
        *,
        tables: Sequence[Any],
        synonyms: Mapping[str, Sequence[str]] | None = None,
    ) -> SqlLaneResult | None:
        candidates = relevant_tables(question, tables, synonyms=synonyms)
        if not candidates:
            return None
        generated = self.generator.generate(question, tables=candidates)
        if generated is None:
            return None

        executor = None
        if self.executor_factory is not None:
            # Scoped to what the prompt was given, not to the tenant's catalog. The difference is
            # the whole point: a table the model was not offered is unexecutable rather than
            # merely discouraged.
            executor = self.executor_factory(set(generated.tables_offered))
        if executor is None:
            return SqlLaneResult(generated=generated)

        result = executor.execute(generated.sql)
        if not result.success:
            logger.warning("generated SQL failed (%s): %s", result.error_code, result.error)
            return SqlLaneResult(
                generated=generated, error=result.error, error_code=result.error_code
            )
        return SqlLaneResult(
            generated=generated, rows={"columns": result.columns, "rows": result.rows}
        )
