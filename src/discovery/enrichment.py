"""Bedrock enrichment of catalog metadata — descriptions and synonyms.

Glue tells you a column is `mtr_stat_cd varchar(2)`. It does not tell you that is a
matter status, or that people call it "case state". An LLM is good at that guess, and
a guess is exactly what it is: `EXTRACTED_MODEL`, review state PENDING, invisible to
retrieval until a human signs it off.

That is the whole reason this module and `glue_scanner.py` are separate files rather
than one "discovery" pass. They produce the same kind of thing — metadata about the
same tables — but one is a declaration and the other is an opinion, and the assertion
contract makes the graph tell them apart.

Enrichment therefore proposes; it never writes. It reads the catalog scan (or the
graph, through a caller-supplied reader) and returns assertions plus the text nodes
they point at. Descriptions are modelled as edges to a `:Description` node rather
than a property set on `:Table`, because a property has no epistemic class and
nowhere to hang a review state — writing one would put a model's guess straight into
the live graph, which is the failure this design exists to prevent.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from src.discovery.glue_scanner import (
    CatalogNode,
    column_node_id,
    table_node_id,
)
from src.graph.assertions import Assertion, EpistemicClass, SourceLocator, build_assertion
from src.metrics.models import TableSchema

logger = logging.getLogger(__name__)

DESCRIBED_AS = "DESCRIBED_AS"
HAS_SYNONYM = "HAS_SYNONYM"
CONCERNS_TOPIC = "CONCERNS_TOPIC"

#: Sits exactly on the retrieval floor (`scope.DEFAULT_MIN_CONFIDENCE`). Deliberate:
#: an enrichment a human has approved should be usable immediately, so the thing
#: holding it back is the review gate, not a number chosen to be timid.
DEFAULT_MODEL_CONFIDENCE = 0.8

#: A description longer than this is the model padding. Truncated rather than
#: rejected — a verbose guess is still a useful starting point for a reviewer.
_MAX_DESCRIPTION_WORDS = 40
_MAX_SYNONYMS = 8
_MAX_TOPICS = 6

_BEDROCK_RETRIES = 3
_RETRY_BASE_DELAY = 1.0
_RETRYABLE = ("throttl", "too many requests", "serviceunavailable", "timeout")

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def method_for(model_id: str) -> str:
    """Assertion method for an enrichment run.

    The model id *is* the version: re-enriching with a newer model supersedes the old
    assertions instead of silently mixing two models' opinions in one graph.
    """
    return f"llm:{model_id}"


@dataclass
class EnrichmentResult:
    nodes: list[CatalogNode] = field(default_factory=list)
    assertions: list[Assertion] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    """Tables the model failed on. Enrichment is best-effort by nature — a failure
    leaves the declared catalog intact and simply un-enriched."""

    def extend(self, other: EnrichmentResult) -> None:
        self.nodes.extend(other.nodes)
        self.assertions.extend(other.assertions)
        self.errors.extend(other.errors)


@dataclass(frozen=True)
class TableEnrichment:
    """The model's parsed opinion about one table."""

    table_description: str = ""
    column_descriptions: dict[str, str] = field(default_factory=dict)
    synonyms: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)


def enrich_tables(
    bedrock_client,
    tables: Sequence[TableSchema],
    *,
    tenant_id: str,
    source_id: str,
    model_id: str,
    domain: str = "legal",
    existing_descriptions: dict[str, str] | None = None,
) -> EnrichmentResult:
    """Propose descriptions and synonyms for each table. Nothing is written.

    `existing_descriptions` (table or `table.column` -> text) suppresses re-guessing
    where a human or the catalog already said something. A model's description must
    never overwrite one that came from a person.
    """
    have = existing_descriptions or {}
    result = EnrichmentResult()

    for table in tables:
        try:
            enrichment = _ask_model(bedrock_client, model_id, table, domain, have)
        except Exception as e:
            logger.warning("enrichment failed for %s: %s", table.full_name, e)
            result.errors.append(f"{table.full_name}: {e}")
            continue
        if enrichment is None:
            continue
        result.extend(
            build_enrichment_assertions(
                enrichment,
                table,
                tenant_id=tenant_id,
                source_id=source_id,
                model_id=model_id,
            )
        )
    return result


def build_enrichment_assertions(
    enrichment: TableEnrichment,
    table: TableSchema,
    *,
    tenant_id: str,
    source_id: str,
    model_id: str,
    confidence: float = DEFAULT_MODEL_CONFIDENCE,
) -> EnrichmentResult:
    """Turn a parsed model opinion into PENDING assertions plus their text nodes."""
    method = method_for(model_id)
    result = EnrichmentResult()
    subject = table_node_id(source_id, table.full_name)
    table_locator = SourceLocator(source_id=source_id, table=table.full_name)

    def add(subject_id: str, predicate: str, node: CatalogNode, locator: SourceLocator) -> None:
        result.nodes.append(node)
        result.assertions.append(
            build_assertion(
                tenant_id=tenant_id,
                subject_id=subject_id,
                predicate=predicate,
                object_id=node.node_id,
                epistemic_class=EpistemicClass.EXTRACTED_MODEL,
                method=method,
                confidence=confidence,
                source_locator=locator,
            )
        )

    if description := _truncate(enrichment.table_description):
        add(subject, DESCRIBED_AS, _description_node(tenant_id, description), table_locator)

    for column, text in enrichment.column_descriptions.items():
        if column not in table.columns:
            # The model invented a column. Dropped silently — it is a hallucination,
            # not a finding, and the catalog is the authority on what columns exist.
            continue
        text = _truncate(text)
        if not text:
            continue
        add(
            column_node_id(source_id, table.full_name, column),
            DESCRIBED_AS,
            _description_node(tenant_id, text),
            SourceLocator(source_id=source_id, table=table.full_name, column=column),
        )

    for term in _dedupe(enrichment.synonyms, _MAX_SYNONYMS):
        add(subject, HAS_SYNONYM, _term_node(tenant_id, term, "Synonym"), table_locator)

    for topic in _dedupe(enrichment.topics, _MAX_TOPICS):
        add(subject, CONCERNS_TOPIC, _term_node(tenant_id, topic, "Topic"), table_locator)

    return result


# ── Bedrock ───────────────────────────────────────────────────────────────────


def _ask_model(
    bedrock_client,
    model_id: str,
    table: TableSchema,
    domain: str,
    have: dict[str, str],
) -> TableEnrichment | None:
    """Prompt for whatever is still missing. Returns None when nothing is."""
    needs_table = not have.get(table.full_name, "").strip()
    needs_columns = [c for c in table.columns if not have.get(f"{table.full_name}.{c}", "").strip()]
    if not needs_table and not needs_columns:
        return None

    prompt = _build_prompt(table, domain, needs_table, needs_columns)
    text = _converse(bedrock_client, model_id, prompt)
    return _parse_response(text)


def _build_prompt(
    table: TableSchema, domain: str, needs_table: bool, needs_columns: list[str]
) -> str:
    columns = ", ".join(f"{name} ({dtype})" for name, dtype in table.columns.items())
    asks: list[str] = []
    if needs_table:
        asks.append("a one-sentence business description of the table")
    if needs_columns:
        asks.append(f"a one-sentence description for each of these columns: {', '.join(needs_columns)}")
    asks.append("business synonyms a user might say instead of the table name")
    asks.append("subject-matter topics the table concerns")

    return (
        f"You are annotating a data catalog for a {domain} organisation.\n"
        f"Table: {table.full_name}\n"
        f"Columns: {columns}\n\n"
        f"Provide {'; '.join(asks)}.\n"
        "Describe only what the names and types support. If a column's meaning is not "
        "clear from its name and type, omit it rather than guessing.\n\n"
        'Reply with JSON only: {"table_description": "...", '
        '"columns": {"column_name": "description"}, '
        '"synonyms": ["..."], "topics": ["..."]}'
    )


def _converse(bedrock_client, model_id: str, prompt: str, max_tokens: int = 1024) -> str:
    """Bedrock Converse — provider-agnostic, so the model stays configurable."""
    last: Exception | None = None
    for attempt in range(_BEDROCK_RETRIES):
        try:
            response = bedrock_client.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": 0},
            )
            return response["output"]["message"]["content"][0]["text"]
        except Exception as e:
            last = e
            if not any(marker in str(e).lower() for marker in _RETRYABLE):
                raise
            if attempt < _BEDROCK_RETRIES - 1:
                time.sleep(_RETRY_BASE_DELAY * (2**attempt))
    raise last if last else RuntimeError("bedrock call failed")


def _parse_response(text: str) -> TableEnrichment | None:
    """Extract the JSON object, tolerating code fences and surrounding prose."""
    payload = _extract_json(text)
    if payload is None:
        logger.warning("enrichment response was not usable JSON: %.200s", text)
        return None
    columns = payload.get("columns")
    return TableEnrichment(
        table_description=str(payload.get("table_description") or ""),
        column_descriptions={
            str(k): str(v) for k, v in (columns or {}).items() if isinstance(columns, dict) and v
        },
        synonyms=[str(s) for s in payload.get("synonyms") or [] if s],
        topics=[str(t) for t in payload.get("topics") or [] if t],
    )


def _extract_json(text: str) -> dict | None:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


# ── Nodes ─────────────────────────────────────────────────────────────────────


def _description_node(tenant_id: str, text: str) -> CatalogNode:
    """Content-addressed, so two tables described identically share one node.

    That is not just deduplication: approving the text once approves it everywhere it
    was proposed, which is how a reviewer stays ahead of a catalog-wide enrichment run.
    """
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    return CatalogNode(
        node_id=f"description:{digest}",
        labels=("Description",),
        props={"tenant_id": tenant_id, "text": text},
    )


def _term_node(tenant_id: str, term: str, label: str) -> CatalogNode:
    slug = _slugify(term)
    return CatalogNode(
        node_id=f"{label.lower()}:{slug}",
        labels=(label,),
        props={"tenant_id": tenant_id, "name": term.strip().lower(), "slug": slug},
    )


def _slugify(value: str) -> str:
    return _SLUG_RE.sub("-", value.strip().lower()).strip("-")


def _truncate(text: str, max_words: int = _MAX_DESCRIPTION_WORDS) -> str:
    words = (text or "").strip().split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + "…"


def _dedupe(values: Iterable[str], limit: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        slug = _slugify(value)
        if not slug or slug in seen:
            continue
        seen.add(slug)
        out.append(value.strip())
        if len(out) >= limit:
            break
    return out
