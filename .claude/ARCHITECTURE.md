# Architecture

## The assertion contract

Every edge in the graph is an **Assertion**. `src/graph/assertions.py` is the only
sanctioned way to create one.

```
tenant_id         never optional
subject/predicate/object
epistemic_class   DECLARED | EXTRACTED_DET | EXTRACTED_MODEL | INFERRED | PREDICTED
method            versioned + specific: regex:bluebook_citation@v3, llm:claude-sonnet-5
confidence        float 0..1
source_locator    unstructured: document_id + page + char_start/end + span_sha256
                  structured:   source_id + table + column + query_sha256
premises[]        INFERRED only, non-empty -> unwinds into a proof tree
rule_id/version   INFERRED only
valid_from/until  world time  — when the fact was true
recorded_at       transaction time — when we learned it
superseded_at     null = current. retraction is append-only, never DELETE
review_state      AUTO_ASSERTED | PENDING | APPROVED | REJECTED
```

### Epistemic classes

| Class | Meaning | Review |
|---|---|---|
| `DECLARED` | A system of record said so (Glue catalog, CMS export) | auto-assert |
| `EXTRACTED_DET` | Deterministic parser, reproducible | auto-assert |
| `EXTRACTED_MODEL` | An LLM said so | human review |
| `INFERRED` | A rule derived it from premises | carries proof tree |
| `PREDICTED` | Topological guess | never in retrieval |

This is the axis that makes the graph defensible. Without it, a parsed citation and
a model's opinion are indistinguishable at query time.

### The five invariants

1. `tenant_id`, `method`, and a source locator are mandatory.
2. `INFERRED` requires non-empty `premises`.
3. `confidence(INFERRED) <= min(confidence(premises))` — a chain of guesses cannot
   launder itself into a certainty.
4. Governing predicates validated against the closed vocabulary at write time.
5. Review state is **derived** from epistemic class, never accepted from the caller,
   so no code path can opt itself out of review.

Cascading retraction is the sixth rule but lives in `src/documents/retract.py`
because it must walk the graph rather than inspect one candidate: retracting an
assertion transitively supersedes every `INFERRED` assertion whose premises include
it.

## Storage: hybrid reification

```
(:Matter)-[:ADVERSE_TO {assertion_id, epistemic_class, confidence, ...}]->(:Party)
     +
(:Assertion {method, source_locator, valid_from, ...})-[:PREMISE]->(:Assertion)
```

The edge carries only the filterable subset, so "traverse edges I trust" is one hop.
Full provenance and the premise DAG live on the `:Assertion` node and are read only
when a user asks *why*.

Deliberately denormalised. Pure reification would triple the hops on every read;
edge-properties alone could not express a proof tree. Cost is two writes per fact
that must stay consistent.

## Tenancy

**One graph, tenants as property filters. Matters as subgraphs within a tenant.**

Not a graph per matter: conflict checking is definitionally cross-matter, and shared
entity nodes *are* the conflict signal. `Acme Corp` appearing in 40 matters is one
node with 40 edges.

Not a graph per tenant either — that was an early design error worth recording.
Named graphs are an RDF/SPARQL concept; **Neptune Database holds one property graph
per cluster** and openCypher cannot see named graphs. Neptune *Analytics* does have
discrete graph resources, but each carries a 32 m-NCU floor with no autoscaling,
which is a per-customer deployment shape rather than multi-tenant SaaS.

So both tenant and matter isolation are property filters, differing only in who can
change them:

- **Tenant** — fixed at authentication from the verified JWT. Never from a request
  parameter, or a caller could widen their own scope.
- **Matter** — driven by Cedar grants. Denylist always beats allowlist, mirroring
  Cedar's `forbid`-overrides-`permit`. Ethical walls change weekly as staffing
  changes, so they must be policy, not a data migration.

`AuthContext.cluster_key()` is a deliberate seam: if a customer ever contracts for
physically separate storage it becomes the cluster selector, and no call site
changes.

## Predicate vocabulary: two tiers

- **Governing** (closed, 14 in the legal pack): `REPRESENTS`, `ADVERSE_TO`, `CITES`,
  `OVERRULES`, `SUBJECT_TO_PRIVILEGE`, `DEADLINE_FOR`, … Anything a conflict check,
  privilege wall, or limitation-period calculation reads. Unapproved predicate =
  rejected write.
- **Descriptive** (open): `CONCERNS_TOPIC`, `IN_INDUSTRY`, `MENTIONS`. Sprawl costs
  retrieval precision, not a malpractice claim.

Grounding is `STANDARD` only — exact and label matching, no LLM. An LLM deciding
that `acts_on_behalf_of` means `REPRESENTS` is a schema decision made by a model.

The test for tier: *would a wrong answer embarrass you, or expose you?*

## Data flow

**S3 is the only source of truth. Graph and vector store are derived indexes that
can be dropped and rebuilt.** A bad extraction run is never a data-loss event.

```
upload -> S3 (immutable, versioned, KMS) + DynamoDB job row
      -> vision model       (one call per page: text, tables, charts, handwriting)
      -> chunk              (page numbers exact by construction; offsets PRESERVED)
      -> model extraction, quote confirmed by search -> EXTRACTED_DET (MENTIONS only)
      -> model extraction, interpretive              -> EXTRACTED_MODEL -> review queue
      -> embed -> vector store
      -> stage assertions -> human review -> promote -> reasoning (INFERRED)
```

Vector writes happen **before** the review gate, deliberately: verbatim text is a
quote, not a claim, and lawyers expect to search a document the moment it lands. The
gate protects the *graph*, not the search index.

Structured sources never move their data. Only metadata (`Table`, `Column`,
`Metric`) enters the graph; rows are queried in place at read time.

## Read path — four tiers

```
1. governed metric   deterministic SQL from YAML, no LLM        (rosetta lineage)
2. graph traversal   openCypher over VERIFIED edges only
3. hybrid            vector retrieve -> expand along edges -> join structured
4. LLM SQL           ad-hoc, firewalled, killable via kill switch
```

Tier 2 is the differentiator: retrieval that expands along assertions you can
defend, not just nearest neighbours.

## Infrastructure — six stacks

Split by deploy cadence and blast radius, not by feature.

| Stack | Contents |
|---|---|
| `network` | VPC, subnets, SGs, VPC endpoints, NAT |
| `data` | Neptune, OpenSearch Serverless, DynamoDB, S3 |
| `auth` | Cognito + Verified Permissions (Cedar) policy store |
| `app` | ECS Fargate (FastAPI: control plane + serve + workers) + ALB |
| `mcp` | Bedrock AgentCore Runtime |
| `web` | CloudFront + S3 (React UI) |

`data` is separate because Neptune is slow to create and holds the only
irreplaceable state. `app` is separate because it is redeployed constantly.

**Neptune Database `db.t4g.medium`** — Graviton, free-tier eligible for 750
instance-hours, medium-only, explicitly not production-grade. Instance class is a CDK
context parameter so it can be raised without a code edit.

Local dev uses **Neo4j in Docker**: both speak openCypher over Bolt, so the same
client code works against local Neo4j and Neptune in AWS.

**AZ constraint:** AgentCore Runtime supports only a subset of AZs in us-east-1, and
the OpenSearch Serverless VPC endpoint a different subset. The VPC is pinned to the
intersection. Do not "simplify" the network stack — it will silently break deploys.
