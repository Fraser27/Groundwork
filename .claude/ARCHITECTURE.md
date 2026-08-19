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

- **Governing** (closed, 17 in the legal pack): `REPRESENTS`, `ADVERSE_TO`, `CITES`,
  `OVERRULES`, `AFFILIATE_OF`, `SUBJECT_TO_PRIVILEGE`, `DEADLINE_FOR`, … Anything a
  conflict check, privilege wall, or limitation-period calculation reads. Unapproved
  predicate = rejected write.
- **Descriptive** (open): `CONCERNS_TOPIC`, `IN_INDUSTRY`, `MENTIONS`. Sprawl costs
  retrieval precision, not a malpractice claim.

Grounding is `STANDARD` only — exact and label matching, no LLM. An LLM deciding
that `acts_on_behalf_of` means `REPRESENTS` is a schema decision made by a model.

The test for tier: *would a wrong answer embarrass you, or expose you?*

## The rules, and what each one actually asks

Three rules ship in the legal pack. They are easy to confuse in the UI, because the
trace groups every refusal under one heading — so a stale-citation warning arrives
looking like a conflict. They are not the same kind of finding and they call for
different reactions.

| Rule | Asks | Kind of finding | Remedy | Effect |
|---|---|---|---|---|
| `conflict_check` | is the firm on both sides of one party? | ethics / conduct | decline, or raise a barrier | notify |
| `conflict_via_affiliate` | is the firm against a company its own client part-owns? | ethics / conduct | as above, usually a barrier | notify |
| `authority_stale` | does this document rest on law that has since been overturned? | quality / competence | revise the advice | notify |

### Notify, withhold, and the one true wall

`effect:` is the second axis on a block, and no rule finding in the legal pack withholds.
A rule finding is something the graph *noticed*; a lawyer decides what it means, and they
cannot decide from evidence they were never shown. Withholding a conflicted party's file
is how "who is the counsel for Halveston" came back empty.

An **ethical screen** is the remaining prohibition, and the shape is right: a screen is a
recorded instruction that a named person must not see a matter. It never passed through
`blocks:` at all. Healthcare's `CONTRAINDICATION_ALERT` keeps `withhold` — a clinician
acting on a suppressed allergy is direct harm — so the two packs disagree by a one-word
YAML diff, which is what the abstraction is for.

A notify finding still sorts first by premise count, still names its premises, and
suppresses nothing.

### `authority_stale` is not an ethics finding

```
when:  (d:Document)-[:CITES]->(a:Authority)          a document relies on a case
       (x:Authority)-[:OVERRULES]->(a:Authority)     another case overturned it
then:  (d)-[:RELIES_ON_STALE_AUTHORITY]->(a)
```

Nothing about acting against a client. The advice was very likely correct when
written — which is exactly why it is worth surfacing rather than treating as an error.
The demo fixture states the consequence: *"Any advice relying on The Aquitaine for a
notice-period calculation should be revised."* A deadline calculated from overruled law
is a negligence exposure, which is why the predicate is governing rather than a note.

**Known rough edge.** The finding is document-level, so it is reported against the whole
passage rather than the topic it concerns. Asking "does acting for Halveston create a
conflict?" can therefore raise a notice about an advice document because it cites an
overruled case *on notice periods* — two unrelated subjects. Correct as far as it goes,
and confusing to read. Much less costly since it became `notify`: the reader sees the
document and the finding together, rather than losing the passage to an unrelated cause.

### Which end a block taints, and why it differs per rule

`blocks:` names the endpoint a finding is *about* — which is what the trace reports and,
for a `withhold` finding, what stops being handed over. It is a separate axis from
`governing`: every block is governing, but most governing predicates are ordinary facts.
Orthogonal to `effect:` on purpose, so narrowing a finding to `notify` does not throw away
which end it concerns.

| Predicate | Blocks | Reasoning |
|---|---|---|
| `POTENTIAL_CONFLICT` | `object` — the party | Blocking the *matter* blacks out the file the disputes team is retained to run. Withholding the party is the barrier's substance; withholding the matter withholds the work. |
| `RELIES_ON_STALE_AUTHORITY` | `subject` — the document | Blocking the authority would suppress "*The Marisol* overrules *The Aquitaine*" — the single fact the reader most needs. |
| `CONTRAINDICATION_ALERT` (healthcare) | `object` — the drug | Evidence about the patient is what the clinician is treating them from; suppressing it would withhold the record to protect the record. |

`POTENTIAL_CONFLICT` was `both` until a conflict about a party the firm also
represented made that client's own file unanswerable — "who is the counsel for
Halveston" returned nothing. The per-person half of a barrier is an ethical screen
(`MatterScreen`, keyed by user), which is the control a real risk memo describes:
named individuals, not a matter nobody may read.

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
