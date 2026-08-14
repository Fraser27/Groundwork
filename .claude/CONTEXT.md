# How this project came to exist

Background reading. Useful if you are picking this up cold and wondering why the
design looks the way it does, or why two other repos are referenced constantly.

## The three-way comparison that produced this

**`../rosetta-sdl`** — a semantic data layer over AWS data lakes. Neo4j ontology,
governed metrics compiled to deterministic Athena SQL, sqlglot SQL firewall, MCP
server on AgentCore, React admin UI. ~8.8k lines of Python.

Its bet is **determinism as governance**: a metric is YAML that compiles to SQL with
no LLM in the path, and anything uncompilable is firewalled or killed and audited.
Narrow and opinionated — "revenue by month" returns the same answer every time. Its
graph is *declared*: it scans a Glue catalog and records what it finds. There is
nothing to be uncertain about, so it has no notion of confidence.

**`../semantica`** (not cloned here; inspected during design) — an open-source
semantic/knowledge-graph library. ~179k lines, 349 files. Polyglot backends (Neo4j,
FalkorDB, AGE, Neptune, plus RDF stores), 20+ ingestors, Rete/Datalog/SPARQL
reasoning, W3C PROV-O provenance, decision records with causal chains.

Its bet is **explainability as governance**: build a real knowledge graph, run
deterministic inference over it, keep a provenance trail. Every edge carries
`confidence` + `context` + `extraction_method`. But it has *no tenancy at all*, and
nothing distinguishes a parsed citation from an LLM's guess at query time.

Neither codebase contains the other's core primitives. Semantica has zero SQL
generation and no concept of a governed metric; rosetta has zero reasoning,
provenance, or graph analytics.

**`../coa`** (AWS `context-ontology-accelerator`, cloned at v0.2.0) — published while
this was being designed. Apache-2.0, read-only mirror, no PRs accepted. Effectively
the union of the other two: governed metrics *and* ontology induction with OWL
reasoning (HermiT/ELK), plus a genuinely good Cedar authorization model and the
property that agents have no standalone identity.

Its limitations for this use case: 16 stacks, table-first (ontology induction reads
table schemas), the managed graph build is not configurable, and it uses
DataZone/SMUS for tenancy which is enterprise-catalog governance rather than
firm-level multi-tenancy.

## What LexGraph takes from each

Structured governance from rosetta. Ontology/reasoning/Cedar ideas from COA.
Per-edge provenance from the Semantica critique — but **stricter**, because
Semantica's version is not sufficient for legal:

| | Semantica | LexGraph |
|---|---|---|
| confidence, source span, method | yes | yes |
| `epistemic_class` | no | **yes — the key addition** |
| premises + proof DAG | partial | first-class, queryable |
| bitemporal | type exists, not universal | on every assertion |
| tenancy | none | `tenant_id` mandatory |
| cascading retraction | no | enforced invariant |

## COA was deployed, studied, and torn down

All 16 stacks went into AWS account 139484103396 (us-east-1) on 2026-08-14, then were
destroyed the same day. Deployed to understand it properly before borrowing from it —
reading a repo tells you less than watching it run. **Nothing is running or billing
now.** The local clone at `../coa` stays as a read-only reference.

Three non-obvious things that cost real time during that deploy, recorded so they are
not rediscovered:

1. **Docker disk exhaustion** — image builds die with "no space left on device" from
   stale images even when the host has space. `docker image prune -af`.
2. **The Lambda concurrency floor is 100, not 10.** COA's `preflight-deploy.sh`
   assumes 10, so it *passes* a check it should fail, then rolls back ~30 minutes in.
   Deploy with `--context lambda_reserved_concurrency=0`. Genuine upstream bug.
3. **A stale `cdk.out` silently wins.** Running `cdk synth` without full context
   writes a template that a later `cdk deploy` reuses. Always `rm -rf cdk.out`, or
   pass identical context to both.

## How the current code was produced

The foundation — `src/graph/assertions.py`, `src/graph/scope.py`,
`src/ontology/loader.py`, both ontology packs — was written and reviewed
line-by-line. Start there when forming a mental model; it also sets the commenting
standard for the repo.

The structured pipeline, unstructured pipeline, UI, and CDK were built by parallel
agents working against those contracts, with non-overlapping file ownership. Their
output is tested but has had less human review — `.claude/STATUS.md` lists the
specific decisions worth a second look, including one case where an agent found a
real bug in rosetta-sdl's derived-metric approach.

The API layer (`src/api/`, `src/auth.py`, `src/query/resolver.py`, `src/mcp/`) is
deliberately unwritten. It is the integration point that has to reconcile all four
workstreams, so it wants a human eye rather than a fifth parallel agent.
