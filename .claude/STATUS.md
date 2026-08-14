# Status — 2026-08-14

Snapshot for picking this up cold. Test count and file lists were accurate at the
time of writing; re-run the suite before trusting them.

```bash
cd /Users/fraseque/Fraser/Playground/lexgraph
.venv/bin/python -m pytest tests/ -q          # 481 passing
cd cdk && npx cdk synth                        # 6 templates, 0 warnings
cd ui && npm run typecheck && npm run build    # clean
```

At handoff: **481 tests passing**, ~6,100 lines of Python, 12 UI pages, 6 CDK stacks.
All four parallel agents have finished.

**Nothing is committed.** `git init` has run but there are no commits, so a first
commit is step one.

Roughly 75% complete. What remains is the integration layer — see "Not done" below.

## Done

### Foundation (hand-written, reviewed)
| File | Lines | What |
|---|---|---|
| `src/graph/assertions.py` | 313 | The assertion contract + 5 invariants |
| `src/graph/scope.py` | 220 | Tenant/matter scoping, bitemporal `as_of` reads |
| `src/graph/client.py` | 68 | Neo4j/Neptune driver; `read_scoped()` rejects unscoped reads |
| `src/ontology/loader.py` | 166 | Domain packs, two-tier predicate gate |
| `ontologies/legal.yaml` | — | 8 entities, 14 governing + 4 descriptive predicates, 2 rules |
| `ontologies/healthcare.yaml` | — | Second pack, keeps domain-agnosticism honest |

### Structured pipeline (agent-built, Rosetta lineage)
`src/discovery/glue_scanner.py`, `src/discovery/enrichment.py`,
`src/metrics/{models,compiler,loader}.py`, `src/query/firewall.py`,
`src/executors/athena.py`, `sample/metrics.yaml` (6 legal metrics).

### Unstructured pipeline (agent-built, COA lineage)
`src/documents/{models,ingest,parse,chunk,embed,review,retract}.py`,
`src/documents/extractors/{deterministic,model}.py`.

### UI (agent-built, mirrors rosetta-sdl)
`ui/` — Vite + React + TS. `index.css` with light/dark CSS variables,
`components/{FieldHelp,EpistemicBadge,ConfidenceBar,ProvenancePanel,Shared}.tsx`,
pages: Dashboard, Documents, Login, Matters, Metrics, QueryBuilder, ReviewQueue,
Tables, TableDetail. Mock data isolated in `ui/src/mocks.ts` so it is trivial to
delete.

### Governance (hand-written)
`src/governance.py` + 24 tests. Runtime-tunable trust thresholds, kill switches, all
four model ids, ontology domain, retrieval knobs. Plain-language `FIELD_HELP` for
every control, with a test asserting no jargon leaks into it.

### CDK (agent-built)
`cdk/lib/{network,data,auth,app,mcp,web}-stack.ts` + `bin/app.ts`, `lib/config.ts`,
`cdk/README.md`. `npx cdk synth` produces all six templates with zero warnings.
`docker build --platform linux/arm64` verified on the app image.

### Scaffolding (agent-built)
`docker-compose.yml` (local Neo4j standing in for Neptune — both speak openCypher over
Bolt), `Dockerfile`, `Dockerfile.ui`, `Makefile`, `.env.example`, `.gitignore`,
`.dockerignore`, `src/config.py`, `src/constants.py`.

## Not done

**Integration layer — DONE and verified running locally (2026-08-14):**
- `src/graph/schema.py` — constraints + indexes. Neo4j-only; skipped against Neptune,
  which has no DDL. Hot-path index leads on `tenant_id`.
- `src/auth.py` — Cognito JWT verification, `Grants`, `AuthContext` construction.
  Dev bypass gated twice (config validate + request time).
- `src/api/deps.py` — `Services` container, principal/tenant dependencies.
- `src/api/app.py` — app factory, exception mapping. `ScopeViolation` -> **404, never
  403**: a 403 confirms the thing exists.
- `src/api/routes_review.py` — review queue, approve/reject, provenance.
- `src/api/routes_governance.py` — settings GET/PATCH with warnings.
- `src/api/routes_catalog.py` — dashboard, matters, ontology, neighbourhood.
- `src/api/routes_documents.py` — multipart upload (store -> transcribe -> chunk ->
  embed -> extract -> stage -> promote) plus the text endpoint for tests and demos.
  Transcription is skipped, not fatal, when no Bedrock client is available.
- `src/api/routes_query.py` + `src/query/resolver.py` — 4-tier resolution.

**Resolver tiers — WIRED (2026-08-14). All four answer.**
- `src/query/graph_reader.py` — tier 2/3. Lexical search over assertions, ranked by
  matched-term count then confidence. Returns `matched_on` so the UI can explain *why*
  a fact was included, and populates `assertions_used` — which is what makes an answer
  auditable rather than merely labelled.
- `src/query/metric_matcher.py` — tier 1. Weighted lexical matching (name/synonym hits
  count double definition prose), declines on ties and on a single matched term.
- `src/query/vector_search.py` — adapter between `Embedder` and the resolver. Absorbs
  Bedrock failure so tier 3 declines instead of 500ing.
- Ingest now chunks and embeds (`routes_documents.py`), and `docker-compose.yml` has an
  OpenSearch service under the `vectors` profile.

**Still missing:**
- `src/mcp/server.py` — MCP on AgentCore. Model on `../rosetta-sdl/src/mcp/server.py`
  and `../rosetta-sdl/agentcore/deploy_agent.py`. `docker-compose.yml` already has an
  `mcp` profile pointing at `src.mcp.server:app`.
- `src/main.py` — not actually needed; compose uses `src.api.app:app`. Add only if a
  Lambda/Mangum entry point is wanted.
- Graph **writes**. Assertions live in `InMemoryAssertionStore`, so they do not survive
  a restart, and `GraphReader` reads from that queue rather than from Cypher. The
  schema and `read_scoped` exist; the writer does not. **Still the single biggest gap** —
  and note `GraphReader`'s interface was kept narrow (`search`/`expand`) precisely so
  swapping the backing store does not touch the resolver.
- Tier 4 has no SQL generator or firewall wired (`sql_generator=None`,
  `firewall=None` in `Services.build_resolver`), so it always declines. The firewall
  exists in `src/query/firewall.py` and is tested; only the Bedrock generator is absent.
- No Athena executor, so tier 1 returns SQL with `answer=None`. `execute=false` is
  therefore the honest mode locally.
- **Vector search uses `InMemoryVectorStore`, not the OpenSearch container.** The
  compose service exists and `VECTOR_ENDPOINT` is read, but no OpenSearch-backed
  `VectorStore` implementation has been written — so embeddings are held in process and
  lost on restart. Embedding calls still need real Bedrock credentials.

**Also still missing:**
- `src/graph/schema.py` — Neptune/Neo4j constraints + indexes. Assertions need an
  index on `assertion_id`, and edges one on `(tenant_id, epistemic_class)` for the
  scoped-traversal hot path. Model on `../rosetta-sdl/src/graph/schema.py`.
- Admin UI wiring for `src/governance.py` — the settings module and its help text
  exist; the `Admin.tsx` page exists; they are not connected because there is no API.

**Never started:**
- Reasoning engine. Rule definitions exist in `ontologies/*.yaml` and the `INFERRED`
  plumbing is in place, but nothing fires them. See open question 2 in `DECISIONS.md`.
- Cedar/AVP policy definitions. `AuthContext` has the shape
  (`matter_allowlist`/`matter_denylist`); no policies are authored.
- Any AWS deployment. CDK synth only — nothing has been deployed.

## Verified running locally — 2026-08-14

```bash
docker-compose up -d neo4j          # NOTE: hyphenated. `docker compose` is not installed
source deploy.env 2>/dev/null; ENVIRONMENT=local AUTH_DEV_BYPASS_TENANT=dev-tenant \
  GRAPH_URI=bolt://localhost:7687 GRAPH_USER=neo4j GRAPH_PASSWORD=lexgraph-dev \
  .venv/bin/python -m uvicorn src.api.app:app --port 8010
```

`/health` returns `status: ok`, `graph: connected`. Schema applied cleanly: 6
constraints, 8 indexes, 4 full-text.

Confirmed working against a real filing (11 extractions — 1 court, 6 party, 1 docket,
2 authority, 1 date):

- **Provenance is byte-exact.** `text[308:318]` == `'03/15/2023'`, verified against the
  original document.
- **Idempotency.** Re-ingesting identical bytes left the count at 11 — content-addressed
  ids collapse duplicates.
- **Tenant isolation.** `GET /api/tenants/other-firm/...` -> 404. Staging another
  tenant's assertion refused at the contract boundary.
- **Ethical wall.** A screened user saw 3 of 6 assertions; a direct probe by assertion id
  was refused. (Walls now *name* the matter and a contact on refusal — see the
  ethical-walls entry in `DECISIONS.md`. Cross-tenant refusals stay silent.)
- **Governance invariant over HTTP.** Raising the floor succeeded; lowering it into the
  model cap returned 422 with the explanation.
- **Kill switch.** Ungoverned query -> 403 with a lawyer-readable refusal.

### Tier resolution, verified live

```
"what is our realization rate"           tier 1 GOVERNED_METRIC  real derived SQL
"show me fees billed by month"           tier 1 GOVERNED_METRIC  grain=month applied
"which matters involve Acme Corporation" tier 2 GRAPH_TRAVERSAL  1 assertion, cited
"zzzq gibberish nonsense"                tier 3 HYBRID           1 citation, 7 related
```

Ingest reported `chunks: 1, chunks_embedded: 1, vector_search: enabled` with no error.

### Four bugs found by wiring the tiers up

1. **`MetricRegistry`, not a dict.** The compiler calls `.resolve()`; a plain dict blew
   up at `compiler.py:638`.
2. **`CompilationResult.is_valid`, not `.ok`.** Wrong attribute name in the matcher.
3. **Definition prose caused a tie that silenced tier 1.** `realization_rate` is
   *documented* as "fees billed over standard value", so on "fees billed by month" it
   scored equal to `fees_billed` and the ambiguity guard declined **both**. Fixed by
   weighting name/synonym hits double.
4. **Single-noun false positive.** "which matters involve Acme Corporation" selected
   `open_matter_count` on the one word "matters". Fixed with `MIN_DISTINCT_TERMS = 2`.
5. **Tier 3 500'd.** I passed `Embedder` where the resolver expected dicts with
   `document_id`. Fixed with the `VectorSearch` adapter, which also swallows Bedrock
   failure so the tier declines rather than erroring.

### Two bugs the local run found that unit tests missed

Both from `\s` matching newlines in the caption regex, so a Title-Case run walked
across line breaks:

1. Plaintiff absorbed the entire court header above it — `party:the-united-states-district-court-...-acme-corporation`.
2. Defendant absorbed the docket line below it — `'Beta Holdings Ltd Case No.'`.

Either would fork the party entity so it never matches the same party elsewhere, which
makes a conflict check under-report *silently*. Fixed with `[ \t]` plus a
`_CAPTION_TRAILER` trim; regression tests in `tests/test_caption_boundaries.py` cover
realistic multi-line filings and assert entity-id stability across layouts.

This is exactly why the local test mattered: 510 unit tests were green with both bugs
present.

## Suggested next steps

1. **Commit.** Nothing is under version control yet.
2. **Graph writer.** The biggest gap: assertions live only in memory, so nothing
   survives a restart. Needs to persist `(:Assertion)` nodes plus the denormalised
   edge (`assertion_id`, `epistemic_class`, `confidence`, `review_state`,
   `superseded_at`) and the `:PREMISE` DAG. `read_scoped` and the indexes are ready.
3. **Reconcile UI ↔ API.** Point `ui/` at `http://localhost:8010`, work through the
   pages, and delete `ui/src/mocks.ts` last — anything still importing it is not wired.
   `GET /matters` now returns a named `withheld` list (`matter_id`, `reason`, `contact`)
   beside `matters`, so `Matters.tsx` should render its banner from that rather than from a
   client-side `matter.walled` flag (see `DECISIONS.md`).
4. `src/mcp/server.py` — the compose `mcp` profile already expects it.
5. Wire the resolver's collaborators: metric matcher, graph reader, vector search.
   The tier scaffolding and kill switch work; the tiers themselves return None.
6. Reasoning engine, so `INFERRED` assertions and proof trees have a producer.

## Review these agent decisions

From the structured-pipeline agent — flagged as worth confirming:

1. **Derived metrics project time as `period`.** The agent found that Rosetta's
   approach of joining CTEs on shared dimension *names* breaks for
   `realization_rate`, whose bases measure time on `invoices.invoice_date` vs
   `time_entries.work_date` — nothing to align on, so it would have compared one
   month's billings to another month's recorded time. Each base now buckets to a
   shared `period` alias. **This was a real bug in the reference implementation.**
   Confirm `period` is the alias you want.
2. **Compiler reads a `SchemaCatalog` protocol, not `GraphClient`** — because
   Cypher outside `src/graph/` is banned. Side benefit: compiler tests need no
   database.
3. **Only one level of derived composition**, on the grounds that a reviewer must be
   able to follow the emitted SQL. Rosetta allowed deeper nesting.
4. **Enrichment proposes, never writes.** Rosetta does `SET t.description = ...`
   directly; a raw property has no epistemic class and nowhere to hang review state.
   Descriptions are now edges to content-addressed `:Description` nodes.
5. **Two governance rules are hard refusals, not warnings:** semi-additive SUM across
   a time grain, and time-slicing on a non-governed axis.
6. **Firewall rejects non-SELECT and stacked statements.** Rosetta only checked table
   names, so `DELETE FROM allowed_table` would have passed.

Remaining lint findings in agent files are `BLE001` blind-except, deliberate for
partial-failure resilience (one inaccessible database must not abort a whole scan).

### From the unstructured-pipeline agent

1. **Captions emit `MENTIONS`, not `ADVERSE_TO`.** An "X v. Y" caption proves a name
   appears; inferring adversity is *interpretation*, which disqualifies it from
   `EXTRACTED_DET`. Correct call — but it means adverse-party edges do not come free
   from captions. If you want them, that belongs in the model extractor or a rule.
2. **Model confidence is hard-capped at 0.79**, just under `DEFAULT_MIN_CONFIDENCE`
   (0.8). An unreviewed model edge therefore sits below the retrieval floor *by
   construction* and cannot shape an answer even if the review gate were bypassed.
   Elegant defence-in-depth. Note the coupling: raising the floor without revisiting
   the cap silently changes behaviour.
3. **Model evidence must be verbatim** (whitespace-normalised only). Paraphrase is
   dropped rather than stored with approximate offsets — offsets into text that never
   existed are worse than no offsets.
4. **Reporters are an allowlist, not a generic pattern.** Recall on obscure state
   reporters is limited *by design*: a fabricated `Authority` is a fact the system
   would then defend, and `authority_stale` could fire on it. Revisit if recall
   complaints arrive; extend the allowlist rather than loosening the pattern.
5. **Stores are in-memory reference implementations behind Protocols.** DynamoDB /
   Neptune / OpenSearch are the real backends and are not yet wired.
   `InMemoryAssertionStore` maintains the reverse-premise index the cascade walks —
   whatever replaces it must too, or cascading retraction breaks silently.

Three bugs its tests caught, worth knowing because they are the class of thing that
would have shipped:

- **Court names absorbed the following line.** `\s+` crossed newlines, so
  `...DISTRICT OF NEW YORK\nCase No. ...` produced a court named "New York Case",
  forking the Court entity on a line-wrapping detail.
- **Retracted assertions stayed in the review queue and could be approved** — a
  cascade can retract a still-pending claim, and `approve` would then revive a
  withdrawn fact. Both paths now filter on `is_current`.
- **Pathological chunk overlap was silently absorbed.** Chunk count scales as
  `length/(target-overlap)`, so a 95% overlap typo is a 20x embedding bill. Capped at
  50%.

### Resolved during handoff

`assertions.py` referred to `writer.retract`, a module that was never written. The
cascade actually lives in `src/documents/retract.py` operating on the store
abstraction. Docstring corrected. If `src/graph/writer.py` is ever added, either
re-home the cascade or have the writer delegate — do not implement it twice.

## Environment

- `.venv` exists (Python 3.14 in the venv; code targets 3.11+ — the agent hit and
  fixed one 3.14-only f-string that would have broken on 3.11, so verify with
  `ruff --target-version py311`)
- Node 22+, pnpm at `$HOME/.local/share/mise/shims`
- Reference projects: `../rosetta-sdl` (structured + UI), `../coa` (AWS COA, cloned
  at v0.2.0)
- COA was deployed to AWS account 139484103396 to study it, then **torn down on
  2026-08-14**. Nothing of ours is running in AWS and nothing is billing. The local
  clone at `../coa` remains as a read-only reference, and `../coa/DEPLOY-NOTES.md`
  records the three deploy blockers worth knowing if it is ever redeployed (see
  `CONTEXT.md`).
