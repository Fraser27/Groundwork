# LexGraph

A governed semantic layer over **both** structured and unstructured data, where
every fact carries its own provenance.

Domain-agnostic by construction; ships with a legal ontology as the default and a
healthcare pack to keep that claim honest.

## Why this exists

Two existing systems each solve half the problem:

- **rosetta-sdl** governs *structured* data well — governed metrics compiled to
  deterministic SQL, an AST-level SQL firewall, a kill switch for ungoverned
  queries. But its graph is *declared*: it scans a Glue catalog and records what it
  finds. There is nothing to be uncertain about.
- **AWS context-ontology-accelerator (COA)** handles ontologies, OWL reasoning and
  document ingestion, with a genuinely good Cedar authorization model. But it is
  16 stacks, table-first, and its graph build is not configurable.

Neither answers the question a regulated customer actually asks: **"why does the
system believe this, and can you prove it?"**

LexGraph's answer is the assertion contract.

## The core idea

Every edge in the graph is an **Assertion** carrying:

| Field | Why |
|---|---|
| `epistemic_class` | How we came to believe it — the axis that makes the graph defensible |
| `method` | Versioned and specific: `vision:claude-haiku-4-5@v1`, `llm:opus-5+verify:quote@v1` |
| `confidence` | Trust floor for retrieval |
| `source_locator` | File + page + verbatim quote, or source + table + column |
| `premises[]` | For inferences — unwinds into a proof tree |
| bitemporal fields | World time *and* transaction time, from day one |
| `tenant_id` | Never optional |

`epistemic_class` is the important one:

```
DECLARED         a system of record said so            → auto-assert
EXTRACTED_DET    a check confirmed it                  → auto-assert
EXTRACTED_MODEL  a model interpreted it                → human review
INFERRED         a rule derived it from premises       → carries proof tree
PREDICTED        topological guess                     → never in retrieval
```

`EXTRACTED_DET` does **not** mean "a regex found it" — there is no regex layer. A
model proposes that some text appears on some page, and a string search against the
chunk either confirms it or does not. The proposal is probabilistic; the confirmation
is not, and the confirmation is what the class records.

Only *presence* predicates (`MENTIONS`) may be `EXTRACTED_DET`, enforced in
`build_assertion`. A quote match establishes that text is present and nothing more, so
anything implying significance — `CITES` implies reliance, `ADVERSE_TO` implies a
position — stays `EXTRACTED_MODEL` and is reviewed. That constraint fixes a real bug: a
parser that auto-asserted `CITES` recorded "the court declined to follow Brown" as
reliance on Brown.

Without this axis, a confirmed quote and a model's opinion are both just edges.

### Five enforced invariants

1. `tenant_id`, `method` and a source locator are mandatory.
2. `INFERRED` requires premises — every inference is explainable.
3. An inference is never more confident than its weakest premise. A chain of
   guesses cannot launder itself into a certainty.
4. Governing predicates are validated against a closed vocabulary at write time.
5. Review state is *derived* from epistemic class, so no code path can opt out.

All five are covered by tests in `tests/test_assertions.py`.

## Two-tier predicate vocabulary

The failure this prevents: an extractor records `is_counsel_to`, your conflict
check queries `REPRESENTS`, and it returns zero rows. It looks like a clean
conflict check. It is not.

- **Governing predicates** (closed, ~14 in the legal pack) drive consequence:
  conflicts, privilege, deadlines, citation authority. An unapproved predicate is
  **rejected at write time**.
- **Descriptive predicates** (open) are subject-matter tags. Sprawl costs retrieval
  precision, not a malpractice claim.

Grounding is `STANDARD` only — exact and label matching, no LLM. An LLM deciding
that `acts_on_behalf_of` means `REPRESENTS` is a schema decision made by a model,
which is not acceptable in a system that has to be defensible.

The test for which tier: *would a wrong answer embarrass you, or expose you?*

## Tenancy

**One graph per tenant; matters are subgraphs.** Not a graph per matter — conflict
checking is definitionally cross-matter, and shared entity nodes *are* the conflict
signal.

An earlier draft of this README claimed tenants were isolated *structurally*, one
named graph each. That was wrong, and the correction matters: named graphs are an
RDF/SPARQL concept, and Neptune Database holds a single property graph per cluster, so
openCypher cannot see them. Both tenant and matter isolation are **property filters**,
differing only in who may change them.

- **Tenant** → fixed at authentication from the verified JWT. Never read from a request
  parameter, or a caller could widen their own scope.
- **Matter** → **allowlist-primary**: a user sees only matters they are assigned to, so
  the default posture is closed. An *ethical screen* is a stronger, separate refusal on
  top, and it beats everything — an assignment, and `platform-admin` too. An admin who
  can read through a wall is not screened.

Screens are **loud** inside a firm: the lawyer is told which matter, why, and who to
contact. Silent filtering is how a wall causes the harm it exists to prevent — a
conflict check comes back clean because the matching matters were invisible. Across
tenants, refusals stay silent; that is a confidentiality boundary between firms rather
than a documented screen inside one.

Assignments and screens live in DynamoDB and are read per request, not carried in the
token: Cognito group claims only refresh on re-login, and a screen has to bite now.
Every change is append-only — the record of who screened whom, when and why is the
compliance artifact.

No module outside `src/graph/` builds Cypher. `src/graph/scope.py` owns the scoping
clause and there is no parameter for "skip scoping" — an unscoped query is not
expressible.

## Roles, admins and tenants

**Roles are Cognito groups.** `src/auth.py` reads the `cognito:groups` claim off the
verified JWT, so a group membership *is* the authorization fact — there is no second
store to keep in sync, and nothing goes stale between a revocation and the next
request. Three groups are created by `cdk/lib/auth-stack.ts`, and their names are
load-bearing: they must match `Grants.is_platform_admin` and `Grants.can_review`.

| Group | Grants |
|---|---|
| `platform-admin` | Settings, sources, matter access administration |
| `reviewer` | Approve or reject staged assertions |
| `matter-owner` | Assign users to matters they own |

A screen still beats `platform-admin`. An admin who can read through an ethical wall is
not screened, so the wall is checked after the role, never before it.

### Creating a tenant

There is no "create tenant" API, deliberately. A tenant is not a row to be created — it
is the value of a user's `custom:tenant_id` claim, and everything else (the S3 prefix,
the graph property filter, the vector index name) derives from it. Creating a tenant is
therefore creating its first user.

### Creating a user

`custom:tenant_id` is **immutable and cannot be set by the user**, because a user who
could change their own tenant would defeat the only isolation boundary in the system.
That has a consequence worth stating plainly: **self-service signup through the hosted
UI produces a user with no tenant, who cannot sign in.** They authenticate, every API
call is refused, and the UI bounces them back to the login page. Users must be created
by an admin.

```bash
POOL=$(aws cloudformation describe-stacks --stack-name LexGraphAuth \
  --query 'Stacks[0].Outputs[?OutputKey==`UserPoolId`].OutputValue' --output text)

aws cognito-idp admin-create-user \
  --user-pool-id "$POOL" \
  --username lawyer@firm.example \
  --user-attributes Name=email,Value=lawyer@firm.example \
                    Name=email_verified,Value=true \
                    Name=custom:tenant_id,Value=demo-firm \
  --desired-delivery-mediums EMAIL
```

Omit `--desired-delivery-mediums` and add `--message-action SUPPRESS` to create the user
without emailing them, then set a password directly:

```bash
aws cognito-idp admin-set-user-password \
  --user-pool-id "$POOL" --username lawyer@firm.example \
  --password '<a strong password>' --permanent
```

### Making someone an admin

```bash
aws cognito-idp admin-add-user-to-group \
  --user-pool-id "$POOL" --username lawyer@firm.example \
  --group-name platform-admin
```

Group claims are refreshed at **token issue**, so the user must sign out and back in
before a new role takes effect. This is the one place where Cognito groups are the wrong
tool and DynamoDB is used instead: matter assignments and ethical screens are read per
request, precisely because a screen has to bite immediately rather than at next login.

### Fixing a user created without a tenant

`custom:tenant_id` is immutable, so it cannot be patched. Delete and recreate:

```bash
aws cognito-idp admin-delete-user --user-pool-id "$POOL" --username <username>
# then admin-create-user as above, with custom:tenant_id set
```

Symptom to recognise: the API returns 401 for a tenant the user never chose — the UI
falls back to a default tenant id when the claim is absent, and the token then does not
match it.

## Storage: hybrid reification

```
(:Matter)-[:ADVERSE_TO {assertion_id, epistemic_class, confidence}]->(:Party)
     +
(:Assertion {method, source_locator, valid_from, ...})-[:PREMISE]->(:Assertion)
```

The edge carries only what a filtered traversal needs, so "walk edges I trust" is
one hop. Full provenance and the premise DAG live on the `:Assertion` node, read
when a user asks *why*. Pure reification would triple the hops on every read;
edge-properties alone could not express a proof tree.

## Data flow

**S3 is the only source of truth. Neptune and OpenSearch are derived indexes you
can throw away and rebuild.** A bad extraction run is never a data-loss event.

```
upload → S3 (immutable) → vision model, page by page → chunk (keep offsets)
       → model extraction, quote confirmed by search → auto-assert as MENTIONS
       → model extraction, interpretive              → review queue
       → embed → vector store
       → stage assertions → human review → promote → reasoning (INFERRED + premises)
```

Verbatim text is searchable immediately; *derived assertions* wait for review. The
gate protects the graph, not the search index.

Structured sources never move their data — only metadata enters the graph, and rows
are queried in place at read time.

## Read path

```
1. vector + keyword     → candidate chunks
2. graph traversal      → expand along VERIFIED edges only (scope.edge_scope)
3. governed metric      → deterministic SQL, no LLM
4. synthesize           → answer + citations to page/char + proof paths
```

Step 2 is the differentiator: retrieval that expands along assertions you can
defend, rather than just nearest neighbours.

## Status

Built and tested:

- `src/graph/assertions.py` — the contract, 5 invariants
- `src/graph/scope.py` — tenant/matter scoping, bitemporal reads
- `src/ontology/loader.py` — domain packs, two-tier predicate gate
- `ontologies/legal.yaml`, `ontologies/healthcare.yaml`
- 70 tests passing

Next: graph client + schema, structured pipeline (Glue scan → enrich → approve →
metrics), unstructured pipeline (transcribe → extract → review), tiered resolver,
FastAPI + Cognito, MCP on AgentCore, React UI with light/dark themes.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest tests/ -q
```
