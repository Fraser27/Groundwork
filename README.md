# Groundwork

A governed semantic layer over **both** structured and unstructured data, where
every fact carries its own provenance.

Domain-agnostic by construction. Three ontology packs ship: `fintech` (the default),
`legal` and `healthcare`. The extra two exist to keep the domain-agnostic claim
honest rather than as demos.

> The product is **Groundwork**. The CDK stacks, Python package, and every identifier this
> repo controls are named `groundwork`. The one thing left as `lexgraph` is the local
> checkout's directory name — renaming that is a filesystem operation outside the repo's
> own control, so it is left to whoever clones it.

## Why this exists

Two kinds of system each solve half the problem.

A **semantic layer over a warehouse** governs structured data well: metrics compiled
to deterministic SQL, an AST-level SQL firewall, a kill switch for ungoverned queries.
But its graph is *declared*: it scans a catalog and records what it finds. There is
nothing to be uncertain about.

A **document knowledge graph** handles ontologies, reasoning and ingestion, and can
tell you that two parties are related. But the claim arrives without a way to check
it: no page, no quote, no record of whether a person ever agreed with the model.

Neither answers the question a regulated customer actually asks: **"why does the
system believe this, and can you prove it?"**

Groundwork's answer is the assertion contract.

## The core idea

Every edge in the graph is an **Assertion** carrying:

| Field | Why |
|---|---|
| `epistemic_class` | How we came to believe it — the axis that makes the graph defensible |
| `method` | Versioned and specific: `llm:nova-2-lite`, `cms:export@v1`, `admin:you@firm.example` |
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

Admins of the **home tenant** (`demo-firm` by default, set by `homeTenant` in
`cdk.json`) create and delete other tenants from the **Platform** page. Not any
platform-admin: that is a role within a firm, so letting it reach across tenants would
let one customer delete another's data.

Creating a tenant **requires inviting its admin user by email**, because a tenant with no
user who can sign in is a namespace nobody can reach. Deletion cascades in a fixed order
across identity, settings, graph, vectors, jobs, audit and finally S3, then leaves a
tombstone: the row survives with `deleted_at` set, so an id cannot be silently reused.
Reusing one is how one firm's people end up looking at a graph built by another.

The first tenant of all is different, because no admin exists yet to create it. That one
is created by making its first Cognito user directly, as in the installation steps above.

### Creating a user

`custom:tenant_id` is **immutable and cannot be set by the user**, because a user who
could change their own tenant would defeat the only isolation boundary in the system.
That has a consequence worth stating plainly: **self-service signup through the hosted
UI produces a user with no tenant, who cannot sign in.** They authenticate, every API
call is refused, and the UI bounces them back to the login page. Users must be created
by an admin.

```bash
POOL=$(aws cloudformation describe-stacks --stack-name GroundworkAuth \
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

## Installing into a brand-new AWS account

Start to finish this is about **40 minutes**, most of it Neptune creating itself. The
order matters: three of these steps fail in ways that name the wrong cause if you skip
them, and each is called out below.

### 1. What you need locally

```bash
node --version     # 20 or newer
python3 --version  # 3.11 or newer
docker ps          # must be running: the app image and the UI bundle build in containers
aws sts get-caller-identity   # credentials with admin-ish rights for the first deploy
```

### 2. Enable Bedrock model access

**Do this first.** It is a console action with no CLI equivalent, it can take a few
minutes to take effect, and skipping it produces an `AccessDeniedException` at the first
document upload rather than at deploy, which reads as a broken app.

In the Bedrock console, under **Model access**, enable:

| Model | Used for |
|---|---|
| `amazon.nova-2-lite-v1:0` | Every text and vision role, by default |
| `amazon.titan-embed-text-v2:0` | Embeddings, and there is no alternative |

That is the whole list. Every default is Nova 2 Lite precisely so a workshop account
needs no Anthropic access; if you want stronger extraction, enable Claude Sonnet as well
and change it per tenant in **Admin**.

### 3. Deploy

```bash
git clone https://github.com/Fraser27/Groundwork.git && cd Groundwork
make setup                    # venv, dependencies, CDK node modules
./scripts/deploy.sh           # or: ./scripts/deploy.sh eu-west-1
```

Non-interactive, and takes 25-30 minutes, most of it Neptune creating itself. It does
everything that was previously three manual steps:

- **Resolves your availability zones.** AZ *names* are shuffled per account, so
  `us-east-1a` is a different physical zone in your account than in anyone else's.
  AgentCore Runtime supports only a subset, and a subnet in the wrong one fails
  `GroundworkMcp` with an error that names the *subnet* rather than the zone. The script
  maps the supported zone IDs to your account's names and writes them to `cdk.json`.
- **Bootstraps and deploys** all six stacks.
- **Closes the circular callback requirement.** The Cognito hosted UI needs the
  CloudFront domain as a callback URL, and CloudFront does not exist until the first
  deploy. The script reads the URL from the stack output, sets `webOrigin`, and
  redeploys the two stacks that consume it: `GroundworkAuth` for the callback and
  `GroundworkData` for the S3 CORS rule that lets a browser upload straight to the bucket.
  Deploying only Auth leaves uploads failing CORS, which looks like a broken button.

It also refuses to start if Bedrock access is missing, by invoking both models for real
rather than asking whether they exist in the region. Those are different questions, and
only the first one predicts whether a document upload will work.

Everything it changes is `cdk.json` context, so re-running is safe and `cdk diff` shows
what a change would actually do.

### 4. Create the first user

No user exists yet, and the tenant a user belongs to is fixed at creation.

```bash
POOL=$(aws cognito-idp list-user-pools --max-results 10 \
  --query "UserPools[?starts_with(Name,'Groundwork')].Id" --output text)

aws cognito-idp admin-create-user --user-pool-id "$POOL" \
  --username you@example.com \
  --user-attributes Name=email,Value=you@example.com Name=email_verified,Value=true \
                    Name=custom:tenant_id,Value=demo-firm

aws cognito-idp admin-add-user-to-group --user-pool-id "$POOL" \
  --username you@example.com --group-name platform-admin
```

Cognito emails a temporary password. `demo-firm` is the home tenant, whose admins may
create and delete other tenants; change it with `homeTenant` in `cdk.json`.

### 5. Check it came up

```bash
curl -s https://dxxxxx.cloudfront.net/api/health
# {"status":"ok","graph":"connected","vector":"enabled","ontology":"fintech", ...}
```

`graph: connected` is the field worth reading. A healthy container with
`graph: degraded` means Neptune is unreachable, which is almost always the TLS or SigV4
half of the Bolt handshake rather than the container.

### What it costs while idle

Roughly **$90-120/month** in `us-east-1` with nothing happening: Neptune is the largest
line (free for the first 750 hours, then about $50), the NAT gateway about $33, and the
ALB about $16. Page transcription is one Bedrock call per page, so document volume is a
real usage-driven cost on top. `cdk/README.md` has the breakdown and the teardown
gotchas. Several resources are `RETAIN`-on-delete by design and keep billing after
`cdk destroy`.

## Trying it with the demo documents

One demo pack per ontology, committed as PDFs. Take the one matching the tenant's active
pack, since a document read under the wrong vocabulary produces claims no rule matches:

| Pack | Documents | Organising unit |
|---|---|---|
| [`sample/fintech-demo.zip`](sample/fintech-demo.zip) | 3 | Facility (`FAC-`) |
| [`sample/legal-demo.zip`](sample/legal-demo.zip) | 5 | Matter (`NTL-`, `MBC-`, `HAL-`) |
| [`sample/healthcare-demo.zip`](sample/healthcare-demo.zip) | 4 | Encounter (`ENC-`) |

Download the zip from GitHub, unzip it, and upload the PDFs through **Documents, then
Upload** the way a user would upload anything else. That is the point of shipping them as
a plain zip rather than a fixture: the demo walks the same path a real user walks, so
nothing about it is special-cased.

The walkthrough below is the legal pack, which has the most interlocking documents. The
fintech and healthcare packs work the same way with their own references and their own
rules: `group_exposure_via_control` and `related_party_lending` for fintech,
`contraindication_alert` for healthcare.

Each filename starts with its matter reference, and that is what to put in **Attach to
matter** — `NTL-2026-0114` for the three `NTL-` files, `MBC-2024-0431` for the facility
agreement, `HAL-2025-0092` for the authority note. The field takes free text as well as a
selection, because a matter is derived from the documents filed under it and so has nothing
to select until its first document is in.

Getting this right is the difference between a demo that works and one that does not. Both
things worth seeing here are *cross-matter*, so filing everything under one reference, or
leaving it all unassigned, removes exactly what there is to look at.

Uploading is per document, so the graph can be watched filling in. It also means you can
stop after four and see what changes when the fifth arrives, which is the most useful thing
this data does.

An administrator can instead load all five at once with **Admin, then Load sample data**,
which reads the same zip server-side. Either way the pipeline is identical: pages are
rasterised, transcribed by the vision model, chunked, embedded and read for claims. Nothing
is written into the graph directly, so every assertion cites a page and a verbatim span that
the Provenance page resolves.

Regenerate the PDFs after editing the content:

```bash
.venv/bin/python sample/generate_demo_pdfs.py
```

The five matters are deliberately connected, because a demo whose facts do not interlock
cannot show what this system is for:

| Matter | Document | What it demonstrates |
|---|---|---|
| `NTL-2026-0114` | Engagement letter | Client, adverse party, an excluded scope |
| `NTL-2026-0114` | Advice on prospects | Cites `The Aquitaine [2019]` |
| `NTL-2026-0114` | Conflict memorandum | Raises the ethical screen |
| `MBC-2024-0431` | Facility agreement | Meridian is a client *and* a shareholder of the adverse party |
| `HAL-2025-0092` | Authority note | `The Marisol [2025]` overrules `The Aquitaine` |

So a conflict check fires because Meridian appears on both sides, and the stale-authority rule
fires because one matter's advice rests on a case another matter records as overruled.

### Starting again

**Admin, then Reset** removes the derived tiers: the graph, the search index, job state and
the catalog cache. S3 is untouched, so **Replay** rebuilds every document from the bucket and
**Scan catalog** rebuilds the schemas from Glue.

Metric definitions are the exception and are preserved by default. They were authored in the
app and have no upstream source, so nothing rebuilds them. Deleting them needs the metrics box
ticked *and* a separate confirmation.

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

The lanes run in **sequence, not in parallel**, and the order is load-bearing: passages
are retrieved first and the graph walks out from them, so a fact reported with `hops` was
reached *because* a passage cited it. Findings about the evidence, meaning conflicts, stale
authority and ethical screens, are applied by the graph before any model sees it, so a
model never reasons over something the wall refused.

Two surfaces read this:

- **Retrieval** drives it through an agent loop over the MCP server and shows the whole
  transcript: each tool call, its raw result, the lane trace, the evidence chain from
  document to passage to fact to finding, and the wall.
- **The MCP server** exposes the same tools to a third-party agent, on the caller's own
  token. An agent has no identity of its own and sees exactly what the person driving it
  would see.

## Status

Deployed to a real AWS account across six stacks, with **2,127 tests passing**. The
document path works end to end there: a presigned browser upload lands in S3, an S3
notification starts ingestion, and pages are transcribed by a vision model, chunked
with offsets kept, embedded, and read for claims that land in the review queue. Rules
fire over what is approved, conclusions carry their premises, and the ethical wall is
applied by the graph before any model sees the evidence.

**Working end to end:**

- **All three tiers answer.** A governed metric compiles to Athena SQL and runs; graph
  traversal reads approved facts; hybrid retrieves passages and walks out from them.
- **Retrieval agent.** A tool-calling loop over the MCP server, with the whole
  transcript as the answer: every call, every raw result, the lane trace and the wall.
- **Catalog enrichment.** A model proposes plain-language descriptions for tables and
  columns; approved ones are given to the model that writes SQL for ungoverned
  questions.
- **Tenant lifecycle.** Create a tenant with an admin user, and delete one with a
  cascade across identity, settings, graph, vectors and S3.
- **Every read is audited**, with the surface it came in on and the basis it rested on.

**What is not yet true**, because a README that overstates is worse than one that
admits:

- **One task only.** Live ingest and retrieval events and the work itself share a
  single container, so websockets need `appDesiredCount: 1` and a background task dies
  with the container. The poll behind ingest is the correctness guarantee.
- **Nothing has been load tested.** The reasoner joins in Python over a tenant's live
  facts, which is fine for thousands and unexamined beyond that.
- **`CatalogStore` is in-memory.** A restart loses the catalogued table list until the
  next scan, which is also what tier 3's SQL lane reads.
- **Neptune runs on `db.t4g.medium`**, which is memory-starved enough that a read can be
  killed under concurrency. Reads retry once; the instance is the real fix.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest tests/ -q
```
