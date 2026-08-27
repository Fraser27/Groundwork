# Decisions, and what they cost

Settled calls with the reasoning attached, so they are not re-litigated from scratch.
Where a decision was a correction of an earlier mistake, the mistake is recorded —
it is usually the more useful half.

## Settled

**Per-edge provenance is the core of the product, not a feature.**
Decided before any code. It is cheap now and architecturally fatal to defer. Every
other decision defers to it.

**Auto-assert tiering.** `DECLARED` and `EXTRACTED_DET` write straight to live;
`EXTRACTED_MODEL` goes to review. Rationale: neither auto-asserted class involves a
model's judgement — a catalog is a system of record, and a regex is reproducible.
Cost: a freshly uploaded matter contributes nothing to answers until someone reviews
the model-extracted edges. Deterministic extractions do show up immediately.

**Bitemporal from day one.** `valid_from`/`valid_until` (world time) plus
`recorded_at`/`superseded_at` (transaction time) on every assertion. "What did the
file show on the date we advised?" is a real legal requirement and retrofitting
either axis is brutal. Query-side rewriting is deliberately minimal for now — the
fields are populated and `scope.edge_scope(as_of=...)` reads them; richer temporal
queries can come later because the data is there.

**Two-tier predicate vocabulary.** Closed for governing predicates, open for
descriptive. The failure being prevented: an extractor records `is_counsel_to`, the
conflict check queries `REPRESENTS`, zero rows come back, and it looks like a clean
result. Not a data-quality annoyance — a malpractice event.

**Retraction is append-only and cascades.** Set `superseded_at`, never `DELETE`, and
transitively supersede every `INFERRED` assertion whose premises include the
retracted one. This is the invariant most systems forget and the one that bites in
litigation.

**Vector writes land before the review gate.** Verbatim text is a quote, not a claim.
Withholding search until review would make uploads invisible for hours. A stricter
posture is defensible but was rejected as unusable.

**MCP on AgentCore Runtime.** User's explicit choice. Inherits the AZ pinning
constraint; documented in `ARCHITECTURE.md`.

**ECS Fargate, not EC2.** rosetta-sdl runs on EC2; this is containerised for
multi-tenant scaling.

**Grounding is `STANDARD` (exact match) only.** COA offers an `ENHANCED` mode where
an LLM adjudicates whether two predicate names mean the same thing. Rejected: that is
a schema decision made by a model, in a system that has to be defensible.

**Governance settings are runtime-configurable, not constants.** `src/governance.py`
holds trust thresholds, kill switches, model ids and ontology domain; defaults from
env, overrides persisted per tenant and editable in the Admin UI. Same pattern as
rosetta-sdl's `system_config` node. Rationale: a firm onboarding a new practice area
wants a different confidence floor, and a model deprecation should be a settings
change rather than a release.

The cap/floor coupling is enforced in code (`GovernanceSettings.validate()`), not
merely documented: `model_confidence_cap` must stay strictly below
`min_confidence_floor`. That gap is what keeps an unreviewed model assertion out of
answers *even if the review gate were bypassed*. The trap it catches is lowering the
floor without lowering the cap — `apply()` refuses it and leaves the original
untouched.

**Ethical walls are named, not silently applied.** *Supersedes an earlier decision here
that walls be reported as an aggregate count with identities never disclosed.*

The old rule had the wrong threat model. It treated a screen as a secret, but an ethical
wall is a documented, acknowledged arrangement and its purpose is that people know it is
there. Silent filtering causes the exact harm the wall exists to prevent: a lawyer runs a
conflict check, reads "no conflicts", and proceeds — while three screened matters *did*
match. A bare count is barely better, because it does not say which client to go and ask
about. "No conflicts found (2 matters withheld — contact risk@firm.com)" is safe; "No
conflicts found" is negligence.

The distinction that actually matters is **inside a firm versus between firms**, not
"aggregate versus identity":

- **Within a tenant**, a refusal names the matter, the recorded reason and a contact.
  `AuthContext.can_read_matter()` returns an `AccessDecision` rather than a bool, and
  `assert_can_read_matter()` raises a `ScopeViolation` carrying `.decision`, `.matter_id`,
  `.reason` and `.contact`, so the HTTP layer builds a structured body instead of parsing a
  sentence back apart.
- **Across tenants**, everything stays silent and indistinguishable from absence. That is a
  confidentiality boundary between two firms, agreed with nobody — not a screen. Tenant-id
  validation and tenant-mismatch handling name nothing.
- `NOT_ASSIGNED` reads differently from `SCREENED`. An assignment gap is somebody
  forgetting to staff you; telling that person to "contact risk" would be nonsense.

Consequences: `GET /matters` returns a separate `withheld` list of named matters with
`reason` and `contact` — never mixed into `matters`, so no caller can treat a screened
matter as readable by forgetting to check a flag. An in-tenant screen is **403** with the
reason and contact; a cross-tenant mismatch stays **404**. In MCP, `search_assertions`
gains `withheld_matters` alongside the count, and a screened id tells the agent it is
screened so it can relay that; an unknown id stays unreachable.

## Infrastructure decisions (from the CDK build)

**OpenSearch Serverless needs two grants, not one.** An IAM policy with
`aoss:APIAccessAll` is insufficient on its own — a separate `CfnAccessPolicy` data
access policy must name the principal, or you get a 403 with no indication the second
half is missing. Lives in `app-stack.ts` (not `data-stack.ts`) because it names the
task role.

**One IAM role with two service principals, not role assumption.** The MCP runtime
assuming the app's task role would need a trust edge from `app` back to `mcp`, which
is a CloudFormation stack cycle. The role is trusted by both `ecs-tasks` and
`bedrock-agentcore` with confused-deputy conditions.

Related trap: CDK attaches policy statements to the stack owning the *role*, so
AgentCore grants written in `mcp-stack.ts` silently emit into the `app` template, and
`cdk deploy GroundworkMcp` alone would not apply them. They live in `app-stack.ts` for
that reason.

**`standbyReplicas: ENABLED` on the vector collection.** With min OCU 0 there is no
floor to double, so an idle group bills nothing either way and the redundancy is free.

**`neptune1.4` parameter group family rather than a pinned `engineVersion`.** Pinning
a specific minor breaks the template the day that minor is retired; a family/engine
mismatch is a create-time failure, so the family is a context knob.

### Two blockers before any deploy

1. **`availabilityZones` is unset**, so CDK picks the first two AZs. AZ *names* are
   shuffled per account, so the AgentCore/OpenSearch VPC-endpoint intersection can
   only be resolved against the target account. Command is in `cdk/README.md`.
2. **The Cognito user pool is `RemovalPolicy.DESTROY`** (`auth-stack.ts:65`). Must
   become `RETAIN` before the first real tenant — there is no import path back for a
   deleted pool.

### Cost

**~$135–190/month idle floor.** Neptune `db.t4g.medium` is free for 750 instance-hours
then ~$50/mo, billing continuously with no scale-to-zero. The eight interface VPC
endpoints (~$58/mo) cost more than the Fargate service and are the first thing to cut
on a throwaway stack.

## The extraction rewrite (2026-08-14) — supersedes earlier extraction decisions

Three decisions recorded earlier were wrong and have been reversed. The reasoning
matters more than the conclusion, because the same traps are easy to walk back into.

**Deleted the regex extractor entirely. Model-only extraction.**
The deterministic parser had real virtues — 0.31 ms, no network, byte-identical across
100 runs — but they did not survive contact with the requirement:

1. Its char offsets indexed **the reconstructed text buffer, not the PDF**, so no
   viewer could seek to them. `verify_offsets` proved they were internally consistent
   with a buffer nobody can see. Re-parsing with a changed reader shifts every offset
   downstream, silently repointing stored assertions.
2. Its allowlists (≈45 reporters, ~20 court patterns) **silently missed anything not
   enumerated**. A novel court simply did not exist in the graph, and the
   `DETERMINISTIC_PREDICATES` guard then blocked the model from rescuing it — two
   individually sound rules combining into a dead end.
3. It **auto-asserted `CITES`**, which implies reliance. "The court declined to follow
   Brown" was recorded as reliance on Brown. The parser cannot read negation,
   hypotheticals, or context; it matched shape, not meaning.

The user's judgement: maintaining allowlists is a cost we are not paying, and a strong
model is better at exactly the classification the regex got wrong.

**Provenance is file + page + verbatim quote.**
What a lawyer actually does: open the PDF at the page, search for the sentence. Works
with any off-the-shelf viewer, no coordinate mapping, robust to re-parsing. Char offsets
demoted to optional debug metadata on `SourceLocator`. A document locator without a
`quote` is now refused at construction — a citation with nothing to search for cannot be
checked.

**Presigned URLs are never persisted.** They expire in hours; an audit trail holding an
expiring credential is provenance that stops resolving. The S3 key is stored, the URL is
minted per request — and only after the caller's matter access is checked, because a
presigned URL bypasses all application authorization once it exists.

**`EXTRACTED_DET` was redefined, not retired.** It no longer means "a regex found it".
It means **a deterministic check confirmed it**: a model proposes that some text appears
on some page, and a string search against the chunk either agrees or does not. The
proposal is probabilistic, the confirmation is not, and the confirmation is what the
class records.

Only `PRESENCE_PREDICATES` (currently `{MENTIONS}`) may be `EXTRACTED_DET`, enforced in
`build_assertion`. A quote-match establishes that text is present and nothing more, so
anything implying significance — `CITES` implies reliance, `ADVERSE_TO` implies a
position — stays `EXTRACTED_MODEL` and is reviewed. That constraint is the fix for the
`CITES` bug above.

**Why the split exists at all:** if every model output required review, a 300-page
bundle would produce hundreds of pending items, and a queue nobody can clear gets
rubber-stamped — destroying the guarantee the queue exists to provide. Splitting on
*checkability* rather than on *who proposed it* keeps the queue to genuine judgement
calls.

## Textract is gone (2026-08-14) — a vision model reads the page

User's explicit instruction, and the reasoning generalises. Textract reads *text*, but
a legal document carries meaning in things that are not text: a cap table, an org
chart, a signature block, handwritten marginalia, a redaction box. Textract returns
nothing for those, and downstream **"nothing" is indistinguishable from "no such
content"** — the same silent-absence failure the closed predicate vocabulary exists to
prevent. A vision model describes a chart in prose, and prose can be quoted, embedded
and cited like any other passage.

Three consequences worth keeping in mind:

**Page numbers became exact rather than inferred.** Each page is rendered and sent as
its own call, so the page number is the page we sent — not something reconstructed from
a coordinate stream. Since provenance is file + page + quote, that is the property that
matters most, and it is now true by construction.

**The OCR model is configured separately from the extraction model** (`OCR_MODEL`,
`GovernanceSettings.ocr_model`). Transcription is mechanical and a cheap model does it
well; judging what a passage means is not. Paying extraction rates to read words off a
page is waste, and the two knobs move for different reasons.

**Cost moved from per-page-of-OCR to per-page-of-inference**, which is why `MAX_PAGES`
(400) is a hard ceiling in `parse.py` rather than a config default. A 400-page bundle is
one model call per page; the limit belongs in the code, not on an invoice.

Infrastructure that went with it: the Textract VPC interface endpoint, and the six
`textract:*` IAM actions on the task role. Bedrock already had both an endpoint and an
`InvokeModel` grant, so the vision call needed nothing new.

`pymupdf` came back as a dependency to rasterise pages — see *Open questions*, it is
AGPL and confined to one class for that reason.

## Drift fixed 2026-08-14 — six items where the code contradicted its own docs

Found by the docs agent while writing `docs/`, which is a good argument for writing
documentation from the code rather than from memory. A wrong explanation is a bug.

1. **README claimed structural tenant isolation** ("its own named graph") — the exact
   error `scope.py` already documented as corrected. Both tenant and matter isolation
   are property filters. README now carries the correction and the reason.
2. **README described `EXTRACTED_DET` as a "deterministic parser"** and char offsets as
   primary provenance. Both wrong since the extraction rewrite. Now: a check confirmed
   it, and provenance is file + page + quote.
3. **Four governance settings were stored, validated, warned about in the UI, and read
   by nothing.** `auto_assert_deterministic`, `require_review_for_governing`,
   `enforce_closed_vocabulary`, `effective_model_confidence()`. An administrator could
   switch a toggle, believe behaviour had changed, and be wrong — worse than having no
   toggle.

   Fixed with `ReviewPolicy` in `assertions.py`, threaded through `build_assertion` and
   passed from the ingest route. **Deliberately one-directional**: every field can send
   *more* to review, never less. There is no setting that makes an unreviewed model
   claim live, because that is the guarantee the contract rests on.

   `require_review_for_governing` earns its keep on `DECLARED` facts: a case-management
   export asserting `ADVERSE_TO` would otherwise drive a conflict check with nobody
   having read it.
4. **`legal.yaml` help text described the deleted regex extractor** — `CITES` said
   "Produced by the Bluebook citation parser, so normally EXTRACTED_DET", which
   `build_assertion` now actively refuses.
5. **`DEFAULT_EXTRACTION_MODEL` differed between `constants.py` and `governance.py`.**
   The pipeline read config while the UI showed governance, so an administrator saw a
   model id the system was not using. Governance now imports from `constants.py`.
6. **`symmetric` and `min_premise_class` were parsed and exposed but enforced nowhere.**
   `ADVERSE_TO` declares `symmetric: true`, so "A adverse to B" and "B adverse to A"
   produced two content hashes and two edges for one fact — fragmenting the exact signal
   a conflict check reads. `Ontology.canonical_pair()` collapses them and the extractor
   applies it. `rule_premise_floor()` exposes the floor for the reasoning engine to
   enforce when it exists.

## Corrections worth remembering

**"Named graph per tenant" was wrong.**
An early draft claimed structural tenant isolation via one named graph each. Named
graphs are RDF/SPARQL only; Neptune Database holds a single property graph per
cluster and openCypher cannot see them. Corrected in `src/graph/scope.py`, including
its docstring, rather than quietly patched — the honest version raises the stakes on
that module, since it is now the only thing between two firms' data.

`graph_name()` became `cluster_key()` and is documented as a seam for a future
per-tenant storage split.

**Neptune Analytics does support many graphs — it still does not fit.**
Analytics has discrete graph resources, so a graph per tenant is expressible. But
each provisions a 32 m-NCU (~32 GB) minimum with no autoscaling, making it a
per-customer deployment shape rather than multi-tenant SaaS. Also: openCypher only
(no SPARQL) and no full-text search. AWS's own COA chose Neptune Database +
OpenSearch for exactly these reasons.

## Deliberately dropped from COA

COA needs 16 stacks; we need 6. What went, and why:

| Dropped | Reason |
|---|---|
| DataZone / SMUS namespaces | Tenant = DynamoDB row + property filter. Also the source of COA's worst teardown failures ("domain not empty") |
| VKG / Ontop | SPARQL federation over source SQL. COA is table-first; we are document-first |
| metric-service stack | Replaced by rosetta-style YAML metrics inside the app |
| Smithy codegen | Java 17 + Gradle + openapi-generator. Real build tax, no benefit at this size |
| `guardrail`, `edge-waf` stacks | Folded into `auth` and `web` |
| `serve`/`sources`/`data-layer`/`ontology` as separate stacks | Merged into `app`; they are just containers |

## Kept from COA

- The **Cedar authorization model** — entity/action taxonomy, and critically the
  `forbid`-overrides-everything pattern. COA uses it to freeze archived namespaces;
  we use it for ethical walls.
- **Agents have no standalone identity** — every MCP call carries the acting user's
  delegated token and is authorized as that user. Non-negotiable for privilege.
- **Human review before entering the graph** — though we review *assertions*, where
  COA reviewed metadata.
- **Terminal-but-recoverable failure states** carrying a reason, so a source can be
  deleted or re-run rather than stranded mid-scan.

## Kept from rosetta-sdl

- Metric compiler: YAML -> deterministic Athena SQL, **no LLM**. Time-grain
  governance, additivity properties, join fanout risk detection.
- sqlglot AST SQL firewall: recursive table extraction from FROM/JOIN/CTEs/
  subqueries/UNIONs, fails closed on parse errors.
- Glue catalog scanner and Bedrock metadata enrichment.
- Kill switch for ungoverned queries with a blocked-query audit trail.
- UI: CSS-variable light/dark theming, `FieldHelp` tooltips, collapsible sidebar,
  page conventions.

## Open questions for later

1. **Licensing — now live, not hypothetical.** `pymupdf` (AGPL-3.0) is a direct
   dependency again, because rasterising a page is a prerequisite for the vision
   reader. It is confined to `PyMuPDFRenderer` behind the `PageRenderer` protocol and
   does nothing but PDF page -> PNG, so `pdftoppm`, Ghostscript or a Lambda layer
   replaces it in one class. **That swap must happen before commercial release.**
   owlready2 (LGPL-3.0) is still avoided in favour of `rdflib` + `owlrl` (BSD).
2. **Reasoning engine.** No OWL reasoner is wired up yet. Options: `owlrl` over
   rdflib (OWL-RL profile, probably sufficient for legal rules), or a small
   purpose-built forward-chainer over the rules in `ontologies/*.yaml`. The rule
   definitions and `INFERRED` plumbing already exist.
3. **Full-text search.** OpenSearch gives vector plus BM25. If litigators need real
   terms-and-connectors search, that needs explicit design — it is the feature that
   forced AWS to keep two stores.
4. **Amazon Verified Permissions vs. hand-rolled Cedar.** COA stores policies in
   DynamoDB and evaluates in a Lambda authorizer. AVP is managed Cedar. Verify AVP
   supports the `forbid`-override pattern before committing.
5. **Neptune Serverless vs. provisioned.** Currently `db.t4g.medium` provisioned.
   Serverless may suit spiky per-firm load better, but has its own floor and is
   unverified under a reasoning workload.
6. **Entity resolution thresholds.** Semantica's default fuzzy threshold of 0.7 would
   happily merge "Acme Corp" with "Acme Holdings". In legal a bad merge is worse than
   a missed one — raise the threshold and route near-misses to review.
7. **A rule block has no user dimension.** An ethical screen is per user
   (`MatterScreen`, keyed `tenant/user/matter`); a `POTENTIAL_CONFLICT` block is not, so
   it withholds a party's facts from everyone including the partner who acts for that
   party. Narrowing it to `blocks: object` stopped a conflict blacking out the matter,
   which was the acute problem, but the residual stands. The coherent answer is probably
   that a derived conflict should *raise a screen* over the individuals who need
   walling off — the control the fixture's own risk memo describes, naming five people —
   rather than `Block` growing a `user_id`. That is a design change, not a patch.
8. **`conflict_check` reads "some counsel represents X", not "we represent X".**
   `Counsel` covers external firms by design, so `counsel:opposing-firm REPRESENTS
   party:calder` plus an adversity would flag ordinary litigation as a conflict. Latent
   rather than live: nothing currently extracts opposing counsel as `REPRESENTS`. The
   discriminator has to be DECLARED by a case-management system rather than read off a
   page, so the fix is a pack change plus an extraction path plus a feed that does not
   exist yet.
