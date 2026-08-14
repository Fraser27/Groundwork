# API contract

The UI was built before the API existed, so `ui/src/api.ts` is currently the
**de facto specification**. These are the endpoints it actually calls (extracted from
the client, not from a design doc). When `src/api/` is written, match these or change
both sides together.

All routes are under a `/api` base and send `Authorization: Bearer <cognito-jwt>`.

## Endpoints the UI calls

```
GET    /health

GET    /tenants
GET    /tenants/{tenant}/settings
GET    /tenants/{tenant}/dashboard          counts by epistemic class, pending review, activity

GET    /tenants/{tenant}/matters

GET    /tenants/{tenant}/sources
GET    /tenants/{tenant}/tables
GET    /tenants/{tenant}/tables/{full_name}  URL-encoded

POST   /tenants/{tenant}/documents           multipart upload
GET    /tenants/{tenant}/documents?matter_id=
GET    /tenants/{tenant}/documents/{id}

GET    /tenants/{tenant}/assertions?review_state=&epistemic_class=&matter_id=
POST   /tenants/{tenant}/assertions/{id}/approve
POST   /tenants/{tenant}/assertions/{id}/reject
GET    /tenants/{tenant}/assertions/{id}/provenance    source span, or premise proof tree

GET    /tenants/{tenant}/metrics
GET    /tenants/{tenant}/metrics/{id}
POST   /tenants/{tenant}/metrics/{id}/compile          returns SQL without executing
GET    /tenants/{tenant}/metrics/{id}/status

POST   /tenants/{tenant}/query
GET    /tenants/{tenant}/graph/neighbourhood?node_id=&depth=

GET    /ontology/{domain}                     legal | healthcare
```

## Rules the API must honour

**`tenant_id` in the path is not trusted.** It is validated against the tenant claim
in the verified JWT. A mismatch is a 403. The path parameter exists for routing and
readability, not authorization.

**Every handler builds an `AuthContext`** (`src/graph/scope.py`) from the verified
token plus the caller's Cedar grants, and passes it down. No handler reaches the graph
without one.

**`POST /metrics/{id}/compile` must not execute.** It returns the deterministic SQL so
a reviewer can read it before approving — mirrors rosetta-sdl's `execute: false`.

**`POST /query` should report which tier answered** (1 governed metric / 2 graph
traversal / 3 hybrid / 4 LLM SQL) plus the generated SQL, because the UI displays it.
Tier 4 is subject to the ungoverned-query kill switch.

**Assertions listing defaults to `review_state=PENDING`** — that is the review queue,
the most important screen in the product.

**`PREDICTED` assertions are never returned** unless the caller explicitly opts in via
`AuthContext.include_suggestions`. Enforced in `scope.edge_scope`, not in handlers.

## Governance settings

The Admin UI edits these. Backed by `src/governance.py`; overrides persist per tenant
in the graph so they survive a restart.

```
GET    /tenants/{tenant}/governance          current settings + FIELD_HELP text
PATCH  /tenants/{tenant}/governance          partial update, validated
GET    /tenants/{tenant}/governance/blocked  refused ungoverned queries, for review
```

`PATCH` must route through `GovernanceSettings.apply()` so validation runs and a
rejected change cannot half-apply. Return 422 with the `GovernanceError` message on
refusal — the messages are written to be shown to an administrator directly, notably
the cap/floor one.

Changing `embedding_model` is a data migration: warn, and require an explicit
re-embed. Existing vectors came from the previous model and mixing generations
degrades retrieval with no visible error.

## Known drift to reconcile

- The UI has `/tenants/{t}/settings` and `/tenants/{t}/dashboard`, which were not in
  my original endpoint sketch. They are reasonable; implement them. `/settings` should
  probably merge into `/governance`.
- `POST /tenants` (tenant creation) is in the sketch but the UI does not call it.
- ~~**`GET /matters` must return only a COUNT of walled matters, never the objects.**~~
  **Done, and the rule changed.** The leak is fixed — the response no longer ships walled
  matter *objects* inside `matters`, which is what the UI's client-side `matter.walled`
  flag required. But screens are now *named*: the shape is
  `{matters: [...visible...], withheld: [{matter_id, reason, contact}], withheld_count: N}`.
  Two lists, never one with a flag, so a caller iterating `matters` cannot reach a screened
  matter by forgetting a boolean. `Matters.tsx` should render `withheld` as a banner naming
  each matter and its contact. See the ethical-walls entry in `DECISIONS.md`: a bare count
  does not tell a lawyer which client to ask about.
- `ui/src/mocks.ts` holds all mock data behind a single `fallback(realCall, fixture)`
  wrapper — pages call the real API first and fall back only on failure, so they light
  up against a live API with no code change. A "Sample data" flag renders whenever a
  fixture is in use, so a demo cannot be mistaken for a deployment. Deleting that file
  is the last step of reconciliation; anything still importing it is not wired up.
