# Groundwork UI

React + TypeScript + Vite front end for Groundwork.

## Run it

```bash
cd ui
npm install
npm run dev          # http://localhost:3000
```

`/api/*` is proxied to `http://localhost:8000`, so run the FastAPI app on port 8000
and no CORS configuration is needed. If the API is not running, every page falls
back to fixtures from `src/mocks.ts` and shows a "sample data" flag in its header.

```bash
npm run typecheck    # tsc -b
npm run build        # tsc -b && vite build -> dist/
npm run lint
npm run preview      # serve the built bundle
```

Use `npm run typecheck` rather than `npx tsc` — there is an unrelated package on
npm actually named `tsc`, and `npx` will fetch that instead of the local compiler.

If `npm run build` fails with `Cannot find module './rolldown-binding.darwin-arm64.node'`,
npm resolved the wrong platform binary for Vite's bundler. Fix it with:

```bash
npm install --os=darwin --cpu=arm64
```

## Configuration

Auth is disabled unless Cognito config is present, which makes local development
against a bare API work with no setup.

In production, CDK writes `/runtime-config.json` and the app reads it at startup:

```json
{
  "cognitoUserPoolId": "...",
  "cognitoClientId": "...",
  "cognitoRegion": "eu-west-2",
  "cognitoDomain": "groundwork-123456789012.auth.eu-west-2.amazoncognito.com",
  "defaultTenantId": "demo-firm"
}
```

For local development, a `.env.local` with `VITE_`-prefixed equivalents works as a
fallback (see `src/vite-env.d.ts` for the names).

The tenant id is read from the `custom:tenant_id` claim on the ID token. It is only
used to build request paths — the server derives the real tenant from the verified
JWT, so editing it in local storage widens nothing.

## What each page is for

| Page | Purpose |
|---|---|
| Dashboard | Counts broken down by epistemic class, pending review, recent activity |
| Ask | Natural-language query showing which resolution tier answered and the SQL |
| Review queue | Pending model-extracted claims with the source span that produced each |
| Matters | Matter list and detail; walled matters are shown as withheld |
| Documents | Upload, the ingest state machine, per-document extracted facts |
| Graph | Force-directed canvas; edges coloured by epistemic class, click for provenance |
| Audit | Search all assertions, proof trees, retraction history, bitemporal as-at reads |
| Tables | Structured sources discovered from the catalogue |
| Metrics | Governed metric CRUD and approval |
| Admin | Tenant settings, ontology pack, models, trust floor, ungoverned kill switch |

## Where the explanations live

Every non-obvious term is explained in plain language for a lawyer, not a data
engineer. All of that copy is in one place:

- `src/epistemic.ts` — `HELP` is the glossary; `EPISTEMIC` describes the five
  classes; `TIERS` describes the four resolution tiers. Edit copy here, not in the
  pages.
- `src/components/FieldHelp.tsx` — the `(?)` tooltip. CSS-only, reachable by
  keyboard.
- `src/components/EpistemicBadge.tsx` — the class badge; its tooltip says both what
  the class means and how much to trust it.
- `src/components/ProvenancePanel.tsx` — the "why does the system believe this"
  panel: quoted span with page and character offsets, or the proof tree.

Predicate and entity help text comes from `ontologies/legal.yaml` via
`GET /ontology/{domain}` and is surfaced as tooltips on the Admin page.

## Theming

Light and dark are both first-class. Everything is a CSS custom property in the
`:root` and `[data-theme="dark"]` blocks at the top of `src/index.css`; the choice
is persisted to `localStorage` under `theme`. The canvas renderer reads the same
variables back off `:root` via `getComputedStyle`, so the graph re-themes with
everything else.

The five epistemic colours (`--epi-declared`, `--epi-extracted-det`,
`--epi-extracted-model`, `--epi-inferred`, `--epi-predicted`) are used consistently
in the badge, the graph edges, the dashboard tiles and the review cards, so the
vocabulary only has to be learned once.

## Removing the mock data

`src/mocks.ts` is the only place fixtures live. To remove it:

1. `grep -rn "mocks" src` to find the call sites.
2. Unwrap each `fallback(api.x(...), MOCK_Y)` to just `api.x(...)`.
3. Delete `src/mocks.ts` and the `MockFlag` component from `src/components/Shared.tsx`.
