# LexGraph — working agreement

Read this first. It is the contract for how to work in this repo, not a description
of what the code does (that is `README.md` and `.claude/ARCHITECTURE.md`).

## What this project is

A governed semantic layer over **both** structured (AWS Glue/Athena) and
unstructured (documents) data, for a **whitelabel Legal AI platform**.
Domain-agnostic by construction; ships with a legal ontology and a healthcare pack
that exists to keep that claim honest.

The differentiator, and the reason most design decisions go the way they do: **every
fact carries provenance, so "why does the system believe this?" always has a real
answer** — an exact document page and character span, or the proof tree of an
inference. If a change would weaken that, it is the wrong change.

## The rules that matter

**1. Never write a graph edge except through `build_assertion()`.**
`src/graph/assertions.py` enforces five invariants. Bypassing it silently breaks
auditability, which is the product. There is no "quick path" for a one-off script.

**2. Never build Cypher outside `src/graph/`.**
`src/graph/scope.py` owns tenant/matter scoping. Reads go through
`GraphClient.read_scoped()`, which rejects any query lacking a `{scope}` token.
Tenant isolation here is *logical* (a property filter), so this module is the only
thing standing between two law firms' data. Treat it accordingly.

**3. The metric compiler must never call an LLM.**
Determinism is the whole point of governed metrics. `src/metrics/compiler.py`
compiles YAML to Athena SQL with no model in the path.

**4. Claims are split by checkability, not by who proposed them.**
There is no regex layer; `src/documents/extractors/model.py` is the only extraction path
and its header explains why the parser was deleted. A model proposes, and a mechanical
check either confirms it or does not: a quote found verbatim in the chunk is
`EXTRACTED_DET` and auto-asserts, but only ever as `MENTIONS`, because presence is all a
quote-match establishes. Everything interpretive is `EXTRACTED_MODEL` and waits for a
human. Never widen what auto-asserts.

**5. Governing predicates are closed.**
An extractor proposing `is_counsel_to` when the vocabulary says `REPRESENTS` gets
**rejected at write time**. A conflict check that returns zero rows because of a
synonym looks exactly like a clean conflict check. That is the failure this prevents.

## Code style

- Python 3.11+, `from __future__ import annotations`, ruff `line-length = 100`.
- TypeScript strict in `ui/` and `cdk/`.
- **Comments: default to none.** Only write one where the WHY is non-obvious — a
  hidden constraint, a subtle invariant, a decision that would otherwise look
  arbitrary. Never restate what the code does.
- No multi-paragraph docstrings, no multi-line comment blocks. One short line.
- Calibrate tone against `src/graph/assertions.py` and `src/graph/scope.py`. Those
  two files set the standard: terse, explains reasoning, no filler.
- No emojis, in code or UI.
- The UI is a tool for lawyers. Dense and professional — Linear or Stripe, not a
  marketing site.

## Testing

```bash
.venv/bin/python -m pytest tests/ -q      # must stay green
```

Tests assert behaviour, not smoke. The invariant tests in `tests/test_assertions.py`
and `tests/test_scope.py` are worth more than their line count — if one goes red the
graph has stopped being defensible.

`boto3` calls are injectable so tests need no AWS credentials. Keep it that way.

## Provenance of this design

Two existing projects each solve half the problem, and both are on this machine:

- `../rosetta-sdl` — governs structured data well. **Borrow from it**: the metric
  compiler, the sqlglot SQL firewall, the Glue scanner, the UI look and feel
  (CSS-variable theming, `FieldHelp` tooltips, page conventions).
- `../coa` (AWS `context-ontology-accelerator`) — ontologies, OWL reasoning,
  document ingest, and a genuinely good Cedar authorization model. **Borrow the
  ideas, not the 16 stacks.** Deployed and inspected; see
  `.claude/DECISIONS.md` for what we took and what we deliberately dropped.
