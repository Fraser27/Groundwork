/**
 * Glossary — the product explaining its own vocabulary.
 *
 * Everything here renders from `epistemic.ts`, which is also what the inline
 * tooltips read. That is deliberate: a glossary maintained separately from the
 * tooltips drifts, and then the same term is explained two different ways.
 *
 * Written for a lawyer, not an engineer. Where a word is genuinely jargon
 * ("epistemic"), it says so and gives the plain equivalent rather than pretending
 * the word is obvious.
 */

import { useEffect, useMemo, useState } from 'react'
import { api, type Ontology } from '../api'
import { getTenantId } from '../auth'
import { ErrorState } from '../components/Shared'
import {
  EPISTEMIC,
  EPISTEMIC_ORDER,
  HELP,
  RETIRED_TIERS,
  REVIEW_STATE_LABEL,
  TIERS,
} from '../epistemic'
import type { EpistemicClass, ReviewState } from '../api'

interface Term {
  term: string
  body: string
  /** Shown in italics under the definition — why it matters, not what it is. */
  why?: string
  group: string
}

/** Terms whose names are opaque enough to need translating before defining. */
const JARGON_NOTE: Record<string, string> = {
  epistemicClass:
    '“Epistemic” means “about knowledge”, so this is simply: how do we know this? You will see it in the interface as the coloured class badge, never as this word.',
  bitemporal:
    '“Bi-temporal” just means two clocks: when something was true in the world, and when we found out about it.',
  premise: 'A premise is a fact that another fact was worked out from.',
}

/**
 * The raw class names, as they appear in exports, the audit trail and the API.
 *
 * The interface shows friendly labels, but anyone reading a CSV export or a log line
 * meets the underlying identifier, and `EXTRACTED_DET` is not self-explanatory.
 */
const CLASS_CODE_NOTE: Record<EpistemicClass, string> = {
  DECLARED: 'Shown as “Declared”.',
  EXTRACTED_DET:
    'Shown as “Extracted (verified)”. “DET” is short for deterministic, meaning the claim was not taken on trust: the quoted words were searched for on the stated page and found. The check settles that the words are there, and nothing about what they mean.',
  EXTRACTED_MODEL:
    'Shown as “Extracted (model)”. A language model read the passage and drew this conclusion from it.',
  INFERRED: 'Shown as “Inferred”. A rule worked it out from other facts.',
  PREDICTED:
    'Shown as “Predicted”. A statistical guess from the shape of the graph, never from a document.',
}

const GROUPS = [
  'How facts are known',
  'Trust and review',
  'Where a fact came from',
  'Vocabulary',
  'Access and isolation',
  'Time',
  'Answering questions',
  'Structured data',
] as const

const TERM_GROUP: Record<string, (typeof GROUPS)[number]> = {
  epistemicClass: 'How facts are known',
  method: 'How facts are known',
  confidence: 'Trust and review',
  confidenceFloor: 'Trust and review',
  reviewState: 'Trust and review',
  supersede: 'Trust and review',
  retraction: 'Trust and review',
  sourceLocator: 'Where a fact came from',
  pageCitation: 'Where a fact came from',
  quote: 'Where a fact came from',
  textOffsets: 'Where a fact came from',
  spanHash: 'Where a fact came from',
  premise: 'Where a fact came from',
  proofTree: 'Where a fact came from',
  governingPredicate: 'Vocabulary',
  descriptivePredicate: 'Vocabulary',
  ontologyDomain: 'Vocabulary',
  tenant: 'Access and isolation',
  matterWall: 'Access and isolation',
  matterAssignment: 'Access and isolation',
  ethicalScreen: 'Access and isolation',
  accessDecision: 'Access and isolation',
  accessAudit: 'Access and isolation',
  platformAdminAccess: 'Access and isolation',
  asOf: 'Time',
  bitemporal: 'Time',
  resolutionTier: 'Answering questions',
  governedMetric: 'Answering questions',
  ungovernedKillSwitch: 'Answering questions',
  ingestState: 'Answering questions',
  timeGrain: 'Structured data',
  additivity: 'Structured data',
}

/** camelCase key -> "Sentence case" heading. */
function humanise(key: string): string {
  const spaced = key.replace(/([A-Z])/g, ' $1').toLowerCase().trim()
  return spaced.charAt(0).toUpperCase() + spaced.slice(1)
}

export default function Glossary() {
  const tenant = getTenantId()
  const [filter, setFilter] = useState('')
  const [onto, setOnto] = useState<Ontology | null>(null)
  const [ontoError, setOntoError] = useState('')
  const [reloadKey, setReloadKey] = useState(0)

  // Two indirections on purpose. The vocabulary comes from the live ontology rather
  // than a copy here, because it is enforced server-side at write time and a stale copy
  // would tell an administrator a predicate is allowed when the API would refuse it.
  // And the *domain* comes from the tenant's own settings rather than a literal, so a
  // healthcare tenant sees the healthcare vocabulary without a rebuild.
  useEffect(() => {
    let cancelled = false
    api
      .getSettings(tenant)
      .then((s) => (cancelled ? null : api.ontology(s.ontology_domain)))
      .then((o) => {
        if (!cancelled && o) {
          setOnto(o)
          setOntoError('')
        }
      })
      .catch((e: Error) => {
        if (!cancelled) {
          setOnto(null)
          setOntoError(e.message)
        }
      })
    return () => {
      cancelled = true
    }
  }, [tenant, reloadKey])

  const terms = useMemo<Term[]>(
    () =>
      Object.entries(HELP).map(([key, body]) => ({
        term: humanise(key),
        body: body as string,
        why: JARGON_NOTE[key],
        group: TERM_GROUP[key] ?? 'Vocabulary',
      })),
    [],
  )

  const needle = filter.trim().toLowerCase()
  const matches = needle
    ? terms.filter(
        (t) =>
          t.term.toLowerCase().includes(needle) ||
          t.body.toLowerCase().includes(needle) ||
          (t.why ?? '').toLowerCase().includes(needle),
      )
    : terms

  const grouped = GROUPS.map((g) => ({
    group: g,
    items: matches.filter((t) => t.group === g),
  })).filter((g) => g.items.length > 0)

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1>Glossary</h1>
          <p className="page-sub">
            Every term this system uses, in plain language. The same wording appears in the
            tooltips throughout the app, so nothing here contradicts what you read elsewhere.
          </p>
        </div>
      </div>

      <div className="search-bar">
        <input
          placeholder="Search terms and definitions…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          aria-label="Search the glossary"
        />
        {needle && (
          <span className="text-dim">
            {matches.length} of {terms.length}
          </span>
        )}
      </div>

      {/* The five classes lead the page: they are the one concept everything else
          depends on, and the badge colours are learned here rather than guessed. */}
      <section className="card">
        <h2>The five ways a fact can be known</h2>
        <p className="text-dim">
          Every fact in the graph carries one of these. It is the single most important
          thing to understand, because it determines whether a fact can influence an
          answer, and whether a person has to approve it first.
        </p>
        <table className="glossary-classes">
          <thead>
            <tr>
              <th>Class</th>
              <th>What it means</th>
              <th>How much to trust it</th>
              <th>Needs approval?</th>
            </tr>
          </thead>
          <tbody>
            {EPISTEMIC_ORDER.map((cls: EpistemicClass) => {
              const meta = EPISTEMIC[cls]
              return (
                <tr key={cls}>
                  <td>
                    <span className="epi-badge" style={{ ['--epi' as string]: meta.colour }}>
                      <span className="epi-dot" aria-hidden="true" />
                      {meta.label}
                    </span>
                    <div className="glossary-code">
                      <code>{cls}</code>
                    </div>
                  </td>
                  <td>
                    {meta.meaning}
                    <div className="glossary-why">{CLASS_CODE_NOTE[cls]}</div>
                  </td>
                  <td className="text-dim">{meta.trust}</td>
                  <td>
                    {meta.autoAsserted ? (
                      <span className="tag tag-green">Goes live directly</span>
                    ) : (
                      <span className="tag tag-amber">Waits for a person</span>
                    )}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </section>

      <section className="card">
        <h2>Review states</h2>
        <p className="text-dim">
          Where a fact sits in the approval process. Derived from the class above rather
          than chosen, so no ingest route can skip review.
        </p>
        <dl className="glossary-list">
          {(Object.keys(REVIEW_STATE_LABEL) as ReviewState[]).map((s) => (
            <div key={s} className="glossary-item">
              <dt>{REVIEW_STATE_LABEL[s]}</dt>
              <dd className="text-dim">
                <code>{s}</code>
              </dd>
            </div>
          ))}
        </dl>
      </section>

      {ontoError && (
        <section className="card">
          <h2>The relationship vocabulary</h2>
          <ErrorState
            title="Could not load the live vocabulary"
            detail={ontoError}
            onRetry={() => setReloadKey((k) => k + 1)}
          />
        </section>
      )}

      {onto && (
        <section className="card">
          <h2>The relationship vocabulary</h2>
          <p className="text-dim">
            Two tiers, and the difference matters. <strong>Governing</strong> relationships
            drive decisions with legal consequence, so the list is closed: a proposed fact
            using anything not on it is refused rather than recorded. That is
            deliberate, the same idea recorded five different ways is how a conflict check
            comes back empty and looks like a clean report.{' '}
            <strong>Descriptive</strong> relationships are open, because a wrong subject-matter
            tag is an inconvenience rather than an exposure.
          </p>

          <h3 className="glossary-subhead">
            Governing, closed list, {onto.governing_predicates.length} in the{' '}
            {onto.domain} vocabulary
          </h3>
          <table className="glossary-classes">
            <thead>
              <tr>
                <th>Relationship</th>
                <th>From → to</th>
                <th>Meaning</th>
              </tr>
            </thead>
            <tbody>
              {onto.governing_predicates.map((p) => (
                <tr key={p.id}>
                  <td>
                    <code>{p.id}</code>
                    {p.symmetric && (
                      <span className="tag tag-neutral" title="Holds in both directions">
                        both ways
                      </span>
                    )}
                  </td>
                  <td className="text-dim">
                    {p.domain.join(', ') || '-'} → {p.range.join(', ') || '-'}
                  </td>
                  <td>
                    {p.description}
                    {p.help && <div className="glossary-why">{p.help}</div>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          <h3 className="glossary-subhead">
            Descriptive, open, {onto.descriptive_predicates.length} defined
          </h3>
          <dl className="glossary-list">
            {onto.descriptive_predicates.map((p) => (
              <div className="glossary-item" key={p.id}>
                <dt>
                  <code>{p.id}</code>
                </dt>
                <dd>
                  {p.description}
                  {p.help && <div className="glossary-why">{p.help}</div>}
                </dd>
              </div>
            ))}
          </dl>

          {onto.rules.length > 0 && (
            <>
              <h3 className="glossary-subhead">Rules that derive new facts</h3>
              <p className="text-dim">
                These run over facts already in the graph and record what follows. Every
                fact they produce keeps a link back to the facts it rests on, so retracting
                one of those retracts the conclusion too.
              </p>
              <dl className="glossary-list">
                {onto.rules.map((r) => (
                  <div className="glossary-item" key={r.id}>
                    <dt>
                      {r.id} <span className="text-dim">{r.version}</span>
                    </dt>
                    <dd>
                      {r.description}
                      <div className="glossary-why">
                        Only fires on facts of class {r.min_premise_class} or stronger.
                      </div>
                    </dd>
                  </div>
                ))}
              </dl>
            </>
          )}
        </section>
      )}

      <section className="card">
        <h2>How a question gets answered</h2>
        <p className="text-dim">
          Three routes, tried in order from most to least trustworthy. Every answer tells
          you which one produced it.
        </p>
        <dl className="glossary-list">
          {([1, 2, 3] as const).map((t) => {
            const meta = TIERS[t]
            return (
              <div key={t} className="glossary-item">
                <dt>
                  <span className="tier-chip" style={{ ['--tier' as string]: meta.colour }}>
                    Tier {t}
                  </span>{' '}
                  {meta.label}
                </dt>
                <dd>
                  {meta.detail}
                  <div className="text-dim glossary-why">{meta.llm}</div>
                </dd>
              </div>
            )
          })}
          {/* Listed, and only listed, because the question log still names it. */}
          {Object.entries(RETIRED_TIERS).map(([t, meta]) => (
            <div key={t} className="glossary-item">
              <dt className="text-dim">
                <span className="tier-chip" style={{ ['--tier' as string]: meta.colour }}>
                  Tier {t}
                </span>{' '}
                {meta.label}
              </dt>
              <dd className="text-dim">
                {meta.detail}
                <div className="glossary-why">
                  It appears in the question log against answers given before it was withdrawn, and
                  nowhere else.
                </div>
              </dd>
            </div>
          ))}
        </dl>
      </section>

      {grouped.map(({ group, items }) => (
        <section className="card" key={group}>
          <h2>{group}</h2>
          <dl className="glossary-list">
            {items.map((t) => (
              <div className="glossary-item" key={t.term}>
                <dt>{t.term}</dt>
                <dd>
                  {t.body}
                  {t.why && <div className="glossary-why">{t.why}</div>}
                </dd>
              </div>
            ))}
          </dl>
        </section>
      ))}

      {grouped.length === 0 && (
        <div className="empty-state">
          <p>No term matches “{filter}”.</p>
        </div>
      )}
    </div>
  )
}
