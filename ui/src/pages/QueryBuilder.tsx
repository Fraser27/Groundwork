/**
 * QueryBuilder — ask a question, and see which part of the system answered it.
 *
 * The tier is not a diagnostic detail: it tells the reader how much of the answer
 * was generated. Tier 1 is a compiled metric with no model involved; tier 4 is a
 * model writing SQL. Both are legitimate, and a reader is entitled to know which
 * one they got before relying on the number.
 */

import { useEffect, useState, type CSSProperties } from 'react'
import { Link } from 'react-router-dom'
import {
  api,
  type Matter,
  type QueryAnswer,
  type QueryBlock,
  type QueryHit,
  type QueryPassage,
  type QueryResult,
  type QueryRows,
  type ResolutionTier,
  type TenantSettings,
} from '../api'
import { getTenantId } from '../auth'
import { HELP, TIERS } from '../epistemic'
import { useProvenance } from '../useProvenance'
import ConfidenceBar from '../components/ConfidenceBar'
import DocumentViewer from '../components/DocumentViewer'
import EpistemicBadge from '../components/EpistemicBadge'
import FieldHelp from '../components/FieldHelp'
import ProvenancePanel from '../components/ProvenancePanel'
import { ErrorState, Spinner, TierBadge } from '../components/Shared'
import { epiStyle } from '../format'

/**
 * `answer` is shaped by whichever tier answered, so it is narrowed rather than rendered.
 *
 * Rendering it directly is what blanked this page: tier 2 puts a list of assertions there, and a
 * list of objects as a React child throws. Narrowing in one place beats guessing at each use.
 */
function asRows(answer: QueryAnswer): QueryRows | null {
  if (answer && typeof answer === 'object' && 'columns' in answer && Array.isArray(answer.rows)) {
    return answer as QueryRows
  }
  return null
}

function asHits(answer: QueryAnswer): QueryHit[] {
  if (Array.isArray(answer)) return answer
  if (answer && typeof answer === 'object' && 'related' in answer) return answer.related ?? []
  return []
}

function asPassages(answer: QueryAnswer): QueryPassage[] {
  if (answer && typeof answer === 'object' && 'passages' in answer) return answer.passages ?? []
  return []
}

/** Tier 2 explains a fact by the terms it matched; tier 3 walked to it, so distance is the
 *  explanation -- ten edges read alike otherwise, quoting indistinguishable from inferring. */
function whyIncluded(h: QueryHit): string {
  const parts: string[] = []
  if (h.matched_on.length > 0) parts.push(`matched on ${h.matched_on.join(', ')}`)
  if (h.hops != null) {
    parts.push(h.hops === 1 ? 'direct from the passage' : `${h.hops} hops from the passage`)
  }
  if (h.source.filename) {
    parts.push(h.source.filename + (h.source.page != null ? `, page ${h.source.page}` : ''))
  }
  return parts.join(' · ')
}

/** Entity ids are `kind:slug`. The slug is what a reader recognises; the kind is noise here. */
function entityLabel(id: string): string {
  const slug = id.includes(':') ? id.slice(id.indexOf(':') + 1) : id
  return slug.replace(/[-_]/g, ' ')
}

/** `rule` is `ethical_screen` for a recorded wall; anything else is an ontology rule that fired. */
function isScreen(b: QueryBlock): boolean {
  return b.rule === 'ethical_screen'
}

/** The tier is what each example is meant to demonstrate, not a promise about the answer. */
const EXAMPLES: { q: string; tier: ResolutionTier }[] = [
  { q: 'What were fees billed by practice area last quarter?', tier: 1 },
  { q: 'Does acting for Halveston create a conflict?', tier: 2 },
  {
    q: 'Which open matters have unbilled work and an adverse party we also act for?',
    tier: 3,
  },
  {
    q: 'What is the average number of days between opening a matter and the first invoice?',
    tier: 4,
  },
]

export default function QueryBuilder() {
  const tenant = getTenantId()
  const [question, setQuestion] = useState('')
  const [matterId, setMatterId] = useState('')
  const [asOf, setAsOf] = useState('')
  const [matters, setMatters] = useState<Matter[]>([])
  const [settings, setSettings] = useState<TenantSettings | null>(null)
  const [floorOverride, setFloorOverride] = useState<number | null>(null)
  const [result, setResult] = useState<QueryResult | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState('')
  // The matter list and trust floor, not the answer. Asking still works without them.
  const [contextError, setContextError] = useState('')
  const [openProvenance, setOpenProvenance] = useState<string | null>(null)
  const [openDocument, setOpenDocument] = useState<{
    documentId: string
    filename: string
    page: number
    quote: string | null
  } | null>(null)
  const { provenance, error: provError } = useProvenance(tenant, openProvenance)

  useEffect(() => {
    Promise.all([api.listMatters(tenant), api.getSettings(tenant)])
      .then(([m, s]) => {
        setMatters(m.matters)
        setSettings(s)
        setContextError('')
      })
      .catch((e: Error) => setContextError(e.message))
  }, [tenant])

  const floor = floorOverride ?? settings?.min_confidence ?? 0.8

  const ask = async (q: string) => {
    setRunning(true)
    setError('')
    setResult(null)
    setOpenProvenance(null)
    try {
      const res = await api.query(tenant, {
        question: q,
        matter_id: matterId || undefined,
        as_of: asOf || undefined,
        min_confidence: floor,
      })
      setResult(res)
    } catch (e) {
      setError((e as Error).message.replace(/^\d+:\s*/, ''))
    } finally {
      setRunning(false)
    }
  }

  const tierMeta = result ? TIERS[result.tier] : null
  const rows = result ? asRows(result.answer) : null
  const hits = result ? asHits(result.answer) : []
  const passages = result ? asPassages(result.answer) : []
  // `assertions_used` is the recorded audit trail and the thing worth deep-linking; it is
  // empty for tiers 1 and 4, and the hits are the same ids for 2 and 3, so fall back rather
  // than lose the action if the field ever arrives absent.
  const usedIds = result?.assertions_used?.length
    ? result.assertions_used
    : hits.map((h) => h.assertion_id)
  // `?? []` rather than trusting the declared type: the field is new, and a type is a claim
  // about the response, not a check on one.
  const blocks = result?.blocks ?? []
  const screens = blocks.filter(isScreen)

  return (
    <>
      <div className="page-header">
        <div className="page-header-row">
          <div>
            <h2>Ask</h2>
            <p>
              Ask in plain language. The answer says which part of the system produced it, shows the
              SQL where there is any, and cites the exact assertions behind every claim.
            </p>
          </div>
        </div>
      </div>

      {contextError && (
        <div className="banner banner-warn">
          <span>
            <strong>Could not load the matter list or the trust floor.</strong> {contextError}. You
            can still ask, but the matter filter is empty and the default floor is in use.
          </span>
        </div>
      )}

      <div className="card">
        <div className="form-group">
          <label>Question</label>
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="e.g. Which matters cite authority that has since been overruled?"
            onKeyDown={(e) => {
              if (e.key === 'Enter' && (e.metaKey || e.ctrlKey) && question.trim()) ask(question)
            }}
          />
          <p className="hint">Cmd/Ctrl + Enter to ask.</p>
        </div>

        <div className="toolbar" style={{ marginBottom: 12 }}>
          <div className="toolbar-field">
            <label>
              Matter
              <FieldHelp text={HELP.matterWall} />
            </label>
            <select value={matterId} onChange={(e) => setMatterId(e.target.value)}>
              <option value="">All matters I can see</option>
              {matters
                .filter((m) => !m.walled)
                .map((m) => (
                  <option key={m.matter_id} value={m.matter_id}>
                    {m.name ? `${m.matter_id} - ${m.name}` : m.matter_id}
                  </option>
                ))}
            </select>
          </div>
          <div className="toolbar-field">
            <label>
              As at
              <FieldHelp title="Bitemporal read" text={HELP.asOf} />
            </label>
            <input type="date" value={asOf} onChange={(e) => setAsOf(e.target.value)} />
          </div>
          <div className="toolbar-field">
            <label>
              Trust floor
              <FieldHelp text={HELP.confidenceFloor} />
            </label>
            <input
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={floor}
              onChange={(e) => setFloorOverride(Number(e.target.value))}
              style={{ minWidth: 90 }}
            />
          </div>
          <div className="toolbar-field toolbar-spacer">
            <label>&nbsp;</label>
            <button
              className="btn btn-primary"
              onClick={() => ask(question)}
              disabled={running || !question.trim()}
            >
              {running ? 'Asking…' : 'Ask'}
            </button>
          </div>
        </div>

        {asOf && (
          <div className="banner banner-info" style={{ marginBottom: 0 }}>
            <span>
              Reading the graph as it stood on {asOf}. Facts recorded later are excluded, and facts
              since retracted are included, this is what the file showed on that date, not what it
              shows now.
            </span>
          </div>
        )}
      </div>

      <div className="card">
        <div className="card-header">
          <h3>Try one of these</h3>
          <span className="card-note">Each takes a different route through the system.</span>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
          {EXAMPLES.map((ex) => (
            <button
              key={ex.q}
              className="btn btn-ghost btn-sm"
              style={{ justifyContent: 'flex-start', textAlign: 'left', whiteSpace: 'normal' }}
              onClick={() => {
                setQuestion(ex.q)
                ask(ex.q)
              }}
            >
              <TierBadge tier={ex.tier} />
              {ex.q}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="banner banner-error" style={{ marginTop: 16 }}>
          <span>{error}</span>
        </div>
      )}

      {running && <Spinner />}

      {result && tierMeta && (
        <>
            <div
              className="tier-banner"
              style={{ marginTop: 16, '--tier-colour': tierMeta.colour } as CSSProperties}
            >
              <div className="tier-num">{result.tier}</div>
              <div className="tier-text">
                <h4>
                  {tierMeta.label}
                  <FieldHelp text={HELP.resolutionTier} />
                  <span className={`tag ${result.governed ? 'tag-green' : 'tag-orange'}`}>
                    {result.governed ? 'governed' : 'ungoverned'}
                  </span>
                </h4>
                <p>{result.explanation}</p>
                <p style={{ color: tierMeta.colour, fontWeight: 550 }}>{tierMeta.llm}</p>
              </div>
            </div>

            {result.warnings.length > 0 && (
              <div className="banner banner-warn" style={{ marginTop: 16 }}>
                <span>{result.warnings.join(' ')}</span>
              </div>
            )}

            {blocks.length > 0 && (
              <div className="withheld-block" style={{ marginTop: 16 }}>
                <div className="withheld-block-head">
                  <h3>
                    {blocks.length} {blocks.length === 1 ? 'block' : 'blocks'} applied to this
                    answer
                    <FieldHelp text={HELP.ethicalScreen} />
                  </h3>
                  <span className="tag tag-red">
                    {screens.length === blocks.length ? 'Screened' : 'Withheld'}
                  </span>
                </div>
                <p className="withheld-block-note">
                  Applied before anything was written or summarised, and by the graph rather than a
                  model, so no part of the answer above rests on what is listed here. They are
                  named on purpose: an answer that looks complete because the inconvenient part
                  was invisible is the failure a screen exists to prevent.
                </p>
                <div className="withheld-list">
                  {blocks.map((b, i) => (
                    <div className="withheld-item" key={`${b.rule}-${b.subject}-${i}`}>
                      <div className="withheld-item-head">
                        <strong>{b.matter_id ?? entityLabel(b.subject)}</strong>
                        <code>{b.rule || 'blocked'}</code>
                      </div>
                      <div className="withheld-field">
                        <span className="withheld-field-label">Reason recorded</span>
                        {b.reason}
                      </div>
                      {isScreen(b) && (
                        <div className="withheld-field">
                          <span className="withheld-field-label">Who to contact</span>
                          {b.contact ?? (
                            <span className="dim">
                              No contact was given. Ask your risk team about this matter.
                            </span>
                          )}
                        </div>
                      )}
                      <div className="withheld-field">
                        <span className="withheld-field-label">In the graph</span>
                        {/* Stated rather than linked, on purpose. `?highlight=` takes assertion
                            ids and a block has none — a screen is a grant, not a fact. A link
                            filtered to a screened matter would draw an empty canvas reading as
                            "this matter holds nothing", which is the silent failure this card
                            exists to prevent. */}
                        <span className="dim">
                          Nothing to open.{' '}
                          {isScreen(b)
                            ? 'A screen is a recorded instruction, not an assertion, and the ' +
                              'facts it covers are never returned to you, so there is no ' +
                              'subgraph to draw.'
                            : 'A rule block names what was refused, not an edge you can open.'}
                          {usedIds.length > 0 &&
                            ' What the answer did use opens in the graph below.'}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {rows && (
              <div className="card">
                <div className="card-header">
                  <h3>Result</h3>
                  <span className="card-note">
                    {rows.rows.length} row{rows.rows.length === 1 ? '' : 's'}
                  </span>
                </div>
                <div style={{ overflowX: 'auto' }}>
                  <table className="data-table">
                    <thead>
                      <tr>
                        {rows.columns.map((c) => (
                          <th key={c}>{c}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {rows.rows.map((r, i) => (
                        <tr key={i}>
                          {r.map((v, j) => (
                            <td key={j} className={typeof v === 'number' ? 'num' : ''}>
                              {typeof v === 'number' ? v.toLocaleString() : (v ?? '-')}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {passages.length > 0 && (
              <div className="card">
                <div className="card-header">
                  <h3>
                    Passages cited
                    <FieldHelp text={HELP.sourceLocator} />
                  </h3>
                  <span className="card-note">{passages.length}</span>
                </div>
                {passages.map((p, i) => (
                  <div className="citation" key={`${p.document_id}-${p.char_start ?? i}`}>
                    <span className="citation-num">[{i + 1}]</span>
                    <div className="citation-body">
                      {p.text && <div className="citation-quote">{p.text}</div>}
                      <div className="citation-loc">
                        {p.filename ?? p.document_id}
                        {p.page != null ? ` · page ${p.page}` : ''}
                      </div>
                    </div>
                    {p.page != null && (
                      <button
                        className="btn btn-ghost btn-sm"
                        onClick={() =>
                          setOpenDocument({
                            documentId: p.document_id,
                            filename: p.filename || p.document_id,
                            page: p.page as number,
                            quote: p.text ?? null,
                          })
                        }
                      >
                        Open at page {p.page}
                      </button>
                    )}
                  </div>
                ))}
              </div>
            )}

            {hits.length > 0 && (
              <div className="card">
                <div className="card-header">
                  <h3>
                    {result.tier === 3 ? 'Related facts in the graph' : 'Facts that answer this'}
                    <FieldHelp text="Each row is an assertion the read was willing to trust: it cleared the confidence floor and its review state allows it to be used. The matched terms say why it came back, so a surprising result traces to the word that pulled it in." />
                  </h3>
                  <div className="card-header-actions">
                    <span className="card-note">{hits.length}</span>
                    {usedIds.length > 0 && (
                      <Link
                        to={`/graph?highlight=${usedIds.map(encodeURIComponent).join(',')}`}
                        className="btn btn-ghost btn-sm"
                        title="Open these assertions in the graph, drawn against the surrounding facts"
                      >
                        See in graph
                      </Link>
                    )}
                  </div>
                </div>
                <div className="path-chain">
                  {hits.map((h) => (
                    <div
                      key={h.assertion_id}
                      className="path-hop"
                      style={epiStyle(h.epistemic_class)}
                    >
                      <EpistemicBadge
                        epistemicClass={h.epistemic_class}
                        size="sm"
                        showLabel={false}
                      />
                      <span>
                        <strong>{entityLabel(h.subject_id)}</strong>{' '}
                        <span className="prov-pred">{h.predicate}</span>{' '}
                        <strong>{entityLabel(h.object_id)}</strong>
                        <span className="dim" style={{ display: 'block', fontSize: 11.5 }}>
                          {whyIncluded(h)}
                        </span>
                      </span>
                      <ConfidenceBar value={h.confidence} floor={floor} width={54} />
                      <button
                        className="btn btn-ghost btn-sm"
                        style={{ marginLeft: 'auto' }}
                        onClick={() => setOpenProvenance(h.assertion_id)}
                      >
                        Why?
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {!rows && hits.length === 0 && passages.length === 0 && (
              <div className="card">
                <div className="card-header">
                  <h3>Answer</h3>
                </div>
                <div className="answer-block">
                  Nothing came back. The question reached {tierMeta.label.toLowerCase()}, but no
                  row, fact or passage cleared the trust floor of {floor.toFixed(2)}.
                  {blocks.length > 0 && (
                    <>
                      {' '}
                      This is not the same as nothing existing: {blocks.length}{' '}
                      {blocks.length === 1 ? 'block is' : 'blocks are'} in force, listed above.
                      Treat the empty result as incomplete rather than clean.
                    </>
                  )}
                </div>
              </div>
            )}

            {result.sql && (
              <div className="card">
                <div className="card-header">
                  <h3>
                    {result.tier === 4 ? 'Generated SQL' : 'Compiled SQL'}
                    <FieldHelp
                      text={
                        result.tier === 4
                          ? 'Written by a language model against the real schema, then checked by the query firewall. It is shown in full because you should read it before relying on the figure.'
                          : 'Compiled from the governed metric definition. Deterministic: the same definition always produces this query, with no model involved.'
                      }
                    />
                  </h3>
                  <span className={`tag ${result.tier === 4 ? 'tag-orange' : 'tag-green'}`}>
                    {result.tier === 4 ? 'model-written' : 'deterministic'}
                  </span>
                </div>
                <pre className="code-block">{result.sql}</pre>
              </div>
            )}

            <div className="card">
              <div className="card-header">
                <h3>
                  The four routes
                  <FieldHelp text={HELP.resolutionTier} />
                </h3>
              </div>
              <div className="tier-ladder">
                {([1, 2, 3, 4] as const).map((t) => (
                  <div
                    key={t}
                    className={`tier-ladder-row${result.tier === t ? ' active' : ''}`}
                    style={{ '--tier-colour': TIERS[t].colour } as CSSProperties}
                  >
                    <span className="tier-ladder-num">{t}</span>
                    <span>
                      <strong>{TIERS[t].label}.</strong> {TIERS[t].detail}
                    </span>
                  </div>
                ))}
              </div>
            </div>
        </>
      )}

      {openProvenance && (
        <div className="modal-overlay" onClick={() => setOpenProvenance(null)}>
          <div
            className="modal modal-wide"
            onClick={(e) => e.stopPropagation()}
            style={{ padding: 0, overflow: 'hidden' }}
          >
            {provError ? (
              <div style={{ padding: 20 }}>
                <ErrorState title="Could not load this provenance" detail={provError} />
              </div>
            ) : provenance ? (
              <ProvenancePanel
                provenance={provenance}
                confidenceFloor={floor}
                onClose={() => setOpenProvenance(null)}
              />
            ) : (
              <Spinner />
            )}
          </div>
        </div>
      )}

      {openDocument && (
        <DocumentViewer
          tenant={tenant}
          documentId={openDocument.documentId}
          filename={openDocument.filename}
          page={openDocument.page}
          quote={openDocument.quote}
          onClose={() => setOpenDocument(null)}
        />
      )}
    </>
  )
}
