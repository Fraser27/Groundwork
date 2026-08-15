/**
 * QueryBuilder — ask a question, and see which part of the system answered it.
 *
 * The tier is not a diagnostic detail: it tells the reader how much of the answer
 * was generated. Tier 1 is a compiled metric with no model involved; tier 4 is a
 * model writing SQL. Both are legitimate, and a reader is entitled to know which
 * one they got before relying on the number.
 */

import { useEffect, useState, type CSSProperties } from 'react'
import { api, type Matter, type QueryResult, type ResolutionTier, type TenantSettings } from '../api'
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
                    {m.matter_id} - {m.name}
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
          {result.blocked ? (
            <div className="banner banner-warn" style={{ marginTop: 16 }}>
              <span>
                <strong>Refused.</strong> {result.blocked_reason} No approved governed metric covers
                this question, and this tenant is configured to refuse rather than fall back to
                model-generated SQL. The question has been logged, it is a candidate for a new
                governed metric.
              </span>
            </div>
          ) : (
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
                    {result.metric_id && (
                      <span className="tag tag-blue tag-mono">{result.metric_id}</span>
                    )}
                  </h4>
                  <p>{result.tier_reason}</p>
                  <p style={{ color: tierMeta.colour, fontWeight: 550 }}>{tierMeta.llm}</p>
                </div>
              </div>

              <div className="card">
                <div className="card-header">
                  <h3>Answer</h3>
                </div>
                <div className="answer-block">{result.answer}</div>
              </div>

              {result.rows && (
                <div className="card">
                  <div className="card-header">
                    <h3>Result</h3>
                    <span className="card-note">{result.rows.rows.length} rows</span>
                  </div>
                  <div style={{ overflowX: 'auto' }}>
                    <table className="data-table">
                      <thead>
                        <tr>
                          {result.rows.columns.map((c) => (
                            <th key={c}>{c}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {result.rows.rows.map((r, i) => (
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

              {result.path && result.path.length > 0 && (
                <div className="card">
                  <div className="card-header">
                    <h3>
                      How the graph was walked
                      <FieldHelp text="Each hop is an assertion the traversal was willing to trust. Expansion follows only edges that clear the trust floor, so the route itself is defensible." />
                    </h3>
                  </div>
                  <div className="path-chain">
                    {result.path.map((h, i) => (
                      <div key={h.assertion_id + i} className="path-hop" style={epiStyle(h.epistemic_class)}>
                        <span className="path-hop-index">{i + 1}</span>
                        <EpistemicBadge
                          epistemicClass={h.epistemic_class}
                          size="sm"
                          showLabel={false}
                        />
                        <span>
                          <strong>{h.subject_label}</strong>{' '}
                          <span className="prov-pred">{h.predicate}</span>{' '}
                          <strong>{h.object_label}</strong>
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

              {result.citations.length > 0 && (
                <div className="card">
                  <div className="card-header">
                    <h3>
                      Citations
                      <FieldHelp text={HELP.sourceLocator} />
                    </h3>
                  </div>
                  {result.citations.map((c, i) => (
                    <div className="citation" key={c.assertion_id}>
                      <span className="citation-num">[{i + 1}]</span>
                      <div className="citation-body">
                        <div
                          style={{
                            display: 'flex',
                            gap: 9,
                            alignItems: 'center',
                            flexWrap: 'wrap',
                          }}
                        >
                          <strong>{c.label}</strong>
                          <EpistemicBadge
                            epistemicClass={c.epistemic_class}
                            size="sm"
                            tipPlacement="above"
                          />
                          <ConfidenceBar value={c.confidence} floor={floor} width={54} />
                        </div>
                        {c.quote && <div className="citation-quote">{c.quote}</div>}
                        <div className="citation-loc">
                          {c.document_id
                            ? `${c.filename ?? c.document_id} · page ${c.page}`
                            : 'structured source'}
                        </div>
                      </div>
                      {c.document_id && c.page != null && (
                        <button
                          className="btn btn-ghost btn-sm"
                          onClick={() =>
                            setOpenDocument({
                              documentId: c.document_id as string,
                              filename: c.filename || (c.document_id as string),
                              page: c.page as number,
                              quote: c.quote ?? null,
                            })
                          }
                        >
                          Open at page {c.page}
                        </button>
                      )}
                      <button
                        className="btn btn-ghost btn-sm"
                        onClick={() => setOpenProvenance(c.assertion_id)}
                      >
                        Why?
                      </button>
                    </div>
                  ))}
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
