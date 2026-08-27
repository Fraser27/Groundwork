/**
 * QueryBuilder — ask a question, and see which part of the system answered it.
 *
 * @deprecated Superseded by Retrieval, and removed from the menu. Reachable only by an existing
 * bookmark. Retrieval renders the same trace, the same evidence panels and the same wall over an
 * agent loop, and its turns show a tool ladder this page cannot.
 *
 * Not deleted, for one reason worth keeping in mind before anybody does delete it: this page is
 * where `EvidencePanels` and the trace idiom came from, and `lanesFromResult` is still the only
 * reader of the single-tier answer shape, which Retrieval now depends on for `ask` turns.
 *
 * The tier is not a diagnostic detail: it tells the reader how much of the answer
 * was generated. Tier 1 is a compiled metric with no model involved; tier 3 leans
 * on similarity to decide what to read. Both are legitimate, and a reader is
 * entitled to know which one they got before relying on the number.
 */

import { useEffect, useState, type CSSProperties } from 'react'
import { Link } from 'react-router-dom'
import {
  api,
  type ComposedResult,
  type Matter,
  type QueryHit,
  type QueryResult,
  type TenantSettings,
} from '../api'
import { getTenantId } from '../auth'
import { HELP, TIERS, tierMeta } from '../epistemic'
import {
  asGenerated,
  asHits,
  asPassages,
  asRows,
  lanesFromComposed,
  lanesFromResult,
} from '../trace'
import { useProvenance } from '../useProvenance'
import ConfidenceBar from '../components/ConfidenceBar'
import DocumentViewer from '../components/DocumentViewer'
import EpistemicBadge from '../components/EpistemicBadge'
import FieldHelp from '../components/FieldHelp'
import ProvenancePanel from '../components/ProvenancePanel'
import QueryTrace from '../components/QueryTrace'
import { ErrorState, Spinner } from '../components/Shared'
import { entityLabel, epiStyle } from '../format'
import { useExampleQuestions, useUnitLabel } from '../useUnitLabel'

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

export default function QueryBuilder() {
  const tenant = getTenantId()
  const unit = useUnitLabel()
  const examples = useExampleQuestions()
  // The pack's first question, so the empty field suggests something this data can actually
  // answer. Falls back to a shape rather than a subject, which is true of any pack.
  const placeholder = examples[0] ?? `Which ${unit.lowerPlural} does this apply to?`
  const [question, setQuestion] = useState('')
  const [matterId, setMatterId] = useState('')
  const [asOf, setAsOf] = useState('')
  const [matters, setMatters] = useState<Matter[]>([])
  const [settings, setSettings] = useState<TenantSettings | null>(null)
  const [floorOverride, setFloorOverride] = useState<number | null>(null)
  const [result, setResult] = useState<QueryResult | null>(null)
  const [composed, setComposed] = useState<ComposedResult | null>(null)
  /** Run every lane instead of stopping at the first tier that can answer. */
  const [everyLane, setEveryLane] = useState(false)
  /**
   * Off by default: the parts and their citations are the reviewable answer, and a summary is a
   * model writing over evidence that already carries its own provenance.
   */
  const [summarise, setSummarise] = useState(false)
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
  /**
   * The floor the server says it used, falling back to the requested one before an answer exists.
   *
   * Results are described with this and never with `floor`. A request may only *raise* the
   * tenant's floor, so a slider set below it is ignored -- and the page used to report "nothing
   * cleared the trust floor of 0.85" from its own state while the field was being dropped in
   * transit, naming a number no read had ever applied.
   */
  const appliedFloor = result?.min_confidence ?? composed?.min_confidence ?? floor

  const ask = async (q: string) => {
    setRunning(true)
    setError('')
    setResult(null)
    setComposed(null)
    setOpenProvenance(null)
    try {
      if (everyLane) {
        setComposed(
          await api.compose(tenant, {
            question: q,
            synthesise: summarise,
            min_confidence: floor,
          }),
        )
      } else {
        setResult(
          await api.query(tenant, {
            question: q,
            matter_id: matterId || undefined,
            as_of: asOf || undefined,
            min_confidence: floor,
          }),
        )
      }
    } catch (e) {
      setError((e as Error).message.replace(/^\d+:\s*/, ''))
    } finally {
      setRunning(false)
    }
  }

  const answerTier = result ? tierMeta(result.tier) : null
  const rows = result ? asRows(result.answer) : null
  const hits = result ? asHits(result.answer) : []
  const passages = result ? asPassages(result.answer) : []
  const generated = result ? asGenerated(result.answer) : null
  // `assertions_used` is the recorded audit trail and the thing worth deep-linking; it is
  // empty for tier 1, and the hits are the same ids for 2 and 3, so fall back rather than
  // lose the action if the field ever arrives absent.
  const usedIds = result?.assertions_used?.length
    ? result.assertions_used
    : hits.map((h) => h.assertion_id)
  // `?? []` rather than trusting the declared type: the field is new, and a type is a claim
  // about the response, not a check on one.
  const blocks = result?.blocks ?? []
  const lanes = result ? lanesFromResult(result) : []
  const composedLanes = composed ? lanesFromComposed(composed) : []
  const composedBlocks = composed?.blocks ?? []
  const composedWarnings = composed?.warnings ?? []

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

      {/* Said plainly rather than left implicit: this page is out of the menu, so anyone here
          arrived by a bookmark and would otherwise not know it had been superseded. */}
      <div className="banner banner-warn">
        <span>
          <strong>This page is deprecated.</strong> Retrieval answers the same questions and shows
          the same evidence, with the agent's tool calls alongside it.{' '}
          <Link to="/retrieval">Open Retrieval</Link>.
        </span>
      </div>

      {contextError && (
        <div className="banner banner-warn">
          <span>
            <strong>Could not load the {unit.lower} list or the trust floor.</strong>{' '}
            {contextError}. You can still ask, but the {unit.lower} filter is empty and the default
            floor is in use.
          </span>
        </div>
      )}

      <div className="card">
        <div className="form-group">
          <label>Question</label>
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder={`e.g. ${placeholder}`}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && (e.metaKey || e.ctrlKey) && question.trim()) ask(question)
            }}
          />
          <p className="hint">Cmd/Ctrl + Enter to ask.</p>
        </div>

        <div className="toolbar" style={{ marginBottom: 12 }}>
          <div className="toolbar-field">
            <label>
              {unit.singular}
              <FieldHelp text={HELP.matterWall} />
            </label>
            <select value={matterId} onChange={(e) => setMatterId(e.target.value)}>
              <option value="">All {unit.lowerPlural} I can see</option>
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

        <label className="checkbox-row" style={{ marginBottom: asOf ? 12 : 0 }}>
          <input
            type="checkbox"
            checked={everyLane}
            onChange={(e) => setEveryLane(e.target.checked)}
          />
          <span>
            Search every lane, not just the first that can answer
            <FieldHelp text="Normally the question stops at the lowest tier that can answer it, which is right when a governed metric matches exactly. This runs the graph, the documents and the catalogue as well, and reports each separately rather than merging them, because a compiled figure, a quoted passage and a model's reading are not the same kind of claim. A matching governed metric still short-circuits: it is exact, and fanning out would add latency and nothing else." />
            <span className="dim" style={{ display: 'block', fontSize: 11.5 }}>
              Slower. {unit.singular} and as-at filters do not apply to this route.
            </span>
          </span>
        </label>

        {everyLane && (
          <label className="checkbox-row" style={{ marginLeft: 22, marginTop: 8 }}>
            <input
              type="checkbox"
              checked={summarise}
              onChange={(e) => setSummarise(e.target.checked)}
            />
            <span>
              Summarise the parts in prose
              <FieldHelp text="A model reads the parts below and writes a paragraph over them. It sees only what survived the ethical wall, so it cannot reason about a blocked fact even accidentally, and it is told to cite each part rather than merge them. Leaving this off is the reviewable form: every part is already an answer carrying its own provenance, and prose is the one element of the response no citation stands behind." />
              <span className="dim" style={{ display: 'block', fontSize: 11.5 }}>
                Adds a model to the response. The parts below are unaffected either way.
              </span>
            </span>
          </label>
        )}

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

      {examples.length > 0 && (
        <div className="card">
          <div className="card-header">
            <h3>Try one of these</h3>
            <span className="card-note">
              Declared by this tenant&apos;s ontology pack, alongside the vocabulary they are asked
              in.
            </span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            {examples.map((q) => (
              <button
                key={q}
                className="btn btn-ghost btn-sm"
                style={{ justifyContent: 'flex-start', textAlign: 'left', whiteSpace: 'normal' }}
                onClick={() => {
                  setQuestion(q)
                  ask(q)
                }}
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      )}

      {error && (
        <div className="banner banner-error" style={{ marginTop: 16 }}>
          <span>{error}</span>
        </div>
      )}

      {running && <Spinner />}

      {result && answerTier && (
        <>
            <div
              className="tier-banner"
              style={{ marginTop: 16, '--tier-colour': answerTier.colour } as CSSProperties}
            >
              <div className="tier-num">{result.tier}</div>
              <div className="tier-text">
                <h4>
                  {answerTier.label}
                  <FieldHelp text={HELP.resolutionTier} />
                  <span className={`tag ${result.governed ? 'tag-green' : 'tag-orange'}`}>
                    {result.governed ? 'governed' : 'ungoverned'}
                  </span>
                </h4>
                <p>{result.explanation}</p>
                <p style={{ color: answerTier.colour, fontWeight: 550 }}>{answerTier.llm}</p>
              </div>
            </div>

            {result.warnings.length > 0 && (
              <div className="banner banner-warn" style={{ marginTop: 16 }}>
                <span>{result.warnings.join(' ')}</span>
              </div>
            )}

            {/* Above the answer cards: the route is what the reader needs in order to know how
                much of what follows to trust. Step 4 opens itself when anything was refused. */}
            <div style={{ marginTop: 16 }}>
              <QueryTrace
                router={result.router}
                gate={result.gate}
                lanes={lanes}
                blocks={blocks}
                floor={appliedFloor}
                usedFactCount={usedIds.length}
                onOpenPassage={(p) =>
                  setOpenDocument({
                    documentId: p.document_id,
                    filename: p.filename || p.document_id,
                    page: p.page as number,
                    quote: p.text ?? null,
                  })
                }
              />
            </div>

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
                      <ConfidenceBar value={h.confidence} floor={appliedFloor} width={54} />
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

            {!rows && hits.length === 0 && passages.length === 0 && !generated && (
              <div className="card">
                <div className="card-header">
                  <h3>Answer</h3>
                </div>
                <div className="answer-block">
                  Nothing came back. The question reached {answerTier.label.toLowerCase()}, but no
                  row, fact or passage cleared the trust floor of {appliedFloor.toFixed(2)}.
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
                    Compiled SQL
                    <FieldHelp text="Compiled from the governed metric definition. Deterministic: the same definition always produces this query, with no model involved." />
                  </h3>
                  <span className="tag tag-green">deterministic</span>
                </div>
                <pre className="code-block">{result.sql}</pre>
              </div>
            )}

            {generated && (
              <div className="card" style={{ borderColor: 'var(--orange)' }}>
                <div className="card-header">
                  <h3>
                    AI-written SQL
                    <FieldHelp text="Written by AI for this question, because no approved metric covers it. It could only name the tables listed below, and had to aggregate rather than return individual rows, both enforced on the query itself rather than asked for in the prompt. Nobody approved what it measures, so read it." />
                  </h3>
                  <span className="tag tag-orange">not from an approved metric</span>
                </div>
                <pre className="code-block">{generated.sql}</pre>
                <div className="dim" style={{ fontSize: 11.5, marginTop: 8 }}>
                  Tables it was allowed to read: {generated.tables_offered.join(', ') || 'none'}
                </div>
                {/* The error, never an empty table. Columns are not validated by the firewall, so
                    a query naming one that does not exist fails here, and showing that as no rows
                    would read as "no data" for a question that was never answered. */}
                {generated.error && (
                  <div className="banner banner-error" style={{ marginTop: 10 }}>
                    This query did not run: {generated.error}
                  </div>
                )}
                {generated.rows && generated.rows.rows.length > 0 && (
                  <div style={{ overflowX: 'auto', marginTop: 10 }}>
                    <table className="data-table">
                      <thead>
                        <tr>
                          {generated.rows.columns.map((c) => (
                            <th key={c}>{c}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {generated.rows.rows.map((r, i) => (
                          <tr key={i}>
                            {r.map((cell, j) => (
                              <td key={j}>{cell ?? '-'}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
                {generated.rows && generated.rows.rows.length === 0 && !generated.error && (
                  <p className="qtrace-note" style={{ marginTop: 10 }}>
                    The query ran and matched no rows. That is an empty result from a real query,
                    not a query that failed.
                  </p>
                )}
              </div>
            )}

            <div className="card">
              <div className="card-header">
                <h3>
                  The three routes
                  <FieldHelp text={HELP.resolutionTier} />
                </h3>
              </div>
              <div className="tier-ladder">
                {([1, 2, 3] as const).map((t) => (
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

      {composed && (
        <>
          <div
            className="tier-banner"
            style={{ marginTop: 16, '--tier-colour': 'var(--purple)' } as CSSProperties}
          >
            <div className="tier-num">{(composed.lanes_run ?? []).length}</div>
            <div className="tier-text">
              <h4>
                Composed from {(composed.lanes_run ?? []).length}{' '}
                {(composed.lanes_run ?? []).length === 1 ? 'lane' : 'lanes'}
                <FieldHelp text={HELP.resolutionTier} />
                <span
                  className={`tag ${composed.fully_deterministic ? 'tag-green' : 'tag-orange'}`}
                >
                  {composed.governance}
                </span>
              </h4>
              <p>{composed.note}</p>
            </div>
          </div>

          {composedWarnings.length > 0 && (
            <div className="banner banner-warn" style={{ marginTop: 16 }}>
              <span>{composedWarnings.join(' ')}</span>
            </div>
          )}

          <div style={{ marginTop: 16 }}>
            <QueryTrace
              router={composed.router}
              gate={composed.gate}
              lanes={composedLanes}
              blocks={composedBlocks}
              floor={appliedFloor}
              onOpenPassage={(p) =>
                setOpenDocument({
                  documentId: p.document_id,
                  filename: p.filename || p.document_id,
                  page: p.page as number,
                  quote: p.text ?? null,
                })
              }
            />
          </div>

          {composed.synthesis ? (
            <div className="card">
              <div className="card-header">
                <h3>
                  Answer
                  <FieldHelp text="Written by a model over the evidence that survived the wall, and only over that. It decided none of it: every part it drew on is above, with its own provenance." />
                </h3>
                <span className="tag tag-orange">phrased by a model</span>
              </div>
              <div className="answer-block">{composed.synthesis}</div>
            </div>
          ) : (
            <div className="card">
              <div className="card-header">
                <h3>Answer</h3>
                <span className="tag tag-green">unsummarised</span>
              </div>
              <div className="answer-block">
                No prose was written over these parts, so what is above is the whole answer. That
                is the reviewable form: each lane's contribution is separate and carries its own
                provenance.
              </div>
            </div>
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
                confidenceFloor={appliedFloor}
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
