/**
 * Retrieval — the agent, with its working shown.
 *
 * Ask answers a question. This shows what an agent did to answer it: every tool it called,
 * every raw result it got back, and for a `compose` call the full lane trace. The transcript
 * is the feature, not a diagnostic behind a toggle, because the reason this exists is to judge
 * whether these tools are ready to hand to somebody else.
 *
 * The prose at the end is the agent's own and is not governed. It is shown last and labelled,
 * so it never reads as the system's finding.
 */

import { useEffect, useRef, useState } from 'react'

import { api, type QueryPassage, type RetrievalEvent, type RetrievalTool } from '../api'
import { getTenantId } from '../auth'
import DocumentViewer from '../components/DocumentViewer'
import { FactsUsed, PassagesCited } from '../components/EvidencePanels'
import FieldHelp from '../components/FieldHelp'
import Markdown from '../components/Markdown'
import ProvenancePanel from '../components/ProvenancePanel'
import QueryTrace from '../components/QueryTrace'
import RunFlow from '../components/RunFlow'
import { EmptyState, ErrorState, Spinner } from '../components/Shared'
import TraceDialog from '../components/TraceDialog'
import { type TraceView, traceOf } from '../trace'
import { useProvenance } from '../useProvenance'
import { useExampleQuestions } from '../useUnitLabel'

/** Turns that carry a full governance trace, so the row says so before it is opened. */
function kindLabel(event: RetrievalEvent): string {
  if (event.cancelled) return 'stopped'
  if (event.is_error) return 'failed'
  return event.result_kind ?? ''
}

/**
 * The two tools that search, and the option to let the agent pick.
 *
 * Offered because the difference is a governance one rather than a performance one, and it is the
 * reader's call: `ask` commits to a single tier and reports one confidence for it, `compose` runs
 * every permitted lane and refuses to average them. Picking one withholds the other from the
 * agent, so the run cannot quietly answer through the tool that was not chosen.
 */
const TOOLS: { value: RetrievalTool; label: string; help: string }[] = [
  {
    value: 'auto',
    label: 'Auto',
    help: 'The agent chooses. It is told to call ask when the question names a governed metric it has already listed, and compose otherwise.',
  },
  {
    value: 'ask',
    label: 'Ask',
    help: 'One tier only: the first that can answer, with one confidence for it. Exact when a governed metric matches, and narrow when none does, because nothing else is searched.',
  },
  {
    value: 'compose',
    label: 'Compose',
    help: 'Every lane the tenant permits, kept apart rather than merged: a compiled metric, quoted passages and graph facts are different kinds of claim and one confidence over them would be invented. Slower, and it reports which lanes did not run.',
  },
]

export default function Retrieval() {
  const tenant = getTenantId()
  // Declared by the tenant's ontology pack, so a retail deployment is not invited to ask about a
  // conflict of interest. Empty for a pack that declares none, and nothing is then offered.
  const examples = useExampleQuestions()
  const [question, setQuestion] = useState('')
  const [tool, setTool] = useState<RetrievalTool>('auto')
  const [events, setEvents] = useState<RetrievalEvent[]>([])
  const [running, setRunning] = useState(false)
  const [error, setError] = useState('')
  const [expanded, setExpanded] = useState<Set<number>>(new Set())
  const [raw, setRaw] = useState<Set<number>>(new Set())
  const [selected, setSelected] = useState<string | null>(null)
  const [passage, setPassage] = useState<QueryPassage | null>(null)
  const [fullTrace, setFullTrace] = useState<{ trace: TraceView; tool: string } | null>(null)
  const stop = useRef<(() => void) | null>(null)

  // Closing the socket stops the run, so leaving the page must not leave one going.
  useEffect(() => () => stop.current?.(), [])

  const { provenance } = useProvenance(tenant, selected)

  const run = () => {
    const q = question.trim()
    if (!q || running) return
    stop.current?.()
    setEvents([])
    setExpanded(new Set())
    setRaw(new Set())
    setError('')
    setRunning(true)

    stop.current = api.runRetrieval(
      tenant,
      q,
      (event) => {
        setEvents((prev) => [...prev, event])
        if (event.kind === 'run_finished' || event.kind === 'run_failed') setRunning(false)
      },
      (detail) => {
        setError(detail)
        setRunning(false)
      },
      tool,
    )
  }

  const toggle = (set: Set<number>, seq: number, apply: (s: Set<number>) => void) => {
    const next = new Set(set)
    if (next.has(seq)) next.delete(seq)
    else next.add(seq)
    apply(next)
  }

  const started = events.find((e) => e.kind === 'run_started')
  const finished = events.find((e) => e.kind === 'run_finished' || e.kind === 'run_failed')
  const turns = events.filter((e) => e.kind === 'tool_call')

  return (
    <div className="page">
      <div className="page-header">
        <div className="page-header-row">
          <div>
            <h2>
              Retrieval
              <FieldHelp text="An agent answers by calling this system's own tools, and every call it makes is listed below with the raw result it got back. The point is to judge the tools before they are exposed to anything outside this platform, so nothing here is summarised away. The prose at the end is the agent's own writing and is not governed." />
            </h2>
            <p>
              An agent answers by calling the same tools an outside client would get. Every call
              and every raw result is below.
            </p>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="retrieval-composer">
          <div className="form-group">
            <label>Question</label>
            <input
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) run()
              }}
              placeholder={examples[0] ?? 'Ask a question about your data'}
              disabled={running}
            />
            <p className="hint">Cmd/Ctrl + Enter to run.</p>
          </div>
          <div className="form-group">
            <label>
              Search with
              <FieldHelp text="Which of the two searching tools the agent may use. Ask and compose are the only tools that consult the metrics, the graph, the documents and the catalog; everything else the agent calls either describes the vocabulary or follows up on something these returned. Choosing one here withholds the other, so the run cannot answer through a tool you did not pick." />
            </label>
            <div className="segmented" role="group" aria-label="Search tool">
              {TOOLS.map((t) => (
                <button
                  key={t.value}
                  type="button"
                  className={`segmented-option${tool === t.value ? ' is-active' : ''}`}
                  aria-pressed={tool === t.value}
                  title={t.help}
                  disabled={running}
                  onClick={() => setTool(t.value)}
                >
                  {t.label}
                </button>
              ))}
            </div>
          </div>
          <button className="btn btn-primary" onClick={run} disabled={running || !question.trim()}>
            {running ? 'Running…' : 'Run agent'}
          </button>
          {running && (
            <button
              className="btn btn-ghost"
              onClick={() => {
                stop.current?.()
                setRunning(false)
              }}
            >
              Stop
            </button>
          )}
        </div>
        {events.length === 0 && !running && examples.length > 0 && (
          <div className="hint" style={{ marginTop: 4 }}>
            Try one of these:{' '}
            {examples.map((e, i) => (
              <span key={e}>
                {i > 0 && ' · '}
                <button className="link-button" onClick={() => setQuestion(e)}>
                  {e}
                </button>
              </span>
            ))}
          </div>
        )}
      </div>

      {error && <ErrorState title="The agent could not run" detail={error} onRetry={run} />}

      <div className={`retrieval-layout${provenance ? '' : ' retrieval-layout-solo'}`}>
        <div className="retrieval-transcript">
          {events.length === 0 && !running && !error && (
            <EmptyState title="No run yet">
              Ask something. Each tool call appears here as it happens, with the raw result it
              returned.
            </EmptyState>
          )}

          <RunFlow
            events={events}
            onSelect={(seq) => {
              // The node stands for a `tool_call`; the row that expands is keyed on it, so
              // selecting a node opens the turn it names rather than scrolling near it.
              setExpanded((prev) => new Set(prev).add(seq))
              document
                .querySelector(`[data-turn-seq="${seq}"]`)
                ?.scrollIntoView({ behavior: 'smooth', block: 'center' })
            }}
          />

          {started && (
            <div className="card">
              <div className="card-header">
                <h3>Run</h3>
                <span className="card-note">
                  {turns.length} of {started.max_turns ?? 12} turns
                  {finished?.stop_reason ? ` · ${finished.stop_reason}` : ''}
                </span>
              </div>
              <p className="hint" style={{ margin: 0 }}>
                <code>{started.model_id}</code> answering: {started.question}
              </p>
              {finished?.was_capped && (
                <div className="banner banner-warn" style={{ marginTop: 10, marginBottom: 0 }}>
                  <span>
                    <strong>This run was stopped early ({finished.stop_reason}).</strong> What is
                    below is what it had gathered, not a finished answer.
                  </span>
                </div>
              )}
            </div>
          )}

          {events
            .filter((e) => e.kind === 'tool_call' || e.kind === 'tool_result')
            .map((event) => {
              const open = expanded.has(event.seq)
              const showRaw = raw.has(event.seq)
              const trace = traceOf(event.result_kind, event.result)

              return (
                <div
                  key={event.seq}
                  data-turn-seq={event.seq}
                  className={`retrieval-turn${event.is_error ? ' retrieval-turn-error' : ''}`}
                >
                  <button
                    className="retrieval-turn-head"
                    onClick={() => toggle(expanded, event.seq, setExpanded)}
                  >
                    <span className="retrieval-turn-index">{event.turn}</span>
                    <code className="retrieval-tool">{event.tool}</code>
                    <span className="retrieval-args">
                      {event.kind === 'tool_call'
                        ? JSON.stringify(event.arguments ?? {})
                        : kindLabel(event)}
                    </span>
                    <span className="retrieval-turn-state">
                      {event.kind === 'tool_call' ? 'called' : kindLabel(event)}
                    </span>
                  </button>

                  {open && (
                    <div className="retrieval-turn-body">
                      {event.cancelled && (
                        <p className="hint">
                          Not run: this run reached its {event.cancelled.replace('_', ' ')}.
                        </p>
                      )}
                      {event.error && <p className="hint">{event.error}</p>}

                      {trace && !showRaw && (
                        <>
                          <div className="retrieval-turn-actions">
                            <button
                              className="btn btn-ghost btn-sm"
                              onClick={() => setFullTrace({ trace, tool: event.tool ?? '' })}
                            >
                              Open full trace
                            </button>
                          </div>
                          <QueryTrace
                            router={trace.router}
                            gate={trace.gate}
                            lanes={trace.lanes}
                            blocks={trace.blocks}
                            floor={trace.floor}
                            warnings={trace.warnings}
                            onOpenPassage={setPassage}
                          />
                          {/* The same panels Ask renders. A citation drawn two ways would be two
                              claims about what a citation is. */}
                          <PassagesCited passages={trace.passages} onOpen={setPassage} />
                          <FactsUsed
                            facts={trace.facts}
                            floor={trace.floor}
                            onExplain={setSelected}
                          />
                        </>
                      )}

                      {!trace && !showRaw && event.kind === 'tool_result' && (
                        <pre className="code-block">{JSON.stringify(event.result, null, 2)}</pre>
                      )}

                      {showRaw && (
                        <pre className="code-block">{JSON.stringify(event, null, 2)}</pre>
                      )}

                      {event.kind === 'tool_result' && (
                        <button
                          className="btn btn-ghost btn-sm"
                          onClick={() => toggle(raw, event.seq, setRaw)}
                        >
                          {showRaw ? 'Rendered' : 'Raw'}
                        </button>
                      )}
                    </div>
                  )}
                </div>
              )
            })}

          {running && <Spinner />}

          {finished?.answer && (
            <div className="card">
              <div className="card-header">
                <h3>
                  Answer
                  <FieldHelp text="Written by the agent over the evidence above. Not governed: no approved definition produced it and no rule checked it, so read it against the tool results rather than instead of them." />
                </h3>
                <span className="card-note">
                  {finished.searched === false
                    ? 'nothing was searched'
                    : "the agent's own words, not governed"}
                </span>
              </div>
              {/* Above the prose, because a reader who has already read it has been misled. The
                  agent's own confident sentences are exactly what this contradicts. */}
              {(finished.warnings ?? []).map((w) => (
                <div key={w} className="banner banner-warn" style={{ margin: '0 0 12px' }}>
                  <span>{w}</span>
                </div>
              ))}
              <Markdown text={finished.answer} />
            </div>
          )}
        </div>

        {provenance && (
          <div className="retrieval-rail">
            <ProvenancePanel
              provenance={provenance}
              onClose={() => setSelected(null)}
              onSelectAssertion={setSelected}
              compact
            />
          </div>
        )}
      </div>

      {fullTrace && (
        <TraceDialog
          trace={fullTrace.trace}
          tool={fullTrace.tool}
          onClose={() => setFullTrace(null)}
          onOpenPassage={setPassage}
          onExplain={setSelected}
        />
      )}

      {passage && (
        <DocumentViewer
          tenant={tenant}
          documentId={passage.document_id}
          filename={passage.filename || passage.document_id}
          page={passage.page ?? 1}
          quote={passage.text}
          onClose={() => setPassage(null)}
        />
      )}
    </div>
  )
}
