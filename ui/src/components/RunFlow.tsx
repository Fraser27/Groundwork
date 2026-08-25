/**
 * RunFlow — the shape of a run, as it happens.
 *
 * The transcript below this says what the agent did, in order, in detail. What it cannot show is
 * the *shape*: how many times it searched, where it doubled back, which call was cut off. That is
 * the thing worth watching, and reconstructing it by reading twelve rows defeats the purpose.
 *
 * SVG rather than canvas. A run is capped at twelve turns, so this is tens of nodes with no
 * simulation and no animation loop, which is also why `GraphExplorer`'s canvas machinery is not
 * worth extracting for it. Nodes are real DOM, so they are clickable and readable by a screen
 * reader for free.
 *
 * Colour comes from the same `LANES` map the trace uses, so a node and the lane it stands for
 * cannot disagree about what they are.
 */

import type { RetrievalEvent, ResultKind } from '../api'

/** Which lane colour stands for each tool result, so the diagram matches the trace below it. */
const KIND_COLOUR: Record<ResultKind, string> = {
  composed: 'var(--purple)',
  resolution: 'var(--green)',
  assertions: 'var(--epi-declared)',
  provenance: 'var(--epi-declared)',
  graph: 'var(--epi-declared)',
  metrics: 'var(--green)',
  ontology: 'var(--teal)',
  json: 'var(--text-dim)',
}

const NODE_W = 118
const NODE_H = 46
const GAP = 26
const PAD = 14
const ROW_Y = 30

type FlowNode = {
  seq: number
  label: string
  sub: string
  colour: string
  state: 'ok' | 'error' | 'cancelled' | 'pending' | 'terminal'
}

/**
 * One node per tool call, plus the question and the outcome.
 *
 * Built from the same `events` array the transcript holds, so it grows as the run streams with
 * no separate state to keep in step.
 */
function nodesFrom(events: RetrievalEvent[]): FlowNode[] {
  const nodes: FlowNode[] = []
  const started = events.find((e) => e.kind === 'run_started')
  if (started) {
    nodes.push({
      seq: started.seq,
      label: 'Question',
      sub: started.question ?? '',
      colour: 'var(--text-dim)',
      state: 'terminal',
    })
  }

  for (const event of events) {
    if (event.kind !== 'tool_call') continue
    // The result arrives as a separate event, so a call with none yet is still in flight.
    const result = events.find(
      (e) => e.kind === 'tool_result' && e.turn === event.turn && e.tool === event.tool,
    )
    const state: FlowNode['state'] = event.cancelled
      ? 'cancelled'
      : result?.is_error
        ? 'error'
        : result
          ? 'ok'
          : 'pending'
    nodes.push({
      seq: event.seq,
      label: event.tool ?? 'tool',
      sub: event.cancelled
        ? event.cancelled.replace('_', ' ')
        : (result?.result_kind ?? 'running'),
      colour: result?.result_kind ? KIND_COLOUR[result.result_kind] : 'var(--text-dim)',
      state,
    })
  }

  const finished = events.find((e) => e.kind === 'run_finished' || e.kind === 'run_failed')
  if (finished) {
    nodes.push({
      seq: finished.seq,
      label: finished.kind === 'run_failed' ? 'Failed' : 'Answer',
      sub: finished.stop_reason ?? '',
      colour: finished.was_capped || finished.kind === 'run_failed' ? 'var(--red)' : 'var(--green)',
      state: 'terminal',
    })
  }
  return nodes
}

export default function RunFlow({
  events,
  onSelect,
}: {
  events: RetrievalEvent[]
  /** Clicking a node opens that turn below, so the diagram is navigation rather than decoration. */
  onSelect?: (seq: number) => void
}) {
  const nodes = nodesFrom(events)
  if (nodes.length < 2) return null

  const width = PAD * 2 + nodes.length * NODE_W + (nodes.length - 1) * GAP
  const height = ROW_Y + NODE_H + 34

  return (
    <div className="card runflow-card">
      <div className="card-header">
        <h3>How it was answered</h3>
        <span className="card-note">{nodes.length - 2} tool calls</span>
      </div>
      <div className="runflow-scroll">
        <svg width={width} height={height} role="img" aria-label="Agent run flow">
          {nodes.slice(0, -1).map((node, i) => {
            const x = PAD + (i + 1) * NODE_W + i * GAP
            return (
              <line
                key={`edge-${node.seq}`}
                x1={x}
                y1={ROW_Y + NODE_H / 2}
                x2={x + GAP}
                y2={ROW_Y + NODE_H / 2}
                stroke="var(--border)"
                strokeWidth={1.5}
              />
            )
          })}

          {nodes.map((node, i) => {
            const x = PAD + i * (NODE_W + GAP)
            const clickable = node.state !== 'terminal' && !!onSelect
            return (
              <g
                key={node.seq}
                transform={`translate(${x}, ${ROW_Y})`}
                onClick={clickable ? () => onSelect?.(node.seq) : undefined}
                style={clickable ? { cursor: 'pointer' } : undefined}
              >
                <rect
                  width={NODE_W}
                  height={NODE_H}
                  rx={6}
                  fill="var(--bg-card)"
                  stroke={node.colour}
                  strokeWidth={node.state === 'terminal' ? 1.5 : 2}
                  // Dashed for a call that never ran, so a cap is visible as a gap in the work
                  // rather than as one more completed step.
                  strokeDasharray={node.state === 'cancelled' ? '4 3' : undefined}
                />
                <text x={9} y={19} className="runflow-label">
                  {node.label.length > 15 ? `${node.label.slice(0, 14)}…` : node.label}
                </text>
                <text x={9} y={34} className="runflow-sub">
                  {node.sub.length > 17 ? `${node.sub.slice(0, 16)}…` : node.sub}
                </text>
                {node.state === 'error' && (
                  <circle cx={NODE_W - 9} cy={10} r={3.5} fill="var(--red)" />
                )}
                {node.state === 'pending' && (
                  <circle cx={NODE_W - 9} cy={10} r={3.5} fill="var(--amber, var(--orange))" />
                )}
              </g>
            )
          })}
        </svg>
      </div>
    </div>
  )
}
