/**
 * EvidenceFlow — which document produced which fact, and which fact triggered the wall.
 *
 * The lane cards say what each lane returned. What they cannot show is how the four columns
 * connect: that these thirty facts came out of those four passages, which came out of three
 * documents, and that one of them is what the ethical wall fired on. That chain is the answer to
 * "why does the system believe this", drawn instead of read.
 *
 * Both joins are already in the payload, so nothing here is inferred:
 *   passage -> fact    `fact.source.document_id === passage.document_id`
 *   fact -> finding    `block.subject === fact.subject_id || fact.object_id`
 *
 * Two things measured on real data shape the layout, and both are easy to get wrong.
 *
 * **One graph column, not a tree.** Every fact comes back with `hops: 1`, so there is no depth to
 * draw and a tiered layout would imply expansion that did not happen.
 *
 * **A fact with no document is the point, not a gap.** The reasoner's conclusions carry
 * `source.source_id: 'reasoner'` and no `document_id`, because they were never in a document. That
 * is the derived fact the whole conflict rests on, so it is drawn in place and marked, never
 * dropped for failing to join.
 *
 * SVG for the same reason as `RunFlow`: tens of nodes, no simulation, no animation loop.
 */

import type { QueryBlock, QueryHit, QueryPassage } from '../api'
import { entityLabel } from '../format'

const COL_W = 176
const COL_GAP = 74
const NODE_H = 30
const NODE_GAP = 7
const PAD_X = 14
const HEAD_Y = 22
const TOP = 44

/** How many nodes a column draws before collapsing the rest into a count. */
const MAX_ROWS = 9

type Node = {
  id: string
  label: string
  sub: string
  colour: string
  /** Drawn dashed: this fact was derived, so it has no document behind it. */
  derived?: boolean
  onClick?: () => void
}

type Column = {
  title: string
  nodes: Node[]
  hidden: number
}

/** `document_id` off a hit, which lives on `source` rather than at the top level. */
function docOf(fact: QueryHit): string {
  return fact.source?.document_id ?? ''
}

function shorten(text: string, max: number): string {
  const clean = text.trim()
  return clean.length > max ? `${clean.slice(0, max - 1)}…` : clean
}

export default function EvidenceFlow({
  passages,
  facts,
  blocks,
  onOpenPassage,
  onExplain,
}: {
  passages: QueryPassage[]
  facts: QueryHit[]
  blocks: QueryBlock[]
  onOpenPassage?: (passage: QueryPassage) => void
  /** Opens the proof tree. A finding reports how many premises it has; this is how they are read. */
  onExplain?: (assertionId: string) => void
}) {
  if (passages.length === 0 && facts.length === 0) return null

  const documents = new Map<string, { filename: string; passages: number }>()
  for (const passage of passages) {
    const existing = documents.get(passage.document_id)
    documents.set(passage.document_id, {
      filename: passage.filename ?? passage.document_id,
      passages: (existing?.passages ?? 0) + 1,
    })
  }

  const docNodes: Node[] = [...documents.entries()].map(([id, doc]) => ({
    id,
    label: shorten(doc.filename, 22),
    sub: `${doc.passages} ${doc.passages === 1 ? 'passage' : 'passages'}`,
    colour: 'var(--epi-declared)',
  }))

  const passageNodes: Node[] = passages.map((passage, i) => ({
    id: `${passage.document_id}-${passage.char_start ?? i}`,
    label: passage.page != null ? `page ${passage.page}` : `passage ${i + 1}`,
    sub: shorten(passage.text ?? passage.filename ?? '', 24),
    colour: 'var(--purple)',
    onClick: onOpenPassage ? () => onOpenPassage(passage) : undefined,
  }))

  // Derived facts first: a conclusion nothing quoted is the least likely to be already known, and
  // it is the one a reader should meet before the thirty facts that merely restate a document.
  const ordered = [...facts].sort((a, b) => Number(!docOf(a)) - Number(!docOf(b)) || 0)
  const factNodes: Node[] = ordered.map((fact) => ({
    id: fact.assertion_id,
    label: shorten(fact.predicate, 22),
    sub: shorten(`${entityLabel(fact.subject_id)} to ${entityLabel(fact.object_id)}`, 26),
    colour: docOf(fact) ? 'var(--blue, var(--purple))' : 'var(--red)',
    derived: !docOf(fact),
    onClick: onExplain ? () => onExplain(fact.assertion_id) : undefined,
  }))

  const findingNodes: Node[] = blocks.map((block, i) => ({
    id: `${block.subject}-${i}`,
    label: shorten(block.rule, 22),
    sub:
      block.premise_count && block.premise_count > 0
        ? `${block.premise_count} premises`
        : 'screen',
    colour: (block.effect ?? 'withhold') === 'withhold' ? 'var(--red)' : 'var(--orange)',
  }))

  const columns: Column[] = [
    { title: 'Documents', nodes: docNodes, hidden: 0 },
    { title: 'Passages', nodes: passageNodes, hidden: 0 },
    { title: 'Facts', nodes: factNodes, hidden: 0 },
    { title: 'Findings', nodes: findingNodes, hidden: 0 },
  ]
    // A column with nothing in it is dropped rather than drawn empty: a tier 1 answer quotes no
    // passages, and an empty Passages column would read as a search that found nothing.
    .filter((column) => column.nodes.length > 0)
    .map((column) => ({
      ...column,
      hidden: Math.max(0, column.nodes.length - MAX_ROWS),
      nodes: column.nodes.slice(0, MAX_ROWS),
    }))

  const rows = Math.max(...columns.map((c) => c.nodes.length))
  const width = PAD_X * 2 + columns.length * COL_W + (columns.length - 1) * COL_GAP
  const height = TOP + rows * (NODE_H + NODE_GAP) + 24

  const xOf = (col: number) => PAD_X + col * (COL_W + COL_GAP)
  const yOf = (row: number) => TOP + row * (NODE_H + NODE_GAP)

  return (
    <div className="card runflow-card">
      <div className="card-header">
        <h3>How the evidence connects</h3>
        <span className="card-note">
          left to right, in the order the lanes actually ran
        </span>
      </div>
      <div className="runflow-scroll">
        <svg width={width} height={height} role="img" aria-label="Evidence flow">
          {/* Edges are drawn column to column rather than node to node. Thirty facts against four
              passages is 120 possible lines, which is a grey wash rather than a diagram. The join
              is exact and reported per node in the panels below; here it is the shape that
              matters. */}
          {columns.slice(0, -1).map((column, col) => {
            const rowsHere = Math.max(column.nodes.length, 1)
            const rowsNext = Math.max(columns[col + 1].nodes.length, 1)
            return (
              <line
                key={`edge-${column.title}`}
                x1={xOf(col) + COL_W}
                y1={yOf((rowsHere - 1) / 2) + NODE_H / 2}
                x2={xOf(col + 1)}
                y2={yOf((rowsNext - 1) / 2) + NODE_H / 2}
                stroke="var(--border)"
                strokeWidth={1.5}
              />
            )
          })}

          {columns.map((column, col) => (
            <g key={column.title}>
              <text x={xOf(col)} y={HEAD_Y} className="runflow-label">
                {column.title.toUpperCase()} · {column.nodes.length + column.hidden}
              </text>
              {column.nodes.map((node, row) => (
                <g
                  key={node.id}
                  transform={`translate(${xOf(col)}, ${yOf(row)})`}
                  onClick={node.onClick}
                  style={node.onClick ? { cursor: 'pointer' } : undefined}
                >
                  <rect
                    width={COL_W}
                    height={NODE_H}
                    rx={5}
                    fill="var(--bg-card)"
                    stroke={node.colour}
                    strokeWidth={1.5}
                    strokeDasharray={node.derived ? '4 3' : undefined}
                  />
                  <text x={8} y={13} className="runflow-label">
                    {node.label}
                  </text>
                  <text x={8} y={24} className="runflow-sub">
                    {node.sub}
                  </text>
                </g>
              ))}
              {column.hidden > 0 && (
                <text
                  x={xOf(col)}
                  y={yOf(column.nodes.length) + 12}
                  className="runflow-sub"
                >
                  and {column.hidden} more
                </text>
              )}
            </g>
          ))}
        </svg>
      </div>
      {factNodes.some((n) => n.derived) && (
        <p className="hint">
          A dashed fact was derived by the reasoner rather than quoted from a document, so it has
          nothing above it in this diagram. Open it to see the premises it was built from.
        </p>
      )}
    </div>
  )
}
