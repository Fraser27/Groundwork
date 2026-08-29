/**
 * Multi-hop connections, drawn as chains rather than as rows.
 *
 * The row list beside this one is not wrong, it is just not readable for this question. "Who helped
 * Sam Parker" is answered by three edges that share two intermediate entities, and as three rows
 * among two hundred the join is left to whoever is reading — a person scanning ids that differ by
 * four characters, or a model guessing. Drawn as a chain the join is the graph's, and every hop
 * still carries the `assertion_id` that makes it checkable.
 *
 * Direction is read off `reversed`, not off subject and object. A chain may traverse an edge against
 * the way it was written, and printing `A REPRESENTS B` for a chain that arrived at A from B would
 * state the relationship backwards — a rendering bug that reads as a false fact.
 */

import type { QueryPath } from '../api'
import { entityLabel } from '../format'
import ConfidenceBar from './ConfidenceBar'
import FieldHelp from './FieldHelp'

export const PATH_HELP =
  'A connection the graph assembled itself: each arrow is one verified relationship, and the ' +
  'chain is what they add up to. This is where a link through an intermediary shows up, which a ' +
  'flat list of relationships cannot show you. Strength is the weakest hop, because a chain is ' +
  'only as good as its worst step. Nothing here is inferred: every hop is a relationship already ' +
  'in the list below, joined up without a model.'

function Chain({ path, floor }: { path: QueryPath; floor: number }) {
  const hops = path.steps.length

  return (
    <div className="path-chain-row">
      <div className="path-chain-meta">
        <span className="path-chain-hops">
          {hops} hop{hops === 1 ? '' : 's'}
        </span>
        <ConfidenceBar value={path.confidence} floor={floor} width={54} />
      </div>
      <div className="path-chain-flow">
        <span className="path-chain-node">{entityLabel(path.nodes[0] ?? '')}</span>
        {path.steps.map((step, i) => (
          <span className="path-chain-step" key={`${step.assertion_id}-${i}`}>
            {/* The id in the tooltip, so a reviewer reading a chain can take a hop to provenance
                without going back to the row list to find which edge it was. */}
            <span className="path-chain-edge" title={step.assertion_id}>
              {step.reversed ? '←' : ''}
              {step.predicate}
              {step.reversed ? '' : '→'}
            </span>
            <span className="path-chain-node">{entityLabel(path.nodes[i + 1] ?? '')}</span>
          </span>
        ))}
      </div>
    </div>
  )
}

/** The chains themselves, with no card around them. For a caller that already has a heading. */
export function PathChainList({ paths, floor }: { paths: QueryPath[]; floor: number }) {
  if (paths.length === 0) return null
  return (
    <div className="path-chains">
      {paths.map((path) => (
        <Chain key={path.steps.map((s) => s.assertion_id).join('-')} path={path} floor={floor} />
      ))}
    </div>
  )
}

/** A card of its own, for the evidence column. Null when nothing chained up, like the panels
 *  beside it: an empty card explaining itself is worse than no card. */
export default function PathChains({ paths, floor }: { paths: QueryPath[]; floor: number }) {
  if (paths.length === 0) return null

  return (
    <div className="card">
      <div className="card-header">
        <h3>
          Connections found
          <FieldHelp text={PATH_HELP} />
        </h3>
        <span className="card-note">{paths.length}</span>
      </div>
      <PathChainList paths={paths} floor={floor} />
    </div>
  )
}
