/**
 * The passages an answer quoted and the relationships it walked.
 *
 * Lifted out of `QueryBuilder` so Retrieval renders evidence the same way Ask does. Two pages
 * drawing a citation differently would be two claims about what a citation is, and the whole
 * argument of this product is that a fact looks the same wherever you meet it.
 *
 * Both panels return null when empty, so a lane that found nothing contributes no card rather
 * than an empty one explaining itself.
 */

import { Link } from 'react-router-dom'

import type { QueryHit, QueryPassage } from '../api'
import { HELP } from '../epistemic'
import { entityLabel, epiStyle } from '../format'
import ConfidenceBar from './ConfidenceBar'
import EpistemicBadge from './EpistemicBadge'
import FieldHelp from './FieldHelp'

/** Why this fact is in the answer: what matched, how far it was walked, and from which file. */
function whyIncluded(hit: QueryHit): string {
  const parts: string[] = []
  if (hit.matched_on?.length) parts.push(`matched on ${hit.matched_on.join(', ')}`)
  if (hit.hops != null) {
    parts.push(hit.hops === 1 ? 'direct from the passage' : `${hit.hops} hops from the passage`)
  }
  if (hit.source?.filename) {
    parts.push(hit.source.filename + (hit.source.page != null ? `, page ${hit.source.page}` : ''))
  }
  return parts.join(' · ')
}

export function PassagesCited({
  passages,
  onOpen,
}: {
  passages: QueryPassage[]
  onOpen: (passage: QueryPassage) => void
}) {
  if (passages.length === 0) return null

  return (
    <div className="card">
      <div className="card-header">
        <h3>
          Passages cited
          <FieldHelp text={HELP.sourceLocator} />
        </h3>
        <span className="card-note">{passages.length}</span>
      </div>
      {passages.map((passage, i) => (
        <div className="citation" key={`${passage.document_id}-${passage.char_start ?? i}`}>
          <span className="citation-num">[{i + 1}]</span>
          <div className="citation-body">
            {passage.text && <div className="citation-quote">{passage.text}</div>}
            <div className="citation-loc">
              {passage.filename ?? passage.document_id}
              {passage.page != null ? ` · page ${passage.page}` : ''}
            </div>
          </div>
          {passage.page != null && (
            <button className="btn btn-ghost btn-sm" onClick={() => onOpen(passage)}>
              Open at page {passage.page}
            </button>
          )}
        </div>
      ))}
    </div>
  )
}

export function FactsUsed({
  facts,
  floor,
  onExplain,
}: {
  facts: QueryHit[]
  floor: number
  /** Opens the proof tree for one fact. The "Why?" that makes a row more than an assertion. */
  onExplain: (assertionId: string) => void
}) {
  if (facts.length === 0) return null

  const ids = facts.map((f) => f.assertion_id).filter(Boolean)

  return (
    <div className="card">
      <div className="card-header">
        <h3>
          Relationships used
          <FieldHelp text="Each row is an assertion the read was willing to trust: it cleared the confidence floor and its review state allows it to be used. The matched terms say why it came back, so a surprising result traces to the word that pulled it in." />
        </h3>
        <div className="card-header-actions">
          <span className="card-note">{facts.length}</span>
          {ids.length > 0 && (
            <Link
              to={`/graph?highlight=${ids.map(encodeURIComponent).join(',')}`}
              className="btn btn-ghost btn-sm"
              title="Open these assertions in the graph, drawn against the surrounding facts"
            >
              See in graph
            </Link>
          )}
        </div>
      </div>
      <div className="path-chain">
        {facts.map((fact) => (
          <div
            key={fact.assertion_id}
            className="path-hop"
            style={epiStyle(fact.epistemic_class)}
          >
            <EpistemicBadge epistemicClass={fact.epistemic_class} size="sm" showLabel={false} />
            <span>
              <strong>{entityLabel(fact.subject_id)}</strong>{' '}
              <span className="prov-pred">{fact.predicate}</span>{' '}
              <strong>{entityLabel(fact.object_id)}</strong>
              <span className="dim" style={{ display: 'block', fontSize: 11.5 }}>
                {whyIncluded(fact)}
              </span>
            </span>
            <ConfidenceBar value={fact.confidence} floor={floor} width={54} />
            <button
              className="btn btn-ghost btn-sm"
              style={{ marginLeft: 'auto' }}
              onClick={() => onExplain(fact.assertion_id)}
            >
              Why?
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}
