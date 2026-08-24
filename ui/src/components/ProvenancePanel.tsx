/**
 * ProvenancePanel — the answer to "why does the system believe this?".
 *
 * Three shapes, depending on how the fact came to be:
 *   - read from a document  → the file, the page, and the words quoted from it, with
 *                             a way to open the file at that page
 *   - read from a database  → the source, table and column
 *   - inferred by a rule    → the proof tree, unwound to the facts at its base
 *
 * Everything else on this page hangs off one of those three.
 */

import { useState, type CSSProperties } from 'react'
import type { Assertion, EpistemicClass, PageCitation, Premise, Provenance } from '../api'
import { getTenantId } from '../auth'
import { EPISTEMIC, HELP } from '../epistemic'
import { epiStyle } from '../format'
import DocumentViewer from './DocumentViewer'
import EpistemicBadge from './EpistemicBadge'
import ConfidenceBar from './ConfidenceBar'
import FieldHelp from './FieldHelp'

export function SourceSpan({
  text,
  before,
  after,
}: {
  text: string
  before?: string | null
  after?: string | null
}) {
  return (
    <blockquote className="prov-quote">
      {before && <span>{before}</span>}
      <mark>{text}</mark>
      {after && <span>{after}</span>}
    </blockquote>
  )
}

/** Subject → predicate → object, in a readable line. */
export function Triple({ a }: { a: Assertion }) {
  return (
    <span className="prov-triple">
      <strong>{a.subject_label || a.subject_id}</strong>{' '}
      <span className="prov-pred">{a.predicate}</span>{' '}
      <strong>{a.object_label || a.object_id}</strong>
    </span>
  )
}

/** One premise row. A premise that is itself inferred says so, and says how deep it goes. */
function PremiseRow({ premise, onSelect }: { premise: Premise; onSelect?: (id: string) => void }) {
  if (!premise.visible) {
    return (
      <div className="proof-row proof-row-hidden">
        <span className="proof-triple">
          A fact you are not cleared to see, on another matter.
        </span>
        <code className="proof-method">{premise.assertion_id.slice(0, 12)}</code>
      </div>
    )
  }

  const style = { '--epi-colour': EPISTEMIC[premise.epistemic_class].colour } as CSSProperties
  const restsOn = premise.premises?.length ?? 0

  return (
    <div className="proof-row" style={style}>
      <EpistemicBadge epistemicClass={premise.epistemic_class} size="sm" showLabel={false} />
      <span className="proof-triple">
        <Triple a={premise} />
      </span>
      <ConfidenceBar value={premise.confidence} width={48} />
      <span className="proof-method">{premise.method}</span>
      {restsOn > 0 &&
        (onSelect ? (
          <button className="btn btn-ghost btn-sm" onClick={() => onSelect(premise.assertion_id)}>
            rests on {restsOn} more
          </button>
        ) : (
          <span className="proof-deeper">rests on {restsOn} more</span>
        ))}
    </div>
  )
}

/** The cap on the conclusion: an inference is never firmer than its weakest premise. */
function weakestPremise(premises: Premise[]): number | null {
  const known = premises.filter((p) => p.visible).map((p) => p.confidence)
  return known.length > 0 ? Math.min(...known) : null
}

/** File, page and quote — with the action that resolves them against the original. */
function CitationBlock({
  citation,
  epistemicClass,
  onOpen,
}: {
  citation: PageCitation
  epistemicClass: EpistemicClass
  onOpen: () => void
}) {
  return (
    <div className="prov-citation" style={epiStyle(epistemicClass)}>
      <div className="prov-citation-head">
        <div className="prov-citation-file">
          <strong>{citation.filename || citation.document_id}</strong>
          {citation.page != null && (
            <div className="prov-citation-page">
              Page {citation.page}
              <FieldHelp title="How to check this" text={HELP.pageCitation} />
            </div>
          )}
        </div>
        {citation.link_unavailable ? (
          <span className="prov-citation-nolink">{citation.link_unavailable}</span>
        ) : (
          <button className="btn btn-primary btn-sm" onClick={onOpen}>
            {citation.page != null ? `Open document at page ${citation.page}` : 'Open document'}
          </button>
        )}
      </div>
      {citation.quote && <SourceSpan text={citation.quote} />}
    </div>
  )
}

export default function ProvenancePanel({
  provenance,
  confidenceFloor,
  onClose,
  onSelectAssertion,
  compact = false,
}: {
  provenance: Provenance
  confidenceFloor?: number
  onClose?: () => void
  /** Follow a premise that is itself inferred. Omitted, the depth is stated but not walkable. */
  onSelectAssertion?: (assertionId: string) => void
  compact?: boolean
}) {
  const [viewing, setViewing] = useState(false)
  const a = provenance.assertion
  const citation = provenance.document
  const premises = provenance.premises ?? []
  const isStructured = !!a.source_locator.source_id && !a.source_locator.document_id
  const ceiling = premises.length > 0 ? weakestPremise(premises) : null
  const offsets =
    a.source_locator.char_start != null && a.source_locator.char_end != null
      ? { start: a.source_locator.char_start, end: a.source_locator.char_end }
      : null
  const spanHash = a.source_locator.span_sha256
  const chunkId = citation?.chunk_id ?? a.source_locator.chunk_id

  return (
    <div className="prov-panel">
      <div className="prov-head">
        <div className="prov-head-title">
          <h4>Why the system believes this</h4>
          <div className="prov-triple">
            <Triple a={a} />
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}>
          <EpistemicBadge epistemicClass={a.epistemic_class} tipAlign="right" />
          {onClose && (
            <button className="btn btn-ghost btn-sm" onClick={onClose} aria-label="Close">
              &#x2715;
            </button>
          )}
        </div>
      </div>

      <div className="prov-body">
        {/* Unstructured: the citation leads, because it is what a reader checks. */}
        {citation && (
          <div>
            <div className="prov-section-title">
              The citation
              <FieldHelp text={HELP.sourceLocator} />
            </div>
            <CitationBlock
              citation={citation}
              epistemicClass={a.epistemic_class}
              onOpen={() => setViewing(true)}
            />
          </div>
        )}

        {a.source_locator.document_id && !citation?.quote && (
          <div className="banner banner-warn" style={{ marginBottom: 0 }}>
            <span>
              <strong>No quoted words are attached to this fact.</strong> It names a document
              but nothing to look for in it, so it cannot be checked against the original.
              Treat that as a reason for caution.
            </span>
          </div>
        )}

        <div>
          <dl className="prov-meta">
            <div>
              <dt>
                Confidence
                <FieldHelp text={HELP.confidence} />
              </dt>
              <dd>
                <ConfidenceBar value={a.confidence} floor={confidenceFloor} width={90} />
              </dd>
            </div>
            {typeof a.raw_confidence === 'number' && a.raw_confidence !== a.confidence && (
              <div>
                <dt>
                  As extracted
                  <FieldHelp text={HELP.rawConfidence} />
                </dt>
                <dd>
                  {a.raw_confidence.toFixed(2)}, before review
                </dd>
              </div>
            )}
            <div>
              <dt>
                Method
                <FieldHelp text={HELP.method} />
              </dt>
              <dd>
                <code>{a.method}</code>
              </dd>
            </div>
            <div>
              <dt>
                Review state
                <FieldHelp text={HELP.reviewState} />
              </dt>
              <dd>{a.review_state.replace('_', '-').toLowerCase()}</dd>
            </div>
            <div>
              <dt>
                Recorded
                <FieldHelp title="Transaction time" text={HELP.bitemporal} />
              </dt>
              <dd>{new Date(a.recorded_at).toLocaleString()}</dd>
            </div>
            {a.valid_from && (
              <div>
                <dt>
                  Valid from
                  <FieldHelp title="World time" text={HELP.bitemporal} />
                </dt>
                <dd>{a.valid_from}</dd>
              </div>
            )}
            {a.matter_id && (
              <div>
                <dt>
                  Matter
                  <FieldHelp text={HELP.matterWall} />
                </dt>
                <dd>
                  <code>{a.matter_id}</code>
                </dd>
              </div>
            )}
            {a.superseded_at && (
              <div>
                <dt>
                  Superseded
                  <FieldHelp text={HELP.supersede} />
                </dt>
                <dd style={{ color: 'var(--red)' }}>{new Date(a.superseded_at).toLocaleString()}</dd>
              </div>
            )}
          </dl>
        </div>

        {/* Structured: the source, table and column. */}
        {isStructured && (
          <div>
            <div className="prov-section-title">
              Source record
              <FieldHelp text={HELP.sourceLocator} />
            </div>
            <dl className="prov-meta">
              <div>
                <dt>Source</dt>
                <dd>
                  <code>{a.source_locator.source_id}</code>
                </dd>
              </div>
              {a.source_locator.table && (
                <div>
                  <dt>Table</dt>
                  <dd>
                    <code>{a.source_locator.table}</code>
                  </dd>
                </div>
              )}
              {a.source_locator.column && (
                <div>
                  <dt>Column</dt>
                  <dd>
                    <code>{a.source_locator.column}</code>
                  </dd>
                </div>
              )}
              {a.source_locator.query_sha256 && (
                <div>
                  <dt>
                    Query fingerprint
                    <FieldHelp text="A hash of the query that produced this row, so the exact read can be replayed." />
                  </dt>
                  <dd>
                    <code>{a.source_locator.query_sha256.slice(0, 16)}</code>
                  </dd>
                </div>
              )}
            </dl>
          </div>
        )}

        {/* Inferred: the facts it was drawn from. */}
        {premises.length > 0 && (
          <div>
            <div className="prov-section-title">
              Proof tree
              <FieldHelp text={HELP.proofTree} />
            </div>
            <div className="proof-tree">
              <div className="proof-row proof-row-root" style={epiStyle(a.epistemic_class)}>
                <EpistemicBadge epistemicClass={a.epistemic_class} size="sm" showLabel={false} />
                <span className="proof-triple">
                  <Triple a={a} />
                </span>
                <ConfidenceBar value={a.confidence} width={48} />
                <span className="proof-method">{a.method}</span>
              </div>
              <div className="proof-children">
                {premises.map((p) => (
                  <PremiseRow key={p.assertion_id} premise={p} onSelect={onSelectAssertion} />
                ))}
              </div>
            </div>
            {ceiling !== null && (
              <p className="proof-note">
                <span className="proof-ceiling">
                  Capped at {ceiling.toFixed(2)} by its weakest premise
                  <FieldHelp
                    text="An inference is never recorded as more confident than the least certain fact it rests on. A chain of guesses cannot become a certainty."
                    align="right"
                  />
                </span>
              </p>
            )}
            {a.rule_id && (
              <p className="proof-note">
                Derived by rule <code>{a.rule_id}</code>
                {a.rule_version ? ` (${a.rule_version})` : ''}. Retracting any premise retracts
                this conclusion with it.
              </p>
            )}
          </div>
        )}

        {a.epistemic_class === 'INFERRED' && premises.length === 0 && (
          <div className="banner banner-warn" style={{ marginBottom: 0 }}>
            <span>
              <strong>Proof tree unavailable.</strong> This fact is recorded as inferred, so it must
              have premises. Failing to load them is a data problem worth raising, not a fact you
              should rely on.
            </span>
          </div>
        )}

        {!compact && provenance.history && provenance.history.length > 0 && (
          <div>
            <div className="prov-section-title">
              History
              <FieldHelp text={HELP.retraction} />
            </div>
            <div className="prov-history">
              {provenance.history.map((e) => (
                <div className="prov-event" key={e.event_id}>
                  <span className="prov-when">{new Date(e.timestamp).toLocaleString()}</span>
                  <span className="prov-actor">{e.actor}</span>
                  <span>
                    <strong>{e.action.toLowerCase()}</strong>
                    {e.note ? `, ${e.note}` : ''}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {(offsets || spanHash || chunkId) && (
          <details className="prov-technical">
            <summary>Technical detail</summary>
            <div className="prov-technical-body">
              <dl className="prov-meta">
                {spanHash && (
                  <div>
                    <dt>
                      Text fingerprint
                      <FieldHelp text={HELP.spanHash} />
                    </dt>
                    <dd>
                      <code title={spanHash}>{spanHash.slice(0, 16)}</code>
                    </dd>
                  </div>
                )}
                {chunkId && (
                  <div>
                    <dt>
                      Passage
                      <FieldHelp text="The identifier of the passage the words were taken from, used when tracing a problem back through the pipeline." />
                    </dt>
                    <dd>
                      <code>{chunkId}</code>
                    </dd>
                  </div>
                )}
                {offsets && (
                  <div>
                    <dt>
                      Text position
                      <FieldHelp text={HELP.textOffsets} />
                    </dt>
                    <dd>
                      <code>
                        {offsets.start}&ndash;{offsets.end}
                      </code>
                    </dd>
                  </div>
                )}
              </dl>
              <p className="card-note">
                Kept for diagnosis only. The citation is the file, the page and the quoted words
                above.
              </p>
            </div>
          </details>
        )}

        <div className="prov-locator" style={{ marginTop: 0 }}>
          <span title={a.assertion_id}>assertion {a.assertion_id.slice(0, 16)}</span>
          {a.premises.length > 0 && <span>{a.premises.length} premises</span>}
        </div>
      </div>

      {viewing && citation && citation.page != null && (
        <DocumentViewer
          tenant={getTenantId()}
          documentId={citation.document_id}
          filename={citation.filename || citation.document_id}
          page={citation.page}
          quote={citation.quote}
          onClose={() => setViewing(false)}
        />
      )}
    </div>
  )
}
