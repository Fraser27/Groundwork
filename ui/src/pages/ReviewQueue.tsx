/**
 * ReviewQueue — where a lawyer signs off on what a language model read into a document.
 *
 * The design constraint: a reviewer must be able to decide without leaving the
 * card. So each card carries the claim, the words quoted from the document, the
 * page they sit on, and how sure the model was — not a link to go and find those
 * things. Opening the file at that page is one click from the card.
 */

import { useEffect, useMemo, useState } from 'react'
import { api, type Assertion, type EpistemicClass, type Matter, type Ontology } from '../api'
import { getTenantId } from '../auth'
import { EPISTEMIC, HELP } from '../epistemic'
import { useProvenance } from '../useProvenance'
import ConfidenceBar from '../components/ConfidenceBar'
import DocumentViewer from '../components/DocumentViewer'
import EpistemicBadge from '../components/EpistemicBadge'
import FieldHelp from '../components/FieldHelp'
import ProvenancePanel, { SourceSpan } from '../components/ProvenancePanel'
import { EmptyState, ErrorState, Spinner, Toast } from '../components/Shared'
import { epiStyle, fmtDateTime } from '../format'

type Decision = 'approved' | 'rejected' | 'corrected'

export default function ReviewQueue() {
  const tenant = getTenantId()
  const [pending, setPending] = useState<Assertion[]>([])
  const [matters, setMatters] = useState<Matter[]>([])
  const [floor, setFloor] = useState(0.8)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [reloadKey, setReloadKey] = useState(0)
  const [toast, setToast] = useState<{ msg: string; type: string } | null>(null)

  const [matterFilter, setMatterFilter] = useState('__all__')
  const [predicateFilter, setPredicateFilter] = useState('__all__')
  const [sort, setSort] = useState<'confidence_asc' | 'confidence_desc' | 'newest'>('confidence_asc')
  const [classFilter, setClassFilter] = useState<EpistemicClass | '__all__'>('__all__')

  /** Local record of decisions, so a reviewed card stays visible with its outcome. */
  const [decided, setDecided] = useState<Record<string, Decision>>({})
  const [busy, setBusy] = useState<Record<string, boolean>>({})
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [inspecting, setInspecting] = useState<Assertion | null>(null)
  /** The claim being corrected, plus the reviewer's replacement. Null when the dialogue is shut. */
  const [correcting, setCorrecting] = useState<Assertion | null>(null)
  const [ontology, setOntology] = useState<Ontology | null>(null)
  const [openingDoc, setOpeningDoc] = useState<Assertion | null>(null)
  const [expanded, setExpanded] = useState<Set<string>>(new Set())

  const showToast = (msg: string, type = 'success') => {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 3500)
  }

  useEffect(() => {
    Promise.all([
      api.listAssertions(tenant, { review_state: 'PENDING', limit: 200 }),
      api.listMatters(tenant),
      api.getSettings(tenant),
    ])
      .then(([a, m, s]) => {
        setPending(a)
        setMatters(m.matters)
        setFloor(s.min_confidence)
        setError('')
        // The vocabulary a correction may choose from. Fetched after settings because it needs
        // the tenant's active pack, and failing to load it only disables Correct rather than
        // the page: approve and reject do not need it.
        return api.ontology(s.ontology_domain)
      })
      .then((o) => setOntology(o ?? null))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false))
  }, [tenant, reloadKey])

  const predicates = useMemo(
    () => [...new Set(pending.map((a) => a.predicate))].sort(),
    [pending],
  )

  // Ids already in the graph, offered to the correction dialog. Picking an existing one is how a
  // reviewer avoids minting a second node for a company that is already there — the fork that
  // makes a later conflict check come back clean for the wrong reason.
  const knownEntityIds = useMemo(
    () =>
      [...new Set(pending.flatMap((a) => [a.subject_id, a.object_id]))]
        .filter(Boolean)
        .sort(),
    [pending],
  )

  const visible = useMemo(() => {
    let out = pending
    if (matterFilter !== '__all__') out = out.filter((a) => a.matter_id === matterFilter)
    if (predicateFilter !== '__all__') out = out.filter((a) => a.predicate === predicateFilter)
    if (classFilter !== '__all__') out = out.filter((a) => a.epistemic_class === classFilter)
    const sorted = [...out]
    if (sort === 'confidence_asc') sorted.sort((a, b) => a.confidence - b.confidence)
    if (sort === 'confidence_desc') sorted.sort((a, b) => b.confidence - a.confidence)
    if (sort === 'newest')
      sorted.sort((a, b) => +new Date(b.recorded_at) - +new Date(a.recorded_at))
    return sorted
  }, [pending, matterFilter, predicateFilter, classFilter, sort])

  const outstanding = visible.filter((a) => !decided[a.assertion_id])
  const belowFloor = outstanding.filter((a) => a.confidence < floor).length

  // Returns whether it was actually recorded, so a bulk run can report a partial failure.
  const decide = async (a: Assertion, decision: Decision, note?: string): Promise<boolean> => {
    setBusy((b) => ({ ...b, [a.assertion_id]: true }))
    try {
      if (decision === 'approved') await api.approveAssertion(tenant, a.assertion_id, note)
      else await api.rejectAssertion(tenant, a.assertion_id, note)
      setDecided((d) => ({ ...d, [a.assertion_id]: decision }))
      setSelected((s) => {
        const next = new Set(s)
        next.delete(a.assertion_id)
        return next
      })
      return true
    } catch (e) {
      showToast(
        `Could not record that decision: ${(e as Error).message.replace(/^\d+:\s*/, '')}`,
        'error',
      )
      return false
    } finally {
      setBusy((b) => ({ ...b, [a.assertion_id]: false }))
    }
  }

  /** Record the reviewer's version. Their claim goes live; the model's is closed, not deleted. */
  const correct = async (
    a: Assertion,
    body: { predicate?: string; subject_id?: string; object_id?: string; reason: string },
  ) => {
    setBusy((b) => ({ ...b, [a.assertion_id]: true }))
    try {
      const r = await api.correctAssertion(tenant, a.assertion_id, body)
      setDecided((d) => ({ ...d, [a.assertion_id]: 'corrected' }))
      setCorrecting(null)
      showToast(
        `Recorded as ${r.corrected.predicate}, declared by you. The model's version is closed ` +
          'and still readable on the Audit page.',
      )
    } catch (e) {
      showToast((e as Error).message.replace(/^\d+:\s*/, ''), 'error')
    } finally {
      setBusy((b) => ({ ...b, [a.assertion_id]: false }))
    }
  }

  const decideMany = async (decision: Decision) => {
    const targets = outstanding.filter((a) => selected.has(a.assertion_id))
    if (decision === 'approved') {
      const risky = targets.filter((a) => a.confidence < floor)
      const msg = risky.length
        ? `Approve ${targets.length} claims? ${risky.length} of them score below the ${floor.toFixed(
            2,
          )} trust floor, so the model was not confident about them.`
        : `Approve ${targets.length} claims?`
      if (!confirm(msg)) return
    } else if (!confirm(`Reject ${targets.length} claims?`)) return

    let ok = 0
    for (const a of targets) if (await decide(a, decision)) ok++
    if (ok === targets.length) showToast(`${targets.length} claims ${decision}`)
    else showToast(`${ok} of ${targets.length} recorded. The rest were not changed.`, 'error')
  }

  const toggleSelect = (id: string) =>
    setSelected((s) => {
      const next = new Set(s)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })

  const toggleExpand = (id: string) =>
    setExpanded((s) => {
      const next = new Set(s)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })

  const retry = () => {
    setLoading(true)
    setReloadKey((k) => k + 1)
  }

  if (loading) return <Spinner />

  return (
    <>
      <div className="page-header">
        <div className="page-header-row">
          <div>
            <h2>Review queue</h2>
            <p>
              Conclusions a language model drew from your documents. None of them can shape an answer
              until someone here approves them, and each one shows the words it was drawn from, with
              the page they are on, so you are checking the source rather than trusting a summary.
            </p>
          </div>
        </div>
      </div>

      {error && (
        <ErrorState
          title="Could not load the review queue"
          detail={error}
          onRetry={retry}
        />
      )}

      <div className="toolbar">
        <div className="toolbar-field">
          <label>
            Matter
            <FieldHelp text={HELP.matterWall} />
          </label>
          <select value={matterFilter} onChange={(e) => setMatterFilter(e.target.value)}>
            <option value="__all__">All matters</option>
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
            Relationship
            <FieldHelp text={HELP.governingPredicate} />
          </label>
          <select value={predicateFilter} onChange={(e) => setPredicateFilter(e.target.value)}>
            <option value="__all__">All relationships</option>
            {predicates.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </div>
        <div className="toolbar-field">
          <label>
            Class
            <FieldHelp text={HELP.epistemicClass} />
          </label>
          <select
            value={classFilter}
            onChange={(e) => setClassFilter(e.target.value as EpistemicClass | '__all__')}
          >
            <option value="__all__">All classes</option>
            <option value="EXTRACTED_MODEL">Model-extracted</option>
            <option value="INFERRED">Inferred</option>
            <option value="PREDICTED">Predicted</option>
          </select>
        </div>
        <div className="toolbar-field">
          <label>Order</label>
          <select value={sort} onChange={(e) => setSort(e.target.value as typeof sort)}>
            <option value="confidence_asc">Least confident first</option>
            <option value="confidence_desc">Most confident first</option>
            <option value="newest">Newest first</option>
          </select>
        </div>
        <div className="toolbar-field toolbar-spacer">
          <label>&nbsp;</label>
          <span className="search-count">
            {outstanding.length} outstanding
            {belowFloor > 0 && ` · ${belowFloor} below the trust floor`}
          </span>
        </div>
      </div>

      {selected.size > 0 && (
        <div className="bulk-bar">
          <strong>{selected.size} selected</strong>
          <span className="card-note">
            Bulk decisions are recorded individually against your name, one audit event each.
          </span>
          <div className="bulk-actions">
            <button className="btn btn-approve btn-sm" onClick={() => decideMany('approved')}>
              Approve selected
            </button>
            <button className="btn btn-reject btn-sm" onClick={() => decideMany('rejected')}>
              Reject selected
            </button>
            <button className="btn btn-ghost btn-sm" onClick={() => setSelected(new Set())}>
              Clear
            </button>
          </div>
        </div>
      )}

      <div className="review-layout">
        <div className="review-list">
          {visible.length === 0 && !error && (
            <div className="card">
              <EmptyState title="Nothing to review">
                Every model-extracted claim has been signed off. New ones appear here as documents
                finish extracting.
              </EmptyState>
            </div>
          )}

          {visible.map((a) => {
            const decision = decided[a.assertion_id]
            const meta = EPISTEMIC[a.epistemic_class]
            const low = a.confidence < floor
            return (
              <div
                key={a.assertion_id}
                className={[
                  'review-card',
                  decision ? 'review-resolved' : '',
                  inspecting?.assertion_id === a.assertion_id ? 'selected' : '',
                ]
                  .filter(Boolean)
                  .join(' ')}
                style={epiStyle(a.epistemic_class)}
              >
                <div className="review-card-head">
                  <div style={{ display: 'flex', gap: 10, minWidth: 0 }}>
                    {!decision && (
                      <input
                        type="checkbox"
                        checked={selected.has(a.assertion_id)}
                        onChange={() => toggleSelect(a.assertion_id)}
                        style={{ marginTop: 4 }}
                        aria-label={`Select claim ${a.assertion_id}`}
                      />
                    )}
                    <div className="review-claim">
                      <strong>{a.subject_label || a.subject_id}</strong>{' '}
                      <span className="prov-pred" title={HELP.governingPredicate}>
                        {a.predicate}
                      </span>{' '}
                      <strong>{a.object_label || a.object_id}</strong>
                      {a.subject_type && a.object_type && (
                        <div style={{ fontSize: 11.5, color: 'var(--text-dim)', marginTop: 3 }}>
                          {a.subject_type} → {a.object_type}
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="review-badges">
                    <EpistemicBadge epistemicClass={a.epistemic_class} tipAlign="right" />
                    <ConfidenceBar value={a.confidence} floor={floor} width={78} />
                  </div>
                </div>

                {/* The quote. This is the whole point of the page. */}
                {a.source_locator.quote ? (
                  <>
                    <div className="prov-section-title">
                      The words in the document
                      <FieldHelp text={HELP.quote} />
                    </div>
                    <SourceSpan
                      text={a.source_locator.quote}
                      before={
                        expanded.has(a.assertion_id)
                          ? (a.source_context ?? '').split(a.source_locator.quote)[0]
                          : undefined
                      }
                      after={
                        expanded.has(a.assertion_id)
                          ? (a.source_context ?? '').split(a.source_locator.quote)[1]
                          : undefined
                      }
                    />
                    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 7 }}>
                      {a.source_locator.document_id && a.source_locator.page != null && (
                        <button className="btn btn-ghost btn-sm" onClick={() => setOpeningDoc(a)}>
                          Open document at page {a.source_locator.page}
                        </button>
                      )}
                      {a.source_context && (
                        <button
                          className="btn btn-ghost btn-sm"
                          onClick={() => toggleExpand(a.assertion_id)}
                        >
                          {expanded.has(a.assertion_id)
                            ? 'Hide surrounding text'
                            : 'Show surrounding text'}
                        </button>
                      )}
                    </div>
                  </>
                ) : (
                  <div className="banner banner-info" style={{ marginBottom: 0 }}>
                    <span>
                      {a.epistemic_class === 'INFERRED'
                        ? 'This claim was derived by a rule rather than read anywhere. Open the proof tree to see what it rests on.'
                        : 'No quoted words are attached to this claim, so there is nothing to check it against. Treat that as a reason for caution.'}
                    </span>
                  </div>
                )}

                <div className="review-meta">
                  {(a.source_locator.filename || a.source_locator.document_id) && (
                    <span>
                      {a.source_locator.filename || a.source_locator.document_id}
                      {a.source_locator.page != null && `, page ${a.source_locator.page}`}
                    </span>
                  )}
                  {a.matter_id && <span>{a.matter_id}</span>}
                  <span>
                    <code>{a.method}</code>
                  </span>
                  <span>{fmtDateTime(a.recorded_at)}</span>
                </div>

                {low && !decision && (
                  <div className="banner banner-warn" style={{ margin: '11px 0 0' }}>
                    <span>
                      Scored {a.confidence.toFixed(2)}, below the {floor.toFixed(2)} trust floor.{' '}
                      {meta.label === 'Inferred'
                        ? 'Check the premises before approving.'
                        : 'The model was not confident, read the quoted words closely.'}
                    </span>
                  </div>
                )}

                <div className="review-actions">
                  {decision ? (
                    <span
                      className={`tag ${decision === 'approved' ? 'tag-green' : 'tag-red'}`}
                      style={{ fontSize: 12 }}
                    >
                      {decision === 'approved'
                        ? 'Approved'
                        : decision === 'corrected'
                          ? 'Corrected'
                          : 'Rejected'}{' '}
                      by you
                    </span>
                  ) : (
                    <>
                      <button
                        className="btn btn-approve btn-sm"
                        disabled={busy[a.assertion_id]}
                        onClick={() => decide(a, 'approved')}
                      >
                        Approve
                      </button>
                      <button
                        className="btn btn-reject btn-sm"
                        disabled={busy[a.assertion_id]}
                        onClick={() => {
                          const note = prompt(
                            'Why is this wrong? The reason is stored on the audit trail.',
                          )
                          if (note === null) return
                          decide(a, 'rejected', note)
                        }}
                      >
                        Reject
                      </button>
                      <button
                        className="btn btn-ghost btn-sm"
                        disabled={busy[a.assertion_id] || !ontology}
                        title={
                          ontology
                            ? 'The relationship is real but this is the wrong reading of it'
                            : 'The vocabulary could not be loaded, so a correction cannot be checked'
                        }
                        onClick={() => setCorrecting(a)}
                      >
                        Correct
                      </button>
                    </>
                  )}
                  <span
                    className="card-note"
                    title="Approving records your name and the time against this fact, and lets it start shaping answers. Rejecting withdraws it and retracts anything inferred from it."
                    style={{ fontSize: 11.5 }}
                  >
                    {decision === 'corrected'
                      ? 'Your version is live; the model\u2019s is closed and still readable.'
                      : decision
                        ? 'Recorded on the audit trail.'
                        : 'Your decision is recorded against this fact.'}
                  </span>
                  <div className="review-actions-right">
                    <button className="btn btn-ghost btn-sm" onClick={() => setInspecting(a)}>
                      Full provenance
                    </button>
                  </div>
                </div>
              </div>
            )
          })}
        </div>

        <aside className="review-sidebar">
          {inspecting ? (
            <InspectorPanel
              key={inspecting.assertion_id}
              assertion={inspecting}
              floor={floor}
              onClose={() => setInspecting(null)}
            />
          ) : (
            <div className="card card-tight">
              <div className="card-header" style={{ marginBottom: 9 }}>
                <h3>
                  What you are deciding
                  <FieldHelp text={HELP.reviewState} />
                </h3>
              </div>
              <p className="card-note">
                A language model read a passage and drew a relationship from it. Approving it lets
                that relationship shape answers, conflict checks and deadline tracking. Rejecting it
                withdraws the claim and retracts anything already inferred from it.
              </p>
              <p className="card-note" style={{ marginTop: 10 }}>
                Quotes the system confirmed are on the page they name do not appear here, and neither
                do records declared by a system of record. Both are checkable without judgement, so
                they need no sign-off. What is left is the judgement calls, which is what keeps this
                queue clearable.
              </p>
              <div className="prov-section-title" style={{ marginTop: 14 }}>
                Trust floor
                <FieldHelp text={HELP.confidenceFloor} />
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <ConfidenceBar value={floor} floor={floor} width={100} />
                <span className="card-note">set in Admin</span>
              </div>
            </div>
          )}
        </aside>
      </div>

      {openingDoc?.source_locator.document_id && openingDoc.source_locator.page != null && (
        <DocumentViewer
          tenant={tenant}
          documentId={openingDoc.source_locator.document_id}
          filename={
            openingDoc.source_locator.filename || openingDoc.source_locator.document_id
          }
          page={openingDoc.source_locator.page}
          quote={openingDoc.source_locator.quote}
          onClose={() => setOpeningDoc(null)}
        />
      )}

      {correcting && ontology && (
        <CorrectionDialog
          assertion={correcting}
          ontology={ontology}
          knownEntityIds={knownEntityIds}
          busy={!!busy[correcting.assertion_id]}
          onCancel={() => setCorrecting(null)}
          onSubmit={(body) => correct(correcting, body)}
        />
      )}

      <Toast toast={toast} />
    </>
  )
}

/**
 * Record what the reviewer says instead.
 *
 * A dialogue rather than an inline edit, because this is not an edit: the model's claim is closed
 * and the reviewer's becomes a separate fact declared by them. Saying so on the form matters --
 * a reviewer who believes they are fixing a typo would be surprised to find both versions in the
 * audit trail, and the two-record shape is the thing that makes the trail honest.
 *
 * Predicates a rule concludes are absent from the picker: only a rule may draw those, and a
 * hand-asserted conflict would carry no premises. The API refuses them anyway, but offering an
 * option that is always rejected is worse than not offering it.
 */
function NewEntityHint() {
  return (
    <p className="hint">
      Not an entity already in the graph — this will create a new one. If the company is already
      here under another spelling, pick that instead: two nodes for one company is what makes a
      later conflict check come back clean.
    </p>
  )
}

function CorrectionDialog({
  assertion,
  ontology,
  knownEntityIds,
  busy,
  onCancel,
  onSubmit,
}: {
  assertion: Assertion
  ontology: Ontology
  knownEntityIds: string[]
  busy: boolean
  onCancel: () => void
  onSubmit: (body: {
    predicate?: string
    subject_id?: string
    object_id?: string
    reason: string
  }) => void
}) {
  const [predicate, setPredicate] = useState(assertion.predicate)
  const [subjectId, setSubjectId] = useState(assertion.subject_id)
  const [objectId, setObjectId] = useState(assertion.object_id)
  const [reason, setReason] = useState('')

  const ruleConclusions = useMemo(
    () => new Set(ontology.rules.map((r) => r.then.match(/\[:([A-Z_]+)\]/)?.[1]).filter(Boolean)),
    [ontology],
  )
  const options = useMemo(() => {
    const all = [...ontology.governing_predicates, ...ontology.descriptive_predicates]
    return all.filter((p) => !ruleConclusions.has(p.id)).sort((a, b) => a.id.localeCompare(b.id))
  }, [ontology, ruleConclusions])

  const changed =
    predicate !== assertion.predicate ||
    subjectId !== assertion.subject_id ||
    objectId !== assertion.object_id

  // Reported, never blocked: a genuinely new party is an ordinary thing to record. The point is
  // that minting a node should be a visible act rather than a side effect of typing, because a
  // second node for one company is what makes a later conflict check come back clean.
  const known = useMemo(() => new Set(knownEntityIds), [knownEntityIds])
  const isNewEntity = (id: string) => id.trim() !== '' && !known.has(id)

  return (
    <div className="modal-overlay" onClick={onCancel}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h3>Record what this should say</h3>
        <p className="modal-sub">
          This does not edit the model's claim. Yours is recorded as a separate fact, declared by
          you rather than extracted, and the model's is closed but still readable. The trail then
          shows a person overrode a specific reading, rather than that the model said something it
          did not.
        </p>

        <div className="consequence">
          <div className="consequence-title">What gets recorded</div>
          <ul>
            <li>
              A new fact: <strong>{subjectId}</strong> {predicate} <strong>{objectId}</strong>,
              declared by you, citing the same page and sentence.
            </li>
            <li>
              The model's version is closed, not deleted. An as-of read before now still shows it.
            </li>
            <li>Your reason is kept and shown on the Audit page beside your name.</li>
          </ul>
        </div>

        <div className="form-group">
          <label>
            Relationship
            <FieldHelp text="The closed vocabulary this tenant uses. Relationships a rule concludes are absent on purpose: only a rule may draw those, from facts that carry it as premises, so one asserted by hand would defend nothing." />
          </label>
          <select value={predicate} onChange={(e) => setPredicate(e.target.value)}>
            {options.map((p) => (
              <option key={p.id} value={p.id}>
                {p.id}
                {p.governing ? ' (governing)' : ''}
              </option>
            ))}
          </select>
        </div>

        <div className="form-row">
          <div className="toolbar-field" style={{ flex: 1, minWidth: 200 }}>
            <label>From</label>
            <input
              value={subjectId}
              list="known-entity-ids"
              onChange={(e) => setSubjectId(e.target.value)}
            />
            {isNewEntity(subjectId) && <NewEntityHint />}
          </div>
          <div className="toolbar-field" style={{ flex: 1, minWidth: 200 }}>
            <label>To</label>
            <input
              value={objectId}
              list="known-entity-ids"
              onChange={(e) => setObjectId(e.target.value)}
            />
            {isNewEntity(objectId) && <NewEntityHint />}
          </div>
          {/* Shared by both fields. Either end of an edge can name a company already in the
              graph, and a reviewer who cannot see the existing spelling will invent a new one. */}
          <datalist id="known-entity-ids">
            {knownEntityIds.map((id) => (
              <option key={id} value={id} />
            ))}
          </datalist>
        </div>

        <div className="form-group">
          <label>
            Reason, required
            <FieldHelp text="Written for whoever reads the file in a year. Say what the document actually supports: “the letter names Calder as the adverse party” explains itself, “model was wrong” does not." />
          </label>
          <textarea
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            placeholder="The engagement letter names Calder as the adverse party, not the client."
            autoFocus
          />
          {!changed && (
            <p className="hint">
              Nothing has changed yet. To accept the claim as the model read it, close this and
              approve instead.
            </p>
          )}
        </div>

        <div className="modal-actions">
          <button className="btn btn-ghost" onClick={onCancel}>
            Cancel
          </button>
          <button
            className="btn btn-primary"
            disabled={busy || !reason.trim() || !changed}
            onClick={() =>
              onSubmit({
                predicate,
                subject_id: subjectId,
                object_id: objectId,
                reason: reason.trim(),
              })
            }
          >
            {busy ? 'Recording…' : 'Record correction'}
          </button>
        </div>
      </div>
    </div>
  )
}

/** Loads full provenance for one claim — the source document and any proof tree. */
function InspectorPanel({
  assertion,
  floor,
  onClose,
}: {
  assertion: Assertion
  floor: number
  onClose: () => void
}) {
  const { provenance, error } = useProvenance(getTenantId(), assertion.assertion_id)
  if (error) return <ErrorState title="Could not load this provenance" detail={error} />
  if (!provenance) return <Spinner />
  return <ProvenancePanel provenance={provenance} confidenceFloor={floor} onClose={onClose} />
}
