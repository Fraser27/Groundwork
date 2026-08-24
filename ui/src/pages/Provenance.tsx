/**
 * Provenance — the audit view.
 *
 * Search every assertion, inspect proof trees, and see what has been retracted.
 * Unlike the review queue this includes facts that were rejected or superseded:
 * an audit trail that hides withdrawn facts is not an audit trail.
 */

import { useEffect, useMemo, useState } from 'react'
import {
  api,
  type Assertion,
  type EpistemicClass,
  type GraphAuditEvent,
  type QueryAuditEvent,
  type ReviewState,
} from '../api'
import { getTenantId } from '../auth'
import { EPISTEMIC_ORDER, HELP, REVIEW_STATE_LABEL } from '../epistemic'
import { useProvenance } from '../useProvenance'
import ConfidenceBar from '../components/ConfidenceBar'
import EpistemicBadge from '../components/EpistemicBadge'
import FieldHelp from '../components/FieldHelp'
import ProvenancePanel from '../components/ProvenancePanel'
import { EmptyState, ErrorState, Spinner } from '../components/Shared'
import { fmtDateTime, fmtNum } from '../format'

export default function Provenance() {
  const tenant = getTenantId()
  const [all, setAll] = useState<Assertion[]>([])
  const [floor, setFloor] = useState(0.8)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [reloadKey, setReloadKey] = useState(0)
  /** Facts is the default: "what does the graph hold, including what it once held" is the question
   *  asked most. Graph changes answers the narrower one, "who altered it", and Questions the read
   *  side, "what did we tell people and on what basis". Three tabs rather than one merged feed:
   *  questions outnumber belief changes by orders of magnitude, so interleaving them would bury
   *  every wipe under a page of queries. */
  const [tab, setTab] = useState<'facts' | 'changes' | 'questions'>('facts')
  const [search, setSearch] = useState('')
  const [classFilter, setClassFilter] = useState<EpistemicClass | '__all__'>('__all__')
  const [stateFilter, setStateFilter] = useState<ReviewState | '__all__' | 'RETRACTED'>('__all__')
  const [asOf, setAsOf] = useState('')
  const [selected, setSelected] = useState<string | null>(null)
  const { provenance, error: provError } = useProvenance(tenant, selected)

  useEffect(() => {
    // 'ALL' is not optional here. Omitting it lets the server apply its PENDING default, so the
    // audit trail showed only unreviewed facts -- an approved graph rendered as "no facts recorded".
    Promise.all([
      api.listAssertions(tenant, { review_state: 'ALL', limit: 500 }),
      api.getSettings(tenant),
    ])
      .then(([a, s]) => {
        setAll(a)
        setFloor(s.min_confidence)
        setError('')
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false))
  }, [tenant, reloadKey])

  const filtered = useMemo(() => {
    let out = all
    if (classFilter !== '__all__') out = out.filter((a) => a.epistemic_class === classFilter)
    if (stateFilter === 'RETRACTED') out = out.filter((a) => !!a.superseded_at)
    else if (stateFilter !== '__all__') out = out.filter((a) => a.review_state === stateFilter)
    if (asOf) {
      // Transaction-time read: what the graph asserted on that date, including
      // facts since withdrawn.
      const t = new Date(asOf).getTime()
      out = out.filter(
        (a) =>
          new Date(a.recorded_at).getTime() <= t &&
          (!a.superseded_at || new Date(a.superseded_at).getTime() > t),
      )
    }
    if (search.trim()) {
      const q = search.toLowerCase()
      out = out.filter(
        (a) =>
          (a.subject_label || a.subject_id).toLowerCase().includes(q) ||
          (a.object_label || a.object_id).toLowerCase().includes(q) ||
          a.predicate.toLowerCase().includes(q) ||
          a.method.toLowerCase().includes(q) ||
          a.assertion_id.toLowerCase().includes(q) ||
          (a.matter_id || '').toLowerCase().includes(q),
      )
    }
    return out
  }, [all, classFilter, stateFilter, asOf, search])

  const retracted = all.filter((a) => !!a.superseded_at).length

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
            <h2>Audit</h2>
            <p>
              Every fact the graph has ever held, including the ones since withdrawn. Facts are never
              edited or deleted, a correction supersedes rather than overwrites, so the record of what
              was believed and when stays intact.
            </p>
          </div>
        </div>
      </div>

      <div className="access-tabs">
        <button
          className={`access-tab${tab === 'facts' ? ' active' : ''}`}
          onClick={() => setTab('facts')}
        >
          Facts
        </button>
        <button
          className={`access-tab${tab === 'changes' ? ' active' : ''}`}
          onClick={() => setTab('changes')}
        >
          Graph changes
        </button>
        <button
          className={`access-tab${tab === 'questions' ? ' active' : ''}`}
          onClick={() => setTab('questions')}
        >
          Questions
        </button>
      </div>

      {error && (
        <ErrorState
          title="Could not load the audit trail"
          detail={error}
          onRetry={retry}
        />
      )}

      {tab === 'changes' && <GraphChanges tenant={tenant} />}

      {tab === 'questions' && <Questions tenant={tenant} onInspect={setSelected} />}

      {tab === 'facts' && (
        <>
      <div className="toolbar">
        <div className="toolbar-field" style={{ flex: 1, minWidth: 260 }}>
          <label>Search</label>
          <input
            placeholder="Search by party, relationship, method, matter or assertion id…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{ width: '100%' }}
          />
        </div>
        <div className="toolbar-field">
          <label>
            How reached
            <FieldHelp text={HELP.epistemicClass} />
          </label>
          <select
            value={classFilter}
            onChange={(e) => setClassFilter(e.target.value as EpistemicClass | '__all__')}
          >
            <option value="__all__">All classes</option>
            {EPISTEMIC_ORDER.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </div>
        <div className="toolbar-field">
          <label>
            State
            <FieldHelp text={HELP.reviewState} />
          </label>
          <select value={stateFilter} onChange={(e) => setStateFilter(e.target.value as never)}>
            <option value="__all__">All states</option>
            {(Object.keys(REVIEW_STATE_LABEL) as ReviewState[]).map((s) => (
              <option key={s} value={s}>
                {REVIEW_STATE_LABEL[s]}
              </option>
            ))}
            <option value="RETRACTED">Superseded or retracted</option>
          </select>
        </div>
        <div className="toolbar-field">
          <label>
            As at
            <FieldHelp text={HELP.asOf} />
          </label>
          <input type="date" value={asOf} onChange={(e) => setAsOf(e.target.value)} />
        </div>
        <div className="toolbar-field toolbar-spacer">
          <label>&nbsp;</label>
          <span className="search-count">
            {filtered.length} of {all.length}
            {retracted > 0 && ` · ${retracted} withdrawn`}
          </span>
        </div>
      </div>

      {asOf && (
        <div className="banner banner-info">
          <span>
            <strong>Historical view.</strong> Showing what the graph asserted on {asOf}: facts
            recorded later are hidden, and facts since withdrawn are shown as they stood. This is the
            view that answers "what did the file show when we advised".
          </span>
        </div>
      )}

      <div className="card">
        <table className="data-table data-table-hover">
          <thead>
            <tr>
              <th>Claim</th>
              <th>
                How reached
                <FieldHelp text={HELP.epistemicClass} />
              </th>
              <th>
                Confidence
                <FieldHelp text={HELP.confidence} />
              </th>
              <th>
                Method
                <FieldHelp text={HELP.method} />
              </th>
              <th>Matter</th>
              <th>State</th>
              <th>Recorded</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((a) => (
              <tr
                key={a.assertion_id}
                onClick={() => setSelected(a.assertion_id)}
                style={a.superseded_at ? { opacity: 0.6 } : undefined}
              >
                <td>
                  <span style={a.superseded_at ? { textDecoration: 'line-through' } : undefined}>
                    <strong>{a.subject_label || a.subject_id}</strong>{' '}
                    <span className="prov-pred">{a.predicate}</span>{' '}
                    <strong>{a.object_label || a.object_id}</strong>
                  </span>
                  {a.premises.length > 0 && (
                    <div className="dim" style={{ fontSize: 11, marginTop: 3 }}>
                      rests on {a.premises.length} premise{a.premises.length === 1 ? '' : 's'}
                    </div>
                  )}
                </td>
                <td>
                  <EpistemicBadge epistemicClass={a.epistemic_class} size="sm" />
                </td>
                <td>
                  <ConfidenceBar value={a.confidence} floor={floor} />
                </td>
                <td>
                  <code style={{ fontSize: 11 }}>{a.method}</code>
                </td>
                <td className="nowrap dim">{a.matter_id || '-'}</td>
                <td className="nowrap">
                  <span
                    className={`tag ${
                      a.review_state === 'APPROVED' || a.review_state === 'AUTO_ASSERTED'
                        ? 'tag-green'
                        : a.review_state === 'REJECTED'
                          ? 'tag-red'
                          : 'tag-orange'
                    }`}
                  >
                    {REVIEW_STATE_LABEL[a.review_state]}
                  </span>
                  {a.superseded_at && (
                    <div style={{ marginTop: 4 }}>
                      <span className="tag tag-neutral" title={HELP.supersede}>
                        superseded
                      </span>
                    </div>
                  )}
                </td>
                <td className="nowrap dim">{fmtDateTime(a.recorded_at)}</td>
              </tr>
            ))}
            {filtered.length === 0 && (
              <tr>
                <td colSpan={7}>
                  <EmptyState title={all.length === 0 ? 'No facts recorded yet' : 'Nothing matches'}>
                    {all.length === 0
                      ? 'The graph is empty. Facts appear here as documents are ingested and claims are reviewed.'
                      : 'Widen the filters, or clear the as-at date.'}
                  </EmptyState>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

        </>
      )}

      {selected && (
        <div className="modal-overlay" onClick={() => setSelected(null)}>
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
                onClose={() => setSelected(null)}
              />
            ) : (
              <Spinner />
            )}
          </div>
        </div>
      )}
    </>
  )
}

/** The sub is kept alongside the email, not replaced by it: an email can be reassigned and the sub
 *  is the recorded identity. Null email means the directory no longer knows the actor. */
function Actor({ sub, email }: { sub: string; email?: string | null }) {
  if (!email) return <span className="mono">{sub}</span>
  return (
    <>
      {email}
      <div>
        <code>{sub}</code>
      </div>
    </>
  )
}

/**
 * Who changed what the system believes.
 *
 * Separate from the fact list because it answers a different question. That one asks what the
 * graph holds; this one asks who altered it -- a reviewer overriding a model, an administrator
 * withdrawing a document. It is the trace back that makes a soft delete auditable rather than a
 * gap: the facts are closed, and this says who closed them and why.
 */
function GraphChanges({ tenant }: { tenant: string }) {
  const [events, setEvents] = useState<GraphAuditEvent[] | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    api
      .graphAudit(tenant)
      .then((e) => {
        setEvents(e)
        setError('')
      })
      .catch((e: Error) => {
        setEvents([])
        setError(e.message)
      })
  }, [tenant])

  if (error) return <ErrorState title="Could not load the change log" detail={error} />
  if (events === null) return <Spinner />

  if (events.length === 0) {
    return (
      <div className="card">
        <EmptyState title="No changes recorded">
          Nothing has been corrected or withdrawn yet. When a reviewer overrides a model, or an
          administrator withdraws a document, it is recorded here and never removed.
        </EmptyState>
      </div>
    )
  }

  return (
    <div className="card">
      <div className="card-header">
        <h3>Graph changes</h3>
        <span className="card-note">
          Append-only &middot; newest first &middot; the facts described are closed, not deleted
        </span>
      </div>
      <table className="data-table">
        <thead>
          <tr>
            <th>When</th>
            <th>Who</th>
            <th>What</th>
            <th>Subject</th>
            <th className="num">Facts</th>
            <th>Reason</th>
          </tr>
        </thead>
        <tbody>
          {events.map((e, i) => (
            <tr key={`${e.at}-${i}`}>
              <td className="nowrap dim">{fmtDateTime(e.at)}</td>
              <td className="audit-actor">
                <Actor sub={e.actor} email={e.actor_email} />
              </td>
              <td>
                <span className={`tag ${e.action === 'SUPERSEDE' ? 'tag-orange' : 'tag-red'}`}>
                  {ACTION_LABEL[e.action] ?? e.action}
                </span>
              </td>
              <td className="mono" style={{ fontSize: 12 }}>
                {e.matter_id || e.document_id || '-'}
              </td>
              <td className="num">{fmtNum(e.affected)}</td>
              <td>{e.reason || <span className="dim">no reason recorded</span>}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="hint">
        A withdrawal removes facts from the current graph and nothing else: a dated read from
        before the entry still reconstructs them, and conclusions drawn earlier are left standing
        because they were true when drawn.
      </p>
    </div>
  )
}

/** Beyond this the ids stop being scannable and push the other columns off the row. */
const IDS_SHOWN = 3

/** Tier 3 cites the graph around a passage, so a row can carry dozens. Collapsed rather than
 *  clipped: an id with no way to reach it is as good as one that was never recorded. */
function FactIds({
  ids,
  total,
  truncated,
  onInspect,
}: {
  ids: string[]
  total: number
  truncated: boolean
  onInspect: (assertionId: string) => void
}) {
  const [expanded, setExpanded] = useState(false)
  const shown = expanded ? ids : ids.slice(0, IDS_SHOWN)
  const hidden = ids.length - shown.length

  return (
    <>
      {fmtNum(total)}
      {truncated && (
        <span
          className="dim"
          title="Over 200 facts were cited and only the first 200 ids were stored. The count is exact."
        >
          {' '}
          (ids capped)
        </span>
      )}
      <div style={{ marginTop: 3 }}>
        {shown.map((id) => (
          <button
            key={id}
            type="button"
            className="link-button mono"
            style={{ fontSize: 11, display: 'block' }}
            onClick={() => onInspect(id)}
          >
            {id}
          </button>
        ))}
        {hidden > 0 && (
          <button
            type="button"
            className="link-button dim"
            style={{ fontSize: 11 }}
            onClick={() => setExpanded(true)}
          >
            +{hidden} more
          </button>
        )}
        {expanded && ids.length > IDS_SHOWN && (
          <button
            type="button"
            className="link-button dim"
            style={{ fontSize: 11 }}
            onClick={() => setExpanded(false)}
          >
            Show fewer
          </button>
        )}
      </div>
    </>
  )
}

/**
 * What was asked, and on what basis.
 *
 * The read side of the audit trail. Without it the graph records how beliefs changed and nothing
 * records that anyone acted on them, so "what did we tell the client, and on what evidence" has no
 * answer. The assertion filter is the inverse and the harder question: a fact turns out to be
 * wrong, and somebody has to find which advice rested on it.
 */
function Questions({
  tenant,
  onInspect,
}: {
  tenant: string
  onInspect: (assertionId: string) => void
}) {
  const [events, setEvents] = useState<QueryAuditEvent[] | null>(null)
  const [scanned, setScanned] = useState(0)
  const [error, setError] = useState('')
  const [factFilter, setFactFilter] = useState('')
  /** Applied on submit, not per keystroke: each change is a server read over a 500-row window. */
  const [applied, setApplied] = useState('')

  useEffect(() => {
    let live = true
    api
      .questionAudit(tenant, { limit: 200, assertionId: applied || undefined })
      .then((r) => {
        if (!live) return
        setEvents(r.questions)
        setScanned(r.scanned)
        setError('')
      })
      .catch((e: Error) => {
        if (!live) return
        setEvents([])
        setError(e.message)
      })
    return () => {
      live = false
    }
  }, [tenant, applied])

  if (error) return <ErrorState title="Could not load the question log" detail={error} />

  return (
    <>
      <div className="toolbar">
        <div className="toolbar-field" style={{ flex: 1, minWidth: 300 }}>
          <label>
            Questions that used a fact
            <FieldHelp text="Paste an assertion id to see which questions rested on it. This is the trace to run when a fact turns out to be wrong: it names the answers that need revisiting." />
          </label>
          <form
            onSubmit={(e) => {
              e.preventDefault()
              setApplied(factFilter.trim())
            }}
            style={{ display: 'flex', gap: 8 }}
          >
            <input
              placeholder="Assertion id"
              value={factFilter}
              onChange={(e) => setFactFilter(e.target.value)}
              style={{ flex: 1 }}
            />
            <button type="submit" className="btn btn-ghost">
              Trace
            </button>
            {applied && (
              <button
                type="button"
                className="btn btn-ghost"
                onClick={() => {
                  setFactFilter('')
                  setApplied('')
                }}
              >
                Clear
              </button>
            )}
          </form>
        </div>
        <div className="toolbar-field toolbar-spacer">
          <label>&nbsp;</label>
          <span className="search-count">
            {events === null ? '-' : `${events.length} question${events.length === 1 ? '' : 's'}`}
            {applied && events !== null && ` of ${scanned} scanned`}
          </span>
        </div>
      </div>

      {applied && (
        <div className="banner banner-info">
          <span>
            <strong>Tracing one fact.</strong> Questions among the last {scanned} that used{' '}
            <code>{applied}</code>. Exact within that window and no further: one question cites many
            facts, so there is no index to read instead.
          </span>
        </div>
      )}

      {events === null ? (
        <Spinner />
      ) : events.length === 0 ? (
        <div className="card">
          <EmptyState title={applied ? 'No question used this fact' : 'No questions recorded'}>
            {applied
              ? 'Nothing in the window scanned rested on it. A question asked before that window may still have.'
              : 'Every answered question is recorded here with the tier that answered and the facts it used. Refused questions are not: they produced no answer, and an administrator sees them in Governance.'}
          </EmptyState>
        </div>
      ) : (
        <div className="card">
          <div className="card-header">
            <h3>Questions asked</h3>
            <span className="card-note">
              Append-only &middot; newest first &middot; refusals are in Governance, not here
            </span>
          </div>
          <table className="data-table">
            <thead>
              <tr>
                <th>When</th>
                <th>Who</th>
                <th>Question</th>
                <th>
                  Answered by
                  <FieldHelp text="Which tier produced the answer. A governed metric is deterministic; an AI-written query is not, and the distinction is recorded rather than inferred later. A route marked retired no longer exists: the log is append-only, so an answer given while it did still says so." />
                </th>
                <th className="num">Facts used</th>
              </tr>
            </thead>
            <tbody>
              {events.map((e, i) => (
                <tr key={`${e.at}-${i}`}>
                  <td className="nowrap dim">{fmtDateTime(e.at)}</td>
                  <td className="audit-actor">
                    <Actor sub={e.actor} email={e.actor_email} />
                  </td>
                  <td>
                    {e.question}
                    {!e.answered && (
                      <div style={{ marginTop: 4 }}>
                        <span className="tag tag-neutral">no answer found</span>
                      </div>
                    )}
                  </td>
                  <td className="nowrap">
                    <span className={`tag ${e.governed ? 'tag-green' : 'tag-orange'}`}>
                      Tier {e.tier} &middot; {TIER_LABEL[e.tier] ?? e.tier_name}
                    </span>
                  </td>
                  <td className="num">
                    {e.facts_used === 0 ? (
                      <span className="dim" title="A metric or an AI-written query cites no facts.">
                        -
                      </span>
                    ) : (
                      <FactIds
                        ids={e.assertion_ids}
                        total={e.facts_used}
                        truncated={e.ids_truncated}
                        onInspect={onInspect}
                      />
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="hint">
            A recorded question is not a copy of the answer: it is who asked, which tier answered,
            and the facts the answer rested on. That is what makes an answer defensible after the
            underlying facts have moved on.
          </p>
        </div>
      )}
    </>
  )
}

/** Plain language for the tier numbers, matching src/query/resolver.py :: Tier.
 *
 * 4 is here and nowhere else in the app: the log is append-only, so rows naming the retired
 * route still have to read as history rather than as a gap.
 */
const TIER_LABEL: Record<number, string> = {
  1: 'approved metric',
  2: 'knowledge graph',
  3: 'passages and graph',
  4: 'AI-written query (retired route)',
}

/** Plain language for the stored action names. */
const ACTION_LABEL: Record<string, string> = {
  SUPERSEDE: 'Corrected by a reviewer',
  WIPE_DOCUMENT: 'Document facts withdrawn',
  WIPE_MATTER: 'Matter facts withdrawn',
}
