/**
 * Provenance — the audit view.
 *
 * Search every assertion, inspect proof trees, and see what has been retracted.
 * Unlike the review queue this includes facts that were rejected or superseded:
 * an audit trail that hides withdrawn facts is not an audit trail.
 */

import { useEffect, useMemo, useState } from 'react'
import { api, type Assertion, type EpistemicClass, type ReviewState } from '../api'
import { getTenantId } from '../auth'
import { EPISTEMIC_ORDER, HELP, REVIEW_STATE_LABEL } from '../epistemic'
import { fallback, MOCK_ASSERTIONS, MOCK_SETTINGS } from '../mocks'
import { useProvenance } from '../useProvenance'
import ConfidenceBar from '../components/ConfidenceBar'
import EpistemicBadge from '../components/EpistemicBadge'
import FieldHelp from '../components/FieldHelp'
import ProvenancePanel from '../components/ProvenancePanel'
import { EmptyState, MockFlag, Spinner } from '../components/Shared'
import { fmtDateTime } from '../format'

export default function Provenance() {
  const tenant = getTenantId()
  const [all, setAll] = useState<Assertion[]>([])
  const [floor, setFloor] = useState(0.8)
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')
  const [classFilter, setClassFilter] = useState<EpistemicClass | '__all__'>('__all__')
  const [stateFilter, setStateFilter] = useState<ReviewState | '__all__' | 'RETRACTED'>('__all__')
  const [asOf, setAsOf] = useState('')
  const [selected, setSelected] = useState<string | null>(null)
  const prov = useProvenance(tenant, selected)

  useEffect(() => {
    Promise.all([
      fallback(api.listAssertions(tenant, { limit: 500 }), MOCK_ASSERTIONS),
      fallback(api.getSettings(tenant), MOCK_SETTINGS),
    ])
      .then(([a, s]) => {
        setAll(a)
        setFloor(s.min_confidence)
      })
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [tenant])

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

  if (loading) return <Spinner />

  return (
    <>
      <div className="page-header">
        <div className="page-header-row">
          <div>
            <h2>Audit</h2>
            <p>
              Every fact the graph has ever held, including the ones since withdrawn. Facts are never
              edited or deleted — a correction supersedes rather than overwrites, so the record of what
              was believed and when stays intact.
            </p>
          </div>
          <MockFlag />
        </div>
      </div>

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
                <td className="nowrap dim">{a.matter_id || '—'}</td>
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
                  <EmptyState title="Nothing matches">
                    Widen the filters, or clear the as-at date.
                  </EmptyState>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {selected && (
        <div className="modal-overlay" onClick={() => setSelected(null)}>
          <div
            className="modal modal-wide"
            onClick={(e) => e.stopPropagation()}
            style={{ padding: 0, overflow: 'hidden' }}
          >
            {prov ? (
              <ProvenancePanel
                provenance={prov}
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
