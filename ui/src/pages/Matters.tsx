import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  api,
  type Assertion,
  type DocumentSummary,
  type Matter,
  type WithheldMatter,
} from '../api'
import { getTenantId } from '../auth'
import { HELP } from '../epistemic'
import { fallback, MOCK_ASSERTIONS, MOCK_DOCUMENTS, MOCK_MATTERS_RESPONSE } from '../mocks'
import ConfidenceBar from '../components/ConfidenceBar'
import EpistemicBadge from '../components/EpistemicBadge'
import FieldHelp from '../components/FieldHelp'
import { EmptyState, IngestPill, MockFlag, Spinner } from '../components/Shared'
import { fmtDate, fmtNum } from '../format'

export default function Matters() {
  const tenant = getTenantId()
  const [matters, setMatters] = useState<Matter[]>([])
  // Kept in its own piece of state, never merged into `matters`. A screened matter must
  // not be able to reach the readable list through a filter or a sort.
  const [withheld, setWithheld] = useState<WithheldMatter[]>([])
  const [docs, setDocs] = useState<DocumentSummary[]>([])
  const [assertions, setAssertions] = useState<Assertion[]>([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState('')
  const [selectedId, setSelectedId] = useState<string | null>(null)

  useEffect(() => {
    Promise.all([
      fallback(api.listMatters(tenant), MOCK_MATTERS_RESPONSE),
      fallback(api.listDocuments(tenant), MOCK_DOCUMENTS),
      fallback(api.listAssertions(tenant, { limit: 200 }), MOCK_ASSERTIONS),
    ])
      .then(([m, d, a]) => {
        setMatters(m.matters)
        setWithheld(m.withheld)
        setDocs(d)
        setAssertions(a)
      })
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [tenant])

  const filtered = useMemo(() => {
    if (!filter.trim()) return matters
    const q = filter.toLowerCase()
    return matters.filter(
      (m) =>
        m.matter_id.toLowerCase().includes(q) ||
        m.name.toLowerCase().includes(q) ||
        (m.client || '').toLowerCase().includes(q),
    )
  }, [matters, filter])

  const selected = matters.find((m) => m.matter_id === selectedId) ?? null

  if (loading) return <Spinner />

  if (selected) {
    const matterDocs = docs.filter((d) => d.matter_id === selected.matter_id)
    const matterAssertions = assertions.filter((a) => a.matter_id === selected.matter_id)
    return (
      <>
        <button className="back-link btn-ghost" style={{ border: 'none', background: 'none', cursor: 'pointer' }} onClick={() => setSelectedId(null)}>
          ← Back to matters
        </button>

        <div className="page-header">
          <div className="page-header-row">
            <div>
              <h2>{selected.name}</h2>
              <p>
                <code>{selected.matter_id}</code>
                {selected.client && ` · ${selected.client}`}
              </p>
            </div>
            <span className={`tag ${selected.status === 'open' ? 'tag-green' : 'tag-neutral'}`}>
              {selected.status}
            </span>
          </div>
        </div>

        <div className="stats-grid">
          <div className="stat-card">
            <div className="label">Documents</div>
            <div className="value accent">{fmtNum(selected.counts?.documents ?? matterDocs.length)}</div>
          </div>
          <div className="stat-card">
            <div className="label">
              Facts
              <FieldHelp text={HELP.epistemicClass} />
            </div>
            <div className="value purple">{fmtNum(selected.counts?.assertions ?? matterAssertions.length)}</div>
          </div>
          <div className="stat-card">
            <div className="label">
              Pending review
              <FieldHelp text={HELP.reviewState} />
            </div>
            <div className={`value ${(selected.counts?.pending_review ?? 0) > 0 ? 'orange' : 'green'}`}>
              {fmtNum(selected.counts?.pending_review ?? 0)}
            </div>
            <div className="sub">
              <Link to="/review">Review queue</Link>
            </div>
          </div>
          <div className="stat-card">
            <div className="label">
              Potential conflicts
              <FieldHelp text="Inferred where the firm both acts for and opposes the same party. Fires only on facts declared by a system of record or confirmed by a check, a conflict flag resting on a model's guess would be worse than none." />
            </div>
            <div className={`value ${(selected.counts?.conflicts ?? 0) > 0 ? 'red' : 'green'}`}>
              {fmtNum(selected.counts?.conflicts ?? 0)}
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h3>Documents</h3>
            <Link to="/documents" className="btn btn-ghost btn-sm">
              Ingest pipeline
            </Link>
          </div>
          <table className="data-table">
            <thead>
              <tr>
                <th>File</th>
                <th>
                  State
                  <FieldHelp text={HELP.ingestState} />
                </th>
                <th className="num">Facts</th>
                <th className="num">Pending</th>
                <th>Uploaded</th>
              </tr>
            </thead>
            <tbody>
              {matterDocs.map((d) => (
                <tr key={d.document_id}>
                  <td>{d.filename}</td>
                  <td>
                    <IngestPill state={d.state} />
                  </td>
                  <td className="num">{fmtNum(d.assertion_count)}</td>
                  <td className="num">
                    {d.pending_review_count > 0 ? (
                      <span className="tag tag-orange">{d.pending_review_count}</span>
                    ) : (
                      '-'
                    )}
                  </td>
                  <td className="nowrap dim">{fmtDate(d.uploaded_at)}</td>
                </tr>
              ))}
              {matterDocs.length === 0 && (
                <tr>
                  <td colSpan={5} className="empty-state">
                    No documents on this matter yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        <div className="card">
          <div className="card-header">
            <h3>
              Facts on this matter
              <FieldHelp text={HELP.epistemicClass} />
            </h3>
            <Link to="/provenance" className="btn btn-ghost btn-sm">
              Audit view
            </Link>
          </div>
          <table className="data-table">
            <thead>
              <tr>
                <th>Claim</th>
                <th>How reached</th>
                <th>
                  Confidence
                  <FieldHelp text={HELP.confidence} />
                </th>
                <th>State</th>
              </tr>
            </thead>
            <tbody>
              {matterAssertions.map((a) => (
                <tr key={a.assertion_id}>
                  <td>
                    <strong>{a.subject_label || a.subject_id}</strong>{' '}
                    <span className="prov-pred">{a.predicate}</span>{' '}
                    <strong>{a.object_label || a.object_id}</strong>
                  </td>
                  <td>
                    <EpistemicBadge epistemicClass={a.epistemic_class} size="sm" />
                  </td>
                  <td>
                    <ConfidenceBar value={a.confidence} floor={0.8} />
                  </td>
                  <td className="nowrap dim">{a.review_state.replace('_', '-').toLowerCase()}</td>
                </tr>
              ))}
              {matterAssertions.length === 0 && (
                <tr>
                  <td colSpan={4} className="empty-state">
                    No facts recorded on this matter yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </>
    )
  }

  return (
    <>
      <div className="page-header">
        <div className="page-header-row">
          <div>
            <h2>Matters</h2>
            <p>
              Matters are subgraphs of one firm-wide graph, not separate graphs — conflict checking is
              by definition cross-matter, and shared parties are the conflict signal.
            </p>
          </div>
          <MockFlag />
        </div>
      </div>

      {withheld.length > 0 && (
        <div className="withheld-block">
          <div className="withheld-block-head">
            <h3>
              {withheld.length} matter{withheld.length === 1 ? '' : 's'} withheld from you
              <FieldHelp text={HELP.ethicalScreen} />
            </h3>
            <span className="tag tag-red">Screened</span>
          </div>
          <p className="withheld-block-note">
            You cannot read these matters, their documents, or anything recorded on them. They are
            named here on purpose: if they were simply hidden, a conflict check could come back
            clean because the matching matter was invisible, and someone would proceed on it.
            Nothing here can be opened, and none of it appears in the list below.
          </p>
          <div className="withheld-list">
            {withheld.map((w) => (
              <div className="withheld-item" key={w.matter_id}>
                <div className="withheld-item-head">
                  <strong>{w.matter_id}</strong>
                  <code>withheld</code>
                </div>
                <div className="withheld-field">
                  <span className="withheld-field-label">Reason recorded</span>
                  {w.reason}
                </div>
                <div className="withheld-field">
                  <span className="withheld-field-label">Who to contact</span>
                  {w.contact ? (
                    w.contact
                  ) : (
                    <span className="dim">
                      No contact was given. Ask your risk team about this matter.
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="search-bar">
        <input
          placeholder="Filter by matter id, name or client…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        />
        {filter && (
          <span className="search-count">
            {filtered.length} of {matters.length}
          </span>
        )}
      </div>

      <div className="card">
        <table className="data-table data-table-hover">
          <thead>
            <tr>
              <th>Matter</th>
              <th>Client</th>
              <th>Status</th>
              <th className="num">Documents</th>
              <th className="num">Facts</th>
              <th className="num">
                Pending
                <FieldHelp text={HELP.reviewState} align="right" />
              </th>
              <th className="num">Conflicts</th>
              <th>Opened</th>
            </tr>
          </thead>
          <tbody>
            {/* Only readable matters reach here — a screened one never enters `matters`. */}
            {filtered.map((m) => (
              <tr key={m.matter_id} onClick={() => setSelectedId(m.matter_id)}>
                <td>
                  <strong>{m.name}</strong>
                  <div className="dim" style={{ fontSize: 11.5 }}>
                    <code>{m.matter_id}</code>
                  </div>
                </td>
                <td className="dim">{m.client || '-'}</td>
                <td>
                  <span className={`tag ${m.status === 'open' ? 'tag-green' : 'tag-neutral'}`}>
                    {m.status}
                  </span>
                </td>
                <td className="num">{fmtNum(m.counts?.documents)}</td>
                <td className="num">{fmtNum(m.counts?.assertions)}</td>
                <td className="num">
                  {m.counts?.pending_review ? (
                    <span className="tag tag-orange">{m.counts.pending_review}</span>
                  ) : (
                    '-'
                  )}
                </td>
                <td className="num">
                  {m.counts?.conflicts ? (
                    <span className="tag tag-red">{m.counts.conflicts}</span>
                  ) : (
                    '-'
                  )}
                </td>
                <td className="nowrap dim">{fmtDate(m.opened_at)}</td>
              </tr>
            ))}
            {filtered.length === 0 && (
              <tr>
                <td colSpan={8}>
                  <EmptyState title="No matters match">
                    Matters arrive from the case management system as declared records.
                  </EmptyState>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </>
  )
}
