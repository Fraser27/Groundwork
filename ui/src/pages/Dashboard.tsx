import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, type DashboardStats } from '../api'
import { getTenantId } from '../auth'
import { EPISTEMIC, EPISTEMIC_ORDER, HELP } from '../epistemic'
import EpistemicBadge from '../components/EpistemicBadge'
import FieldHelp from '../components/FieldHelp'
import { ErrorState, IngestPill, Spinner } from '../components/Shared'
import { epiStyle, fmtDateTime, fmtNum } from '../format'

export default function Dashboard() {
  const tenant = getTenantId()
  const [stats, setStats] = useState<DashboardStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [reloadKey, setReloadKey] = useState(0)

  useEffect(() => {
    api
      .dashboard(tenant)
      .then((d) => {
        setStats(d)
        setError('')
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false))
  }, [tenant, reloadKey])

  const retry = () => {
    setLoading(true)
    setReloadKey((k) => k + 1)
  }

  if (loading) return <Spinner />
  if (error || !stats)
    return (
      <ErrorState
        title="Could not load the dashboard"
        detail={error}
        onRetry={retry}
      />
    )

  const total = EPISTEMIC_ORDER.reduce((n, c) => n + (stats.assertions_by_class[c] || 0), 0)
  const docStates = Object.entries(stats.documents_by_state) as [string, number][]

  return (
    <>
      <div className="page-header">
        <div className="page-header-row">
          <div>
            <h2>Dashboard</h2>
            <p>
              What the graph holds, and how it came to hold it. Every count below is broken down by
              how the facts were reached, because that is what determines how far they can be relied
              on.
            </p>
          </div>
        </div>
      </div>

      {stats.pending_review > 0 && (
        <div className="banner banner-warn">
          <span>
            <strong>{stats.pending_review} claims are waiting for review.</strong> A language model
            proposed them; until a person approves them they are excluded from every answer.{' '}
            <Link to="/review">Open the review queue</Link>.
          </span>
        </div>
      )}

      <div className="card-header" style={{ marginBottom: 12 }}>
        <h3>
          Facts by how they were reached
          <FieldHelp title="Epistemic class" text={HELP.epistemicClass} />
        </h3>
        <span className="card-note">{fmtNum(total)} total</span>
      </div>

      <div className="stats-grid">
        {EPISTEMIC_ORDER.map((c) => {
          const meta = EPISTEMIC[c]
          const n = stats.assertions_by_class[c] || 0
          const share = total ? Math.round((n / total) * 100) : 0
          return (
            <div className="stat-card epi-tile" key={c} style={epiStyle(c)}>
              <div className="label">
                <EpistemicBadge epistemicClass={c} size="sm" />
              </div>
              <div className="value" style={{ color: meta.colour }}>
                {fmtNum(n)}
              </div>
              <div className="sub">
                {share}% of the graph &middot;{' '}
                {meta.autoAsserted ? 'no review needed' : c === 'PREDICTED' ? 'never in answers' : 'reviewed'}
              </div>
            </div>
          )
        })}
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <div className="label">
            Pending review
            <FieldHelp text={HELP.reviewState} />
          </div>
          <div className={`value ${stats.pending_review > 0 ? 'orange' : 'green'}`}>
            {fmtNum(stats.pending_review)}
          </div>
          <div className="sub">
            {stats.pending_review > 0 ? (
              <Link to="/review">Sign off claims</Link>
            ) : (
              'Nothing awaiting sign-off'
            )}
          </div>
        </div>
        <div className="stat-card">
          <div className="label">
            Matters
            <FieldHelp text={HELP.matterWall} />
          </div>
          <div className="value accent">{fmtNum(stats.matters)}</div>
          <div className="sub">
            <Link to="/matters">Browse matters</Link>
          </div>
        </div>
        <div className="stat-card">
          <div className="label">
            Governed metrics
            <FieldHelp text={HELP.governedMetric} />
          </div>
          <div className="value green">
            {fmtNum(stats.metrics.approved)}
            <span style={{ fontSize: 15, color: 'var(--text-dim)', fontWeight: 500 }}>
              {' '}
              / {stats.metrics.total}
            </span>
          </div>
          <div className="sub">approved &middot; compile without a model</div>
        </div>
        <div className="stat-card">
          <div className="label">Documents live</div>
          <div className="value purple">{fmtNum(stats.documents_by_state.LIVE ?? 0)}</div>
          <div className="sub">
            <Link to="/documents">Ingest pipeline</Link>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <h3>
            Documents in the pipeline
            <FieldHelp text={HELP.ingestState} />
          </h3>
          <Link to="/documents" className="btn btn-ghost btn-sm">
            View all
          </Link>
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
          {docStates.length === 0 && <span className="card-note">No documents ingested yet.</span>}
          {docStates.map(([state, n]) => (
            <div
              key={state}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                padding: '7px 11px',
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius-sm)',
              }}
            >
              <IngestPill state={state as never} />
              <strong style={{ fontVariantNumeric: 'tabular-nums' }}>{fmtNum(n)}</strong>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <h3>Recent activity</h3>
          <Link to="/provenance" className="btn btn-ghost btn-sm">
            Full audit trail
          </Link>
        </div>
        <table className="data-table">
          <thead>
            <tr>
              <th>When</th>
              <th>
                Who or what
                <FieldHelp text={HELP.method} />
              </th>
              <th>Action</th>
              <th>Detail</th>
            </tr>
          </thead>
          <tbody>
            {stats.recent_activity.map((e) => (
              <tr key={e.event_id}>
                <td className="nowrap dim">{fmtDateTime(e.timestamp)}</td>
                <td className="nowrap">
                  <code style={{ fontSize: 11.5 }}>{e.actor}</code>
                </td>
                <td className="nowrap">
                  <span style={{ display: 'inline-flex', alignItems: 'center', gap: 7 }}>
                    {e.action}
                    {e.epistemic_class && (
                      <EpistemicBadge
                        epistemicClass={e.epistemic_class}
                        size="sm"
                        showLabel={false}
                        tipPlacement="above"
                      />
                    )}
                  </span>
                </td>
                <td className="dim">{e.detail}</td>
              </tr>
            ))}
            {stats.recent_activity.length === 0 && (
              <tr>
                <td colSpan={4} className="empty-state">
                  No activity recorded yet.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </>
  )
}
