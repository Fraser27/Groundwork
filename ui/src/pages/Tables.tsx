import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, type Source, type TableSummary } from '../api'
import { getTenantId } from '../auth'
import { HELP } from '../epistemic'
import { fallback, MOCK_SOURCES, MOCK_TABLES } from '../mocks'
import EpistemicBadge from '../components/EpistemicBadge'
import FieldHelp from '../components/FieldHelp'
import { EmptyState, MockFlag, Spinner } from '../components/Shared'
import { fmtDateTime, fmtNum } from '../format'

export default function Tables() {
  const tenant = getTenantId()
  const [tables, setTables] = useState<TableSummary[]>([])
  const [sources, setSources] = useState<Source[]>([])
  const [filter, setFilter] = useState('')
  const [sourceFilter, setSourceFilter] = useState('__all__')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([
      fallback(api.listTables(tenant), MOCK_TABLES),
      fallback(api.listSources(tenant), MOCK_SOURCES),
    ])
      .then(([t, s]) => {
        setTables(t)
        setSources(s)
      })
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [tenant])

  const filtered = useMemo(() => {
    let out = tables
    if (sourceFilter !== '__all__') out = out.filter((t) => t.source_id === sourceFilter)
    if (filter.trim()) {
      const q = filter.toLowerCase()
      out = out.filter(
        (t) =>
          t.full_name.toLowerCase().includes(q) ||
          t.database.toLowerCase().includes(q) ||
          (t.description || '').toLowerCase().includes(q),
      )
    }
    return out
  }, [tables, filter, sourceFilter])

  const sourceName = (id: string) => sources.find((s) => s.source_id === id)?.name ?? id

  if (loading) return <Spinner />

  return (
    <>
      <div className="page-header">
        <div className="page-header-row">
          <div>
            <h2>Structured sources</h2>
            <p>
              Tables discovered in the data catalogue. Only the metadata enters the graph — rows never
              move, and are queried in place when a question needs them.
            </p>
          </div>
          <MockFlag />
        </div>
      </div>

      <div className="banner banner-info">
        <span>
          Everything on this page is{' '}
          <span title={HELP.epistemicClass}>
            <strong>declared</strong>
          </span>
          : a catalogue scan reported it, so there is nothing to be uncertain about and nothing to
          review. What a model reads into your documents is where judgement is needed.
        </span>
      </div>

      <div className="stats-grid">
        {sources.map((s) => (
          <div className="stat-card" key={s.source_id}>
            <div className="label">{s.kind}</div>
            <div className="value accent" style={{ fontSize: 17 }}>
              {s.name}
            </div>
            <div className="sub">
              {fmtNum(s.table_count)} tables &middot;{' '}
              <span
                style={{ color: s.status === 'connected' ? 'var(--green)' : 'var(--orange)' }}
              >
                {s.status}
              </span>
              <br />
              scanned {fmtDateTime(s.last_scanned_at)}
            </div>
          </div>
        ))}
      </div>

      <div className="toolbar">
        <div className="toolbar-field" style={{ flex: 1, minWidth: 260 }}>
          <label>Search</label>
          <input
            placeholder="Filter by table, database or description…"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            style={{ width: '100%' }}
          />
        </div>
        <div className="toolbar-field">
          <label>Source</label>
          <select value={sourceFilter} onChange={(e) => setSourceFilter(e.target.value)}>
            <option value="__all__">All sources</option>
            {sources.map((s) => (
              <option key={s.source_id} value={s.source_id}>
                {s.name}
              </option>
            ))}
          </select>
        </div>
        <div className="toolbar-field toolbar-spacer">
          <label>&nbsp;</label>
          <span className="search-count">
            {filtered.length} of {tables.length}
          </span>
        </div>
      </div>

      <div className="card">
        <table className="data-table data-table-hover">
          <thead>
            <tr>
              <th>Table</th>
              <th>Source</th>
              <th className="num">Rows</th>
              <th>
                How reached
                <FieldHelp text={HELP.epistemicClass} />
              </th>
              <th>Description</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((t) => (
              <tr
                key={t.full_name}
                onClick={() => {
                  window.location.href = `/tables/${encodeURIComponent(t.full_name)}`
                }}
              >
                <td>
                  <Link to={`/tables/${encodeURIComponent(t.full_name)}`}>{t.name}</Link>
                  <div className="dim" style={{ fontSize: 11.5 }}>
                    <code>{t.database}</code>
                  </div>
                </td>
                <td className="nowrap dim">{sourceName(t.source_id)}</td>
                <td className="num">{fmtNum(t.row_count)}</td>
                <td>
                  <EpistemicBadge epistemicClass={t.epistemic_class} size="sm" />
                </td>
                <td className="dim">{t.description || '—'}</td>
              </tr>
            ))}
            {filtered.length === 0 && (
              <tr>
                <td colSpan={5}>
                  <EmptyState title="No tables match">
                    Add a source in Admin and run a scan to populate the catalogue.
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
