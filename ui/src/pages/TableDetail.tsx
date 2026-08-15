import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api, type TableDetail as TableDetailType } from '../api'
import { getTenantId } from '../auth'
import { HELP } from '../epistemic'
import EpistemicBadge from '../components/EpistemicBadge'
import FieldHelp from '../components/FieldHelp'
import { EmptyState, ErrorState, Spinner } from '../components/Shared'
import { fmtDateTime, fmtNum } from '../format'

export default function TableDetail() {
  const tenant = getTenantId()
  const { name } = useParams<{ name: string }>()
  const [table, setTable] = useState<TableDetailType | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [reloadKey, setReloadKey] = useState(0)

  useEffect(() => {
    if (!name) return
    api
      .getTable(tenant, name)
      .then((t) => {
        setTable(t)
        setError('')
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false))
  }, [tenant, name, reloadKey])

  const retry = () => {
    setLoading(true)
    setReloadKey((k) => k + 1)
  }

  if (loading) return <Spinner />
  if (error || !table)
    return (
      <>
        <Link to="/tables" className="back-link">
          Back to structured sources
        </Link>
        {error ? (
          <ErrorState
            title="Could not load this table"
            detail={error}
            onRetry={retry}
          />
        ) : (
          <EmptyState title="Table not in the catalogue">
            <code>{name}</code> was not found. It may have been dropped, or the catalogue may need
            rescanning from Admin.
          </EmptyState>
        )}
      </>
    )

  return (
    <>
      <Link to="/tables" className="back-link">
        ← Back to structured sources
      </Link>

      <div className="page-header">
        <div className="page-header-row">
          <div>
            <h2>{table.name}</h2>
            <p>{table.description || 'No description recorded.'}</p>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="detail-grid-3">
          <div className="detail-field">
            <div className="label">Full name</div>
            <div className="value">
              <code>{table.full_name}</code>
            </div>
          </div>
          <div className="detail-field">
            <div className="label">Database</div>
            <div className="value">
              <span className="tag tag-blue">{table.database}</span>
            </div>
          </div>
          <div className="detail-field">
            <div className="label">Rows</div>
            <div className="value">{fmtNum(table.row_count)}</div>
          </div>
          <div className="detail-field">
            <div className="label">
              How reached
              <FieldHelp text={HELP.epistemicClass} />
            </div>
            <div className="value">
              <EpistemicBadge epistemicClass={table.epistemic_class} size="sm" />
            </div>
          </div>
          <div className="detail-field">
            <div className="label">
              Method
              <FieldHelp text={HELP.method} />
            </div>
            <div className="value">
              <code>{table.method}</code>
            </div>
          </div>
          <div className="detail-field">
            <div className="label">Last scanned</div>
            <div className="value">{fmtDateTime(table.scanned_at)}</div>
          </div>
        </div>
        <p className="card-note" style={{ borderTop: '1px solid var(--border)', paddingTop: 12 }}>
          Rows are not copied into LexGraph. Only this metadata is recorded, and a query reads the
          table in place at the moment it is asked — so nothing here can go stale relative to the
          source.
        </p>
      </div>

      <div className="card">
        <div className="card-header">
          <h3>Columns ({table.columns.length})</h3>
          <Link to="/metrics" className="btn btn-ghost btn-sm">
            Define a metric on this table
          </Link>
        </div>
        <table className="data-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Type</th>
              <th>Description</th>
              <th>Key</th>
            </tr>
          </thead>
          <tbody>
            {table.columns.map((c) => (
              <tr key={c.name}>
                <td>
                  <strong>{c.name}</strong>
                </td>
                <td>
                  <code style={{ fontSize: 11.5 }}>{c.data_type}</code>
                </td>
                <td className="dim">{c.description || '-'}</td>
                <td className="nowrap">
                  {c.is_primary_key && <span className="tag tag-green">primary</span>}{' '}
                  {c.is_partition && (
                    <span
                      className="tag tag-orange"
                      title="A partition key. Often a string rather than a real date, which is why it cannot be used as a metric's time axis."
                    >
                      partition
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )
}
