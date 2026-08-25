import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api, type TableDetail as TableDetailType } from '../api'
import { canReview, getTenantId, isPlatformAdmin } from '../auth'
import { HELP } from '../epistemic'
import EditableDescription from '../components/EditableDescription'
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
  const [busy, setBusy] = useState('')
  const admin = isPlatformAdmin()
  const reviewer = canReview()

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

  /** Re-read after any write. The server owns precedence, so the page asks rather than guesses. */
  const reload = () => setReloadKey((k) => k + 1)

  const describe = async (text: string, column?: string) => {
    if (!name) return
    await api.setDescription(tenant, name, text, column)
    reload()
  }

  const generate = async () => {
    if (!name) return
    setBusy('')
    try {
      await api.enrichCatalog(tenant, [name])
      // The run is a background task, so there is nothing to await. Said plainly rather than
      // spinning: a spinner that stops meaning anything is worse than a sentence.
      setBusy('Descriptions requested. Reload in a moment to review what the model proposed.')
    } catch (e) {
      setBusy((e as Error).message)
    }
  }

  const approveAll = async () => {
    if (!name) return
    setBusy('')
    try {
      const r = await api.approveTableEnrichment(tenant, name)
      const failed = Object.keys(r.failed ?? {}).length
      setBusy(
        failed > 0
          ? `Approved ${r.approved} of ${r.pending}. ${failed} could not be approved.`
          : `Approved ${r.approved}.`,
      )
      reload()
    } catch (e) {
      setBusy((e as Error).message)
    }
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
          <div style={{ flex: 1 }}>
            <h2>{table.name}</h2>
            <EditableDescription
              value={table.description ?? ''}
              source={table.description_source}
              pending={table.pending_description}
              canEdit={admin}
              canApprove={reviewer}
              onSave={(text) => describe(text)}
              onApprove={approveAll}
            />
          </div>
        </div>
      </div>

      {busy && (
        <div className="banner banner-info">
          <span>{busy}</span>
        </div>
      )}

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
          Rows are not copied into Groundwork. Only this metadata is recorded, and a query reads the
          table in place at the moment it is asked, so nothing here can go stale relative to the
          source.
        </p>
      </div>

      <div className="card">
        <div className="card-header">
          <h3>Columns ({table.columns.length})</h3>
          <div className="card-header-actions">
            {(table.pending_enrichment ?? 0) > 0 && reviewer && (
              <button className="btn btn-approve btn-sm" onClick={approveAll}>
                Approve {table.pending_enrichment} proposed
              </button>
            )}
            {admin && (
              <button
                className="btn btn-ghost btn-sm"
                onClick={generate}
                title="Ask the configured enrichment model to describe this table and its columns. Each description waits for review before any query uses it."
              >
                Generate descriptions
              </button>
            )}
            <Link to="/metrics" className="btn btn-ghost btn-sm">
              Define a metric on this table
            </Link>
          </div>
        </div>
        <table className="data-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Type</th>
              <th>
                Description
                <FieldHelp text="Given to the model that writes SQL for questions no approved metric covers, so a column described as a matter status produces a better query than one called mtr_stat_cd. A description written here outranks a model's, and a model's is only used once approved." />
              </th>
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
                <td>
                  <EditableDescription
                    value={c.description ?? ''}
                    source={c.description_source}
                    pending={c.pending_description}
                    canEdit={admin}
                    canApprove={reviewer}
                    onSave={(text) => describe(text, c.name)}
                    onApprove={approveAll}
                    placeholder="-"
                  />
                </td>
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
