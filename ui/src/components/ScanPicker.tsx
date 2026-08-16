/**
 * Choose which Glue databases to read into the graph.
 *
 * Scanning used to take everything the task role could see. A firm's catalog holds other teams'
 * databases, so that made the Tables page unusable and the graph misleading -- the point of a
 * governed layer is that it holds what somebody chose to govern, and "everything AWS would let me
 * read" is not a choice.
 *
 * Table counts are shown because names alone do not answer "which of these do I want", and an
 * empty database is almost always the wrong pick. Databases already in the graph are marked, since
 * a re-scan replaces their tables rather than adding to them.
 */

import { useEffect, useState } from 'react'

import { api, type GlueDatabase } from '../api'
import { EmptyState, ErrorState, Spinner } from './Shared'

export default function ScanPicker({
  tenant,
  busy,
  onCancel,
  onScan,
}: {
  tenant: string
  busy: boolean
  onCancel: () => void
  onScan: (databases: string[]) => void
}) {
  const [databases, setDatabases] = useState<GlueDatabase[] | null>(null)
  const [error, setError] = useState('')
  const [picked, setPicked] = useState<Set<string>>(new Set())

  useEffect(() => {
    api
      .glueDatabases(tenant)
      .then((r) => {
        setDatabases(r.databases)
        // Pre-ticked: a re-scan of what is already in the graph is the common case, and the
        // alternative is making somebody re-pick the same set every time.
        setPicked(new Set(r.databases.filter((d) => d.scanned).map((d) => d.name)))
        setError('')
      })
      .catch((e: Error) => {
        setDatabases([])
        setError(e.message)
      })
  }, [tenant])

  const toggle = (name: string) => {
    const next = new Set(picked)
    if (next.has(name)) next.delete(name)
    else next.add(name)
    setPicked(next)
  }

  return (
    <div className="modal-overlay" onClick={onCancel}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h3>Which databases should the graph know about?</h3>
        <p className="modal-sub">
          Scanning records a database's tables and columns as facts declared by a system of record.
          Nothing here is read until you choose it, and rows never leave the source: only the
          schema enters the graph.
        </p>

        {error && <ErrorState title="Could not list the catalog" detail={error} />}

        {databases === null ? (
          <Spinner />
        ) : databases.length === 0 && !error ? (
          <EmptyState title="No databases visible">
            The task role can see no Glue databases in this region. Create one, or check that the
            catalog is in the region this deployment runs in.
          </EmptyState>
        ) : (
          <div className="form-group">
            <div className="scan-db-list">
              {databases.map((d) => (
                <label key={d.name} className="checkbox-row scan-db-row">
                  <input
                    type="checkbox"
                    checked={picked.has(d.name)}
                    onChange={() => toggle(d.name)}
                  />
                  <span>
                    <strong className="mono">{d.name}</strong>
                    <span className="dim" style={{ display: 'block', fontSize: 11.5 }}>
                      {d.error
                        ? `could not be read: ${d.error}`
                        : d.table_count === null
                          ? 'table count unavailable'
                          : `${d.table_count} table${d.table_count === 1 ? '' : 's'}`}
                      {d.scanned ? ' · already in the graph' : ''}
                    </span>
                  </span>
                </label>
              ))}
            </div>
            {picked.size === 0 && (
              <p className="hint">
                Nothing selected. Scanning with no database chosen would read the whole catalog,
                which is what this screen exists to avoid.
              </p>
            )}
          </div>
        )}

        <div className="modal-actions">
          <button className="btn btn-ghost" onClick={onCancel}>
            Cancel
          </button>
          <button
            className="btn btn-primary"
            disabled={busy || picked.size === 0}
            onClick={() => onScan([...picked])}
          >
            {busy ? 'Scanning…' : `Scan ${picked.size} database${picked.size === 1 ? '' : 's'}`}
          </button>
        </div>
      </div>
    </div>
  )
}
