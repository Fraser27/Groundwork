/**
 * Metrics — governed metric definitions.
 *
 * The point the page has to make: these compile to SQL deterministically. The
 * same question always produces the same query, and no model is involved. That
 * is why a metric is worth the effort of defining, and why approving one is a
 * governance act rather than a save button.
 */

import { useEffect, useMemo, useState } from 'react'
import { api, type Metric, type TableSummary } from '../api'
import { getTenantId } from '../auth'
import { HELP } from '../epistemic'
import FieldHelp from '../components/FieldHelp'
import { EmptyState, ErrorState, Spinner, Toast } from '../components/Shared'
import { fmtDateTime } from '../format'

interface Form {
  metric_id: string
  name: string
  definition: string
  expression: string
  source_table: string
  grain: string
  time_grain_column: string
  time_grains: string
  aggregation: Metric['aggregation']
  filters: string
  synonyms: string
}

const EMPTY: Form = {
  metric_id: '',
  name: '',
  definition: '',
  expression: '',
  source_table: '',
  grain: '',
  time_grain_column: '',
  time_grains: 'month, quarter, year',
  aggregation: 'additive',
  filters: '',
  synonyms: '',
}

const toForm = (m: Metric): Form => ({
  metric_id: m.metric_id,
  name: m.name,
  definition: m.definition,
  expression: m.expression,
  source_table: m.source_table,
  grain: m.grain.join(', '),
  time_grain_column: m.time_grain_column || '',
  time_grains: m.time_grains.join(', '),
  aggregation: m.aggregation,
  filters: m.filters.join('\n'),
  synonyms: m.synonyms.join(', '),
})

const list = (s: string) =>
  s
    .split(',')
    .map((x) => x.trim())
    .filter(Boolean)

const fromForm = (f: Form): Partial<Metric> => ({
  metric_id: f.metric_id,
  name: f.name,
  definition: f.definition,
  expression: f.expression,
  source_table: f.source_table,
  grain: list(f.grain),
  time_grain_column: f.time_grain_column || null,
  time_grains: list(f.time_grains),
  aggregation: f.aggregation,
  filters: f.filters
    .split('\n')
    .map((x) => x.trim())
    .filter(Boolean),
  synonyms: list(f.synonyms),
})

const AGGREGATION_HELP: Record<Metric['aggregation'], string> = {
  additive: 'Safe to sum across any period and any dimension. Fees billed behave this way.',
  semi_additive:
    'A balance. It may be summed across dimensions but not across time, adding month-end work in progress across twelve months produces a meaningless number.',
  non_additive:
    'Never summable. A distinct count of open matters cannot be added across periods, because the same matter appears in several of them.',
}

export default function Metrics() {
  const tenant = getTenantId()
  const [metrics, setMetrics] = useState<Metric[]>([])
  const [tables, setTables] = useState<TableSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [filter, setFilter] = useState('')
  const [modal, setModal] = useState(false)
  const [editing, setEditing] = useState<Metric | null>(null)
  const [form, setForm] = useState<Form>(EMPTY)
  const [saving, setSaving] = useState(false)
  const [sql, setSql] = useState<Record<string, string>>({})
  const [toast, setToast] = useState<{ msg: string; type: string } | null>(null)

  const showToast = (msg: string, type = 'success') => {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 4000)
  }

  const load = () => {
    Promise.all([api.listMetrics(tenant), api.listTables(tenant)])
      .then(([m, t]) => {
        setMetrics(m)
        setTables(t)
        setError('')
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false))
  }

  useEffect(load, [tenant])

  const filtered = useMemo(() => {
    if (!filter.trim()) return metrics
    const q = filter.toLowerCase()
    return metrics.filter(
      (m) =>
        m.name.toLowerCase().includes(q) ||
        m.metric_id.toLowerCase().includes(q) ||
        m.definition.toLowerCase().includes(q) ||
        m.expression.toLowerCase().includes(q) ||
        m.synonyms.some((s) => s.toLowerCase().includes(q)),
    )
  }, [metrics, filter])

  const save = async () => {
    setSaving(true)
    try {
      const body = fromForm(form)
      if (editing) {
        await api.updateMetric(tenant, editing.metric_id, body)
        showToast(`Updated ${form.name}. Saved as a draft, approve it to make it answerable.`)
      } else {
        await api.createMetric(tenant, body)
        showToast(`Created ${form.name} as a draft.`)
      }
      setModal(false)
      load()
    } catch (e) {
      showToast((e as Error).message.replace(/^\d+:\s*/, ''), 'error')
    } finally {
      setSaving(false)
    }
  }

  const setStatus = async (m: Metric, status: Metric['status']) => {
    if (status === 'approved') {
      if (
        !confirm(
          `Approve ${m.name}?\n\nOnce approved, questions that match it are answered by compiling this definition to SQL, with no model involved. That makes the definition itself the governance surface.`,
        )
      )
        return
    }
    try {
      await api.setMetricStatus(tenant, m.metric_id, status)
      showToast(`${m.name} ${status}`)
      load()
    } catch (e) {
      // Never optimistic: approval is the governance act, so a failed write must not look done.
      showToast(
        `Could not ${status === 'approved' ? 'approve' : 'change'} ${m.name}: ${(e as Error).message.replace(/^\d+:\s*/, '')}`,
        'error',
      )
    }
  }

  const toggleSql = async (m: Metric) => {
    if (sql[m.metric_id]) {
      setSql((s) => {
        const next = { ...s }
        delete next[m.metric_id]
        return next
      })
      return
    }
    try {
      const res = await api.compileMetric(tenant, m.metric_id)
      setSql((s) => ({ ...s, [m.metric_id]: res.sql }))
    } catch (e) {
      showToast(
        `Could not compile ${m.name}: ${(e as Error).message.replace(/^\d+:\s*/, '')}`,
        'error',
      )
    }
  }

  if (loading) return <Spinner />

  return (
    <>
      <div className="page-header">
        <div className="page-header-row">
          <div>
            <h2>Governed metrics</h2>
            <p>
              A metric here is a definition, not a query. When a question matches one, the SQL is
              compiled from this definition — the same question always yields the same query, and no
              language model writes any part of it. That is why the number is defensible.
            </p>
          </div>
        </div>
      </div>

      {error && <ErrorState title="Could not load metrics" detail={error} onRetry={load} />}

      <div className="banner banner-info">
        <span>
          <strong>Approving a metric is a governance decision.</strong> A draft is inert. An approved
          metric becomes the single sanctioned way to compute that figure, so anyone asking the
          question gets the same answer, phrased the same way, from the same SQL.
        </span>
      </div>

      <div className="toolbar">
        <button
          className="btn btn-primary"
          onClick={() => {
            setEditing(null)
            setForm(EMPTY)
            setModal(true)
          }}
        >
          New metric
        </button>
        <div className="toolbar-field" style={{ flex: 1, minWidth: 240 }}>
          <label>&nbsp;</label>
          <input
            placeholder="Filter by name, id, definition or synonym…"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            style={{ width: '100%' }}
          />
        </div>
        <div className="toolbar-field">
          <label>&nbsp;</label>
          <span className="search-count">
            {filtered.length} of {metrics.length}
          </span>
        </div>
      </div>

      <div className="card">
        <table className="data-table">
          <thead>
            <tr>
              <th>Metric</th>
              <th>Expression</th>
              <th>Source</th>
              <th>
                Time grains
                <FieldHelp text={HELP.timeGrain} />
              </th>
              <th>
                Additivity
                <FieldHelp text={HELP.additivity} />
              </th>
              <th>Status</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {filtered.map((m) => (
              <>
                <tr key={m.metric_id}>
                  <td>
                    <strong>{m.name}</strong>
                    <div className="dim" style={{ fontSize: 11.5 }}>
                      <code>{m.metric_id}</code> v{m.version}
                    </div>
                    <div className="dim" style={{ fontSize: 12, marginTop: 3, maxWidth: 280 }}>
                      {m.definition}
                    </div>
                    {m.synonyms.length > 0 && (
                      <div style={{ marginTop: 5, display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                        {m.synonyms.map((s) => (
                          <span key={s} className="tag tag-neutral" title="A question using this wording matches this metric.">
                            {s}
                          </span>
                        ))}
                      </div>
                    )}
                  </td>
                  <td>
                    <code style={{ fontSize: 11.5 }}>{m.expression}</code>
                  </td>
                  <td>
                    <span className="tag tag-blue tag-mono">{m.source_table}</span>
                    {m.grain.length > 0 && (
                      <div className="dim" style={{ fontSize: 11, marginTop: 4 }}>
                        by {m.grain.join(', ')}
                      </div>
                    )}
                  </td>
                  <td className="nowrap">
                    {m.time_grains.map((g) => (
                      <span key={g} className="tag tag-purple" style={{ marginRight: 4 }}>
                        {g}
                      </span>
                    ))}
                    {m.time_grain_column && (
                      <div className="dim" style={{ fontSize: 11, marginTop: 4 }}>
                        on <code>{m.time_grain_column}</code>
                      </div>
                    )}
                  </td>
                  <td>
                    <span
                      className={`tag ${
                        m.aggregation === 'additive'
                          ? 'tag-green'
                          : m.aggregation === 'semi_additive'
                            ? 'tag-orange'
                            : 'tag-red'
                      }`}
                      title={AGGREGATION_HELP[m.aggregation]}
                    >
                      {m.aggregation.replace('_', '-')}
                    </span>
                  </td>
                  <td>
                    <span
                      className={`tag ${
                        m.status === 'approved'
                          ? 'tag-green'
                          : m.status === 'deprecated'
                            ? 'tag-red'
                            : 'tag-orange'
                      }`}
                      title={
                        m.status === 'approved'
                          ? 'Answerable. Questions matching this metric compile to its SQL.'
                          : m.status === 'draft'
                            ? 'Not answerable yet. A draft is never used to answer a question.'
                            : 'Withdrawn. Kept for the audit trail but no longer used.'
                      }
                    >
                      {m.status}
                    </span>
                    {m.updated_by && (
                      <div className="dim" style={{ fontSize: 11, marginTop: 4 }}>
                        {m.updated_by.split('@')[0]} &middot; {fmtDateTime(m.updated_at)}
                      </div>
                    )}
                  </td>
                  <td className="nowrap">
                    <button
                      className={`btn btn-sm ${sql[m.metric_id] ? 'btn-primary' : 'btn-ghost'}`}
                      onClick={() => toggleSql(m)}
                      style={{ marginRight: 5 }}
                      title="Compile this definition to SQL. Deterministic, no model is invoked."
                    >
                      SQL
                    </button>
                    <button
                      className="btn btn-ghost btn-sm"
                      style={{ marginRight: 5 }}
                      onClick={() => {
                        setEditing(m)
                        setForm(toForm(m))
                        setModal(true)
                      }}
                    >
                      Edit
                    </button>
                    {m.status !== 'approved' ? (
                      <button className="btn btn-approve btn-sm" onClick={() => setStatus(m, 'approved')}>
                        Approve
                      </button>
                    ) : (
                      <button className="btn btn-ghost btn-sm" onClick={() => setStatus(m, 'deprecated')}>
                        Deprecate
                      </button>
                    )}
                  </td>
                </tr>
                {sql[m.metric_id] && (
                  <tr key={`${m.metric_id}-sql`}>
                    <td colSpan={7} style={{ padding: 0 }}>
                      <pre className="code-block" style={{ margin: 0, borderRadius: 0, border: 'none' }}>
                        {sql[m.metric_id]}
                      </pre>
                    </td>
                  </tr>
                )}
              </>
            ))}
            {filtered.length === 0 && (
              <tr>
                <td colSpan={7}>
                  <EmptyState title={metrics.length === 0 ? 'No metrics defined' : 'No metrics match'}>
                    {metrics.length === 0
                      ? 'A governed metric turns a recurring question into a fixed, auditable calculation. Use New metric to define the first one.'
                      : 'Clear the filter to see every metric.'}
                  </EmptyState>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {modal && (
        <div className="modal-overlay" onClick={() => setModal(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h3>{editing ? `Edit ${editing.name}` : 'New metric'}</h3>
            <p className="modal-sub">
              Saved as a draft. A draft never answers a question until it is approved.
            </p>

            <div className="form-row">
              <div className="form-group">
                <label>
                  Metric id
                  <FieldHelp text="A stable identifier, e.g. m_007. Fixed after creation, because queries and the audit trail reference it." />
                </label>
                <input
                  value={form.metric_id}
                  disabled={!!editing}
                  onChange={(e) => setForm({ ...form, metric_id: e.target.value })}
                  placeholder="m_005"
                />
              </div>
              <div className="form-group">
                <label>Name</label>
                <input
                  value={form.name}
                  onChange={(e) => setForm({ ...form, name: e.target.value })}
                  placeholder="fees_billed"
                />
              </div>
            </div>

            <div className="form-group">
              <label>
                Definition
                <FieldHelp text="What this number means in business terms, including what it excludes. This is what a reader sees when they ask what the figure represents." />
              </label>
              <input
                value={form.definition}
                onChange={(e) => setForm({ ...form, definition: e.target.value })}
                placeholder="Total fees invoiced, excluding disbursements and VAT."
              />
            </div>

            <div className="form-group">
              <label>
                SQL expression
                <FieldHelp text="The aggregate that computes the figure, e.g. SUM(i.fee_amount). This is the only SQL anyone writes; the surrounding query is compiled." />
              </label>
              <input
                className="input-mono"
                value={form.expression}
                onChange={(e) => setForm({ ...form, expression: e.target.value })}
                placeholder="SUM(i.fee_amount)"
              />
            </div>

            <div className="form-group">
              <label>Source table</label>
              <select
                value={form.source_table}
                onChange={(e) => setForm({ ...form, source_table: e.target.value })}
              >
                <option value="">Select a table…</option>
                {tables.map((t) => (
                  <option key={t.full_name} value={t.full_name}>
                    {t.full_name}
                  </option>
                ))}
              </select>
              <p className="hint">
                Rows stay where they are. The compiled query reads this table in place.
              </p>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label>
                  Grain
                  <FieldHelp text="The dimensions this metric may be broken down by. Anything not listed here cannot be used to slice it, which stops a figure being cut in a way its definition does not support." />
                </label>
                <input
                  value={form.grain}
                  onChange={(e) => setForm({ ...form, grain: e.target.value })}
                  placeholder="matter_id, practice_area"
                />
              </div>
              <div className="form-group">
                <label>
                  Additivity
                  <FieldHelp text={HELP.additivity} />
                </label>
                <select
                  value={form.aggregation}
                  onChange={(e) =>
                    setForm({ ...form, aggregation: e.target.value as Metric['aggregation'] })
                  }
                >
                  <option value="additive">Additive</option>
                  <option value="semi_additive">Semi-additive (a balance)</option>
                  <option value="non_additive">Non-additive</option>
                </select>
                <p className="hint">{AGGREGATION_HELP[form.aggregation]}</p>
              </div>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label>
                  Time axis column
                  <FieldHelp text="The real date or timestamp column the time grain applies to. A partition string that merely looks like a date must not be used here, or the grain it is meant to guard is unenforceable." />
                </label>
                <input
                  className="input-mono"
                  value={form.time_grain_column}
                  onChange={(e) => setForm({ ...form, time_grain_column: e.target.value })}
                  placeholder="issued_date"
                />
              </div>
              <div className="form-group">
                <label>
                  Permitted time grains
                  <FieldHelp text={HELP.timeGrain} />
                </label>
                <input
                  value={form.time_grains}
                  onChange={(e) => setForm({ ...form, time_grains: e.target.value })}
                  placeholder="month, quarter, year"
                />
                <p className="hint">
                  Anything omitted is refused rather than silently approximated.
                </p>
              </div>
            </div>

            <div className="form-group">
              <label>
                Fixed filters
                <FieldHelp text="Conditions always applied, one per line. These are part of the definition, a caller cannot remove them, so the figure cannot be quietly widened." />
              </label>
              <textarea
                className="input-mono"
                value={form.filters}
                onChange={(e) => setForm({ ...form, filters: e.target.value })}
                placeholder="i.status = 'ISSUED'"
              />
            </div>

            <div className="form-group">
              <label>
                Synonyms
                <FieldHelp text="Alternative wordings that should match this metric. Adding the words your firm actually uses is what keeps questions on the governed path instead of falling through to generated SQL." />
              </label>
              <input
                value={form.synonyms}
                onChange={(e) => setForm({ ...form, synonyms: e.target.value })}
                placeholder="billings, revenue, turnover"
              />
            </div>

            <div className="modal-actions">
              <button className="btn btn-ghost" onClick={() => setModal(false)}>
                Cancel
              </button>
              <button
                className="btn btn-primary"
                onClick={save}
                disabled={saving || !form.metric_id || !form.name || !form.expression}
              >
                {saving ? 'Saving…' : editing ? 'Save draft' : 'Create draft'}
              </button>
            </div>
          </div>
        </div>
      )}

      <Toast toast={toast} />
    </>
  )
}
