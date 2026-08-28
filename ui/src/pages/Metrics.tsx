/**
 * Metrics — governed metric definitions.
 *
 * The point the page has to make: these compile to SQL deterministically. The
 * same question always produces the same query, and no model is involved. That
 * is why a metric is worth the effort of defining, and why approving one is a
 * governance act rather than a save button.
 */

import { useEffect, useMemo, useState } from 'react'
import {
  api,
  type Column,
  type CompiledMetric,
  type Metric,
  type MetricJoin,
  type MetricParameter,
  type MetricType,
  type MetricWrite,
  type TableSummary,
} from '../api'
import { getTenantId, isPlatformAdmin } from '../auth'
import { HELP } from '../epistemic'
import FieldHelp from '../components/FieldHelp'
import { EmptyState, ErrorState, Spinner, Toast } from '../components/Shared'
import { fmtDateTime } from '../format'
import { fillUnit, useUnitLabel } from '../useUnitLabel'

/**
 * The definition as the modal holds it. Lists are strings only where the editor is a text field.
 *
 * Every field the API accepts has a home here, including the ones with no editor: this form is a
 * fetch-then-save round trip, so anything it cannot hold it destroys on save.
 */
interface Form {
  metric_id: string
  name: string
  definition: string
  expression: string
  type: MetricType
  source_table: string
  joins: MetricJoin[]
  base_metrics: string[]
  grain: string
  time_grain_column: string
  time_grains: string
  aggregation: Metric['aggregation']
  value_type: string
  unit: string
  format: string
  filters: string
  parameters: MetricParameter[]
  synonyms: string
  entity_columns: Record<string, string>
  owner: string
}

const EMPTY: Form = {
  metric_id: '',
  name: '',
  definition: '',
  expression: '',
  type: 'simple',
  source_table: '',
  joins: [],
  base_metrics: [],
  grain: '',
  time_grain_column: '',
  time_grains: 'month, quarter, year',
  aggregation: 'additive',
  value_type: 'number',
  unit: '',
  format: '',
  filters: '',
  parameters: [],
  synonyms: '',
  entity_columns: {},
  owner: '',
}

const VALUE_TYPES = ['number', 'currency', 'percent', 'count', 'duration']
// Mirrors JOIN_TYPES in src/metrics/models.py, minus the two that need no join columns.
const JOIN_TYPES = ['INNER', 'LEFT', 'RIGHT', 'FULL']
// Mirrors OPERATORS in src/metrics/models.py. A parameter declared with anything else is refused.
const OPERATORS = ['=', '!=', '>', '<', '>=', '<=', 'IN', 'NOT IN', 'LIKE', 'BETWEEN']
// Mirrors _TEMPORAL_TYPE_PREFIXES in src/metrics/compiler.py, which decides the same question.
const TEMPORAL_TYPE = /^(date|timestamp|time)/i

const toForm = (m: Metric): Form => ({
  metric_id: m.metric_id,
  name: m.name,
  definition: m.definition,
  expression: m.expression,
  type: m.type || 'simple',
  source_table: m.source_table,
  joins: (m.joins || []).map((j) => ({ ...j })),
  base_metrics: [...(m.base_metrics || [])],
  grain: m.grain.join(', '),
  time_grain_column: m.time_grain_column || '',
  time_grains: m.time_grains.join(', '),
  aggregation: m.aggregation,
  value_type: m.value_type || 'number',
  unit: m.unit || '',
  format: m.format || '',
  filters: m.filters.join('\n'),
  parameters: m.parameters.map((p) => ({ ...p, description: p.description || '' })),
  synonyms: m.synonyms.join(', '),
  entity_columns: { ...(m.entity_columns || {}) },
  owner: m.owner || '',
})

const list = (s: string) =>
  s
    .split(',')
    .map((x) => x.trim())
    .filter(Boolean)

const fromForm = (f: Form): MetricWrite => ({
  metric_id: f.metric_id,
  name: f.name,
  definition: f.definition,
  expression: f.expression,
  type: f.type,
  source_table: f.source_table,
  // Blank rows are the editor's own scaffolding, never something a stored definition contains.
  joins: f.joins.filter((j) => j.table || j.source_column || j.target_column),
  base_metrics: f.base_metrics,
  grain: list(f.grain),
  time_grain_column: f.time_grain_column,
  time_grains: list(f.time_grains),
  aggregation: f.aggregation,
  value_type: f.value_type,
  unit: f.unit,
  format: f.format,
  filters: f.filters
    .split('\n')
    .map((x) => x.trim())
    .filter(Boolean),
  parameters: f.parameters.filter((p) => p.column || p.description),
  synonyms: list(f.synonyms),
  entity_columns: f.entity_columns,
  owner: f.owner,
})

const AGGREGATION_HELP: Record<Metric['aggregation'], string> = {
  additive: 'Safe to sum across any period and any dimension. Fees billed behave this way.',
  semi_additive:
    'A balance. It may be summed across dimensions but not across time, adding month-end work in progress across twelve months produces a meaningless number.',
  non_additive:
    'Never summable. A distinct count of open {units} cannot be added across periods, because the same {unit} appears in several of them.',
}

/**
 * What the compiler noticed and did not refuse.
 *
 * Fan-out inflation over a join, a result that must not be summed again, base metrics in
 * different units. Each is a way the figure can be wrong while the SQL is perfectly valid, so
 * they belong beside the SQL rather than in a log nobody reads.
 */
function CompilerWarnings({ warnings }: { warnings: string[] }) {
  if (!warnings || warnings.length === 0) return null
  return (
    <div
      className="banner banner-warn"
      style={{ marginBottom: 0, borderRadius: 0, border: 'none' }}
    >
      <span>
        <strong>
          Compiles, with {warnings.length === 1 ? 'a caveat' : `${warnings.length} caveats`}.
        </strong>{' '}
        {warnings.length === 1 ? (
          warnings[0]
        ) : (
          <ul style={{ margin: '4px 0 0', paddingLeft: 18 }}>
            {warnings.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        )}
      </span>
    </div>
  )
}

const OTHER = '__other__'

/**
 * A column reference: the scanned schema when there is one, free text when there is not.
 *
 * The catalog is allowed to be empty, so a bare select would make a metric unauthorable until
 * somebody runs a scan. A name the schema does not contain stays visible and selected rather
 * than being dropped, because a column the scan has not seen is not necessarily wrong.
 */
function ColumnField({
  value,
  columns,
  onChange,
  empty,
  placeholder,
  withType,
}: {
  value: string
  columns: Column[]
  onChange: (v: string) => void
  empty: string
  placeholder: string
  withType?: boolean
}) {
  const [typed, setTyped] = useState(false)
  if (columns.length === 0 || typed) {
    return (
      <div className="field-row">
        <input
          className="input-mono"
          value={value}
          placeholder={placeholder}
          onChange={(e) => onChange(e.target.value)}
        />
        {columns.length > 0 && (
          <button className="btn btn-ghost btn-sm" onClick={() => setTyped(false)}>
            List
          </button>
        )}
      </div>
    )
  }
  const unlisted = value !== '' && !columns.some((c) => c.name === value)
  return (
    <select
      value={value}
      onChange={(e) => (e.target.value === OTHER ? setTyped(true) : onChange(e.target.value))}
    >
      <option value="">{empty}</option>
      {columns.map((c) => (
        <option key={c.name} value={c.name}>
          {withType ? `${c.name} (${c.data_type})` : c.name}
        </option>
      ))}
      {unlisted && <option value={value}>{value} (not in the scanned schema)</option>}
      <option value={OTHER}>Type a column name…</option>
    </select>
  )
}

export default function Metrics() {
  const tenant = getTenantId()
  const unit = useUnitLabel()
  // Presentation only. `require_admin` answers 403 whatever the browser drew, so hiding the
  // control is about not offering an action that will fail rather than about preventing it.
  const admin = isPlatformAdmin()
  const [metrics, setMetrics] = useState<Metric[]>([])
  const [tables, setTables] = useState<TableSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [filter, setFilter] = useState('')
  const [modal, setModal] = useState(false)
  const [editing, setEditing] = useState<Metric | null>(null)
  const [form, setForm] = useState<Form>(EMPTY)
  const [saving, setSaving] = useState(false)
  const [seeding, setSeeding] = useState(false)
  const [sql, setSql] = useState<Record<string, CompiledMetric>>({})
  const [preview, setPreview] = useState<CompiledMetric | null>(null)
  const [previewing, setPreviewing] = useState(false)
  const [toast, setToast] = useState<{ msg: string; type: string } | null>(null)
  const [columns, setColumns] = useState<Record<string, Column[]>>({})
  const [showJoins, setShowJoins] = useState(false)
  const [showParams, setShowParams] = useState(false)

  const showToast = (msg: string, type = 'success') => {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 4000)
  }

  // Any edit discards the compiled SQL on screen. Leaving it up would show an author SQL that no
  // longer matches the definition they are about to save, which is worse than showing none.
  const update = (patch: Partial<Form>) => {
    setForm((f) => ({ ...f, ...patch }))
    setPreview(null)
  }

  const openModal = (m: Metric | null) => {
    setEditing(m)
    setForm(m ? toForm(m) : EMPTY)
    setPreview(null)
    setShowJoins((m?.joins?.length ?? 0) > 0)
    setShowParams((m?.parameters?.length ?? 0) > 0)
    setModal(true)
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

  // Joined so the effect below depends on the set of tables in play, not on every keystroke.
  const wanted = useMemo(
    () => [...new Set([form.source_table, ...form.joins.map((j) => j.table)])].join('\n'),
    [form.source_table, form.joins],
  )

  useEffect(() => {
    const missing = wanted.split('\n').filter((t) => t && !(t in columns))
    if (missing.length === 0) return
    let live = true
    // A failure caches an empty column list, which is what makes the pickers fall back to text
    // rather than blocking on a catalog that may never have been scanned.
    Promise.all(
      missing.map((t) =>
        api
          .getTable(tenant, t)
          .then((d) => [t, d.columns || []] as const)
          .catch(() => [t, [] as Column[]] as const),
      ),
    ).then((pairs) => {
      if (live) setColumns((c) => ({ ...c, ...Object.fromEntries(pairs) }))
    })
    return () => {
      live = false
    }
  }, [wanted, columns, tenant])

  const sourceColumns = columns[form.source_table] || []
  // The governed time axis has to be a real date or timestamp. A month partition string that
  // looks like one would make the declared grain unenforceable, which is the mistake
  // _check_time_axis_bypass exists to catch, so the picker cannot express it.
  const temporalColumns = sourceColumns.filter((c) => TEMPORAL_TYPE.test(c.data_type || ''))
  const derived = form.type === 'derived'
  // The shape the model requires of each type. Checked here only to spare the author a 422.
  const shapeReady = derived ? form.base_metrics.length > 0 : !!form.source_table

  const databases = useMemo(() => [...new Set(tables.map((t) => t.database))].sort(), [tables])

  // Only simple metrics: the compiler refuses a derived metric composed of derived metrics, so
  // offering one here would produce a definition that cannot compile.
  const baseCandidates = metrics.filter(
    (m) => (m.type || 'simple') === 'simple' && m.metric_id !== form.metric_id,
  )
  const baseTokens = (m: Metric) => [m.metric_id, m.name]
  const unresolvedBases = form.base_metrics.filter(
    (b) => !baseCandidates.some((m) => baseTokens(m).includes(b)),
  )
  const unknownGrain =
    sourceColumns.length > 0
      ? list(form.grain).filter((g) => !sourceColumns.some((c) => c.name === g))
      : []

  // By name, not id: a derived metric's expression refers to its bases by their output name, so
  // the two halves of the definition stay written in the same vocabulary.
  const toggleBase = (m: Metric) => {
    const tokens = baseTokens(m)
    update({
      base_metrics: form.base_metrics.some((b) => tokens.includes(b))
        ? form.base_metrics.filter((b) => !tokens.includes(b))
        : [...form.base_metrics, m.name],
    })
  }

  const patchJoin = (i: number, patch: Partial<MetricJoin>) =>
    update({ joins: form.joins.map((j, n) => (n === i ? { ...j, ...patch } : j)) })

  const patchParam = (i: number, patch: Partial<MetricParameter>) =>
    update({ parameters: form.parameters.map((p, n) => (n === i ? { ...p, ...patch } : p)) })

  const save = async () => {
    setSaving(true)
    try {
      const body = fromForm(form)
      const saved = editing
        ? await api.updateMetric(tenant, editing.metric_id, body)
        : await api.createMetric(tenant, body)
      // An author who skipped the preview still has to hear the caveats, so the count goes in
      // the toast and the SQL button carries the text.
      const caveats = saved.warnings?.length
        ? ` It compiles with ${saved.warnings.length === 1 ? 'a caveat' : `${saved.warnings.length} caveats`}, open SQL to read ${saved.warnings.length === 1 ? 'it' : 'them'}.`
        : ''
      showToast(
        editing
          ? `Updated ${form.name}. Saved as a draft, approve it to make it answerable.${caveats}`
          : `Created ${form.name} as a draft.${caveats}`,
      )
      setModal(false)
      load()
    } catch (e) {
      showToast((e as Error).message.replace(/^\d+:\s*/, ''), 'error')
    } finally {
      setSaving(false)
    }
  }

  /**
   * Compile the form as it stands, without saving it.
   *
   * The point of a deterministic compiler is that a human can read the SQL before the definition
   * is allowed to answer anything, and after saving is too late: an approved metric is already
   * the sanctioned answer. Refusals come back as the same 422 the save would give.
   */
  const runPreview = async () => {
    setPreviewing(true)
    try {
      setPreview(await api.previewMetric(tenant, fromForm(form)))
    } catch (e) {
      setPreview(null)
      showToast((e as Error).message.replace(/^\d+:\s*/, ''), 'error')
    } finally {
      setPreviewing(false)
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

  /**
   * Load the pack that ships for this tenant's ontology.
   *
   * Here rather than only in the API because an empty Metrics page gives a reader nothing to
   * read: a definition is easier to understand by editing one than by inventing one, and the
   * shapes worth studying -- a hard time-grain restriction, a ratio composed from two other
   * metrics -- are not the ones a first attempt at the form produces.
   *
   * Drafts, so nothing it loads answers a question until somebody approves it, and metrics
   * authored here are left alone.
   */
  const seed = async () => {
    setSeeding(true)
    try {
      const r = await api.seedMetrics(tenant)
      showToast(
        r.created === 0
          ? `Nothing to load: all ${r.skipped} example metrics are already here.`
          : `Loaded ${r.created} example metric${r.created === 1 ? '' : 's'} as drafts. They name a fictional company's tables, so read each one against your own catalog before approving it.`,
        r.created === 0 ? 'info' : 'success',
      )
      load()
    } catch (e) {
      showToast((e as Error).message.replace(/^\d+:\s*/, ''), 'error')
    } finally {
      setSeeding(false)
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
      setSql((s) => ({ ...s, [m.metric_id]: res }))
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
              compiled from this definition, the same question always yields the same query, and no
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
        <button className="btn btn-primary" onClick={() => openModal(null)}>
          New metric
        </button>
        {admin && (
          <button className="btn btn-ghost" onClick={seed} disabled={seeding}>
            {seeding ? 'Loading…' : 'Load examples'}
            <FieldHelp text="Loads the example metrics that ship for this tenant's ontology pack, as drafts. They name a fictional company's tables, so each one has to be read against your own catalog before it is approved. Anything you authored here is left alone." />
          </button>
        )}
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
        <div className="table-scroll">
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
                        title={fillUnit(AGGREGATION_HELP[m.aggregation], unit)}
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
                        onClick={() => openModal(m)}
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
                        <CompilerWarnings warnings={sql[m.metric_id].warnings} />
                        <pre className="code-block" style={{ margin: 0, borderRadius: 0, border: 'none' }}>
                          {sql[m.metric_id].sql}
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
      </div>

      {modal && (
        <div className="modal-overlay" onClick={() => setModal(false)}>
          <div className="modal modal-wide" onClick={(e) => e.stopPropagation()}>
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
                  onChange={(e) => update({ metric_id: e.target.value })}
                  placeholder="m_005"
                />
              </div>
              <div className="form-group">
                <label>Name</label>
                <input
                  value={form.name}
                  onChange={(e) => update({ name: e.target.value })}
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
                onChange={(e) => update({ definition: e.target.value })}
                placeholder="Total fees invoiced, excluding disbursements and VAT."
              />
            </div>

            <div className="form-group">
              <label>
                Type
                <FieldHelp text="A simple metric aggregates rows from one table. A derived metric composes other metrics, each recomputed from its own rows at the shared grain, which is what keeps a ratio correct at every level rather than averaging averages." />
              </label>
              <select
                value={form.type}
                onChange={(e) => update({ type: e.target.value as MetricType })}
              >
                <option value="simple">Simple, aggregates one table</option>
                <option value="derived">Derived, composes other metrics</option>
              </select>
            </div>

            {/* Kept for a derived metric that carries one, so hiding never strands a value. */}
            {(!derived || !!form.source_table) && (
              <div className="form-group">
                <label>Source table</label>
                <select
                  value={form.source_table}
                  onChange={(e) => update({ source_table: e.target.value })}
                >
                  <option value="">Select a table…</option>
                  {databases.map((db) => (
                    <optgroup key={db} label={db}>
                      {tables
                        .filter((t) => t.database === db)
                        .map((t) => (
                          <option key={t.full_name} value={t.full_name}>
                            {t.full_name}
                          </option>
                        ))}
                    </optgroup>
                  ))}
                </select>
                <p className="hint">
                  Rows stay where they are. The compiled query reads this table in place.
                </p>
              </div>
            )}

            {(!derived || form.joins.length > 0) && (
              <details
                className="form-section"
                open={showJoins}
                onToggle={(e) => setShowJoins(e.currentTarget.open)}
              >
                <summary>
                  Joins{form.joins.length > 0 ? ` (${form.joins.length})` : ''}
                  <FieldHelp text="Tables the compiled query joins to the source table so their columns can be used as dimensions. A join that multiplies rows inflates the figure, which is why the compiler reports fan-out as a caveat beside the SQL rather than refusing it." />
                </summary>
                <div className="metric-rows">
                  {form.joins.map((j, i) => (
                    <div key={i} className="metric-row">
                      <select
                        value={j.join_type}
                        onChange={(e) => patchJoin(i, { join_type: e.target.value })}
                        style={{ width: 88, flexShrink: 0 }}
                      >
                        {JOIN_TYPES.map((t) => (
                          <option key={t} value={t}>
                            {t}
                          </option>
                        ))}
                      </select>
                      <select
                        value={j.table}
                        onChange={(e) => patchJoin(i, { table: e.target.value })}
                        style={{ flex: 2 }}
                      >
                        <option value="">Join table…</option>
                        {databases.map((db) => (
                          <optgroup key={db} label={db}>
                            {tables
                              .filter((t) => t.database === db && t.full_name !== form.source_table)
                              .map((t) => (
                                <option key={t.full_name} value={t.full_name}>
                                  {t.name}
                                </option>
                              ))}
                          </optgroup>
                        ))}
                      </select>
                      <span className="op">ON</span>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <ColumnField
                          value={j.source_column}
                          columns={sourceColumns}
                          onChange={(v) => patchJoin(i, { source_column: v })}
                          empty="Source column…"
                          placeholder="source column"
                        />
                      </div>
                      <span className="op">=</span>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <ColumnField
                          value={j.target_column}
                          columns={columns[j.table] || []}
                          onChange={(v) => patchJoin(i, { target_column: v })}
                          empty="Joined column…"
                          placeholder="joined column"
                        />
                      </div>
                      <button
                        className="btn btn-danger btn-sm"
                        onClick={() => update({ joins: form.joins.filter((_, n) => n !== i) })}
                        title="Remove this join"
                      >
                        Remove
                      </button>
                    </div>
                  ))}
                </div>
                {form.joins.length === 0 && (
                  <p className="hint" style={{ marginTop: 0, marginBottom: 8 }}>
                    No joins. The compiled query reads the source table only.
                  </p>
                )}
                <button
                  className="btn btn-ghost btn-sm"
                  style={{ marginTop: form.joins.length > 0 ? 8 : 0 }}
                  onClick={() =>
                    update({
                      joins: [
                        ...form.joins,
                        { table: '', source_column: '', target_column: '', join_type: 'INNER' },
                      ],
                    })
                  }
                >
                  Add join
                </button>
              </details>
            )}

            {(derived || form.base_metrics.length > 0) && (
              <div className="form-group">
                <label>
                  Base metrics
                  <FieldHelp text="The metrics this one composes. Only simple metrics are offered: the compiler refuses a derived metric built from another derived metric, so that the SQL stays readable. Each base becomes a CTE recomputed at the shared grain." />
                </label>
                {baseCandidates.length === 0 ? (
                  <p className="hint">
                    No simple metrics to compose yet. Define one first, then come back.
                  </p>
                ) : (
                  <div className="base-metric-list">
                    {baseCandidates.map((m) => {
                      const on = form.base_metrics.some((b) => baseTokens(m).includes(b))
                      return (
                        <label key={m.metric_id} className="base-metric-option">
                          <input type="checkbox" checked={on} onChange={() => toggleBase(m)} />
                          <code>{m.name}</code>
                          <span className="dim" style={{ flex: 1, minWidth: 0 }}>
                            {m.definition}
                          </span>
                          {m.status !== 'approved' && (
                            <span className="tag tag-orange">{m.status}</span>
                          )}
                        </label>
                      )
                    })}
                  </div>
                )}
                {unresolvedBases.length > 0 && (
                  <p className="hint">
                    Referenced but not found here: <code>{unresolvedBases.join(', ')}</code>. Kept
                    as written, and the compiler refuses a reference it cannot resolve.
                  </p>
                )}
              </div>
            )}

            <div className="form-group">
              <label>
                {derived ? 'Expression over the base metrics' : 'SQL expression'}
                <FieldHelp
                  text={
                    derived
                      ? 'Arithmetic over the base metrics, referenced by their names, e.g. fees_billed / hours_recorded. The compiler emits each base as a CTE and applies this to the joined result, so no aggregate belongs here.'
                      : 'The aggregate that computes the figure, e.g. SUM(fee_amount). This is the only SQL anyone writes; the surrounding query is compiled. Name columns unqualified: the compiler assigns the table alias, so a prefix written here may not match the one it emits. Preview SQL shows exactly what will run.'
                  }
                />
              </label>
              <input
                className="input-mono"
                value={form.expression}
                onChange={(e) => update({ expression: e.target.value })}
                placeholder={derived ? 'fees_billed / NULLIF(hours_recorded, 0)' : 'SUM(fee_amount)'}
              />
            </div>

            <div className="form-row">
              <div className="form-group">
                <label>
                  Grain
                  <FieldHelp text="The dimensions this metric may be broken down by. Anything not listed here cannot be used to slice it, which stops a figure being cut in a way its definition does not support." />
                </label>
                <div className="field-row">
                  <input
                    value={form.grain}
                    onChange={(e) => update({ grain: e.target.value })}
                    placeholder="matter_id, practice_area"
                  />
                  {sourceColumns.length > 0 && (
                    <select
                      value=""
                      style={{ width: 96, flexShrink: 0 }}
                      onChange={(e) =>
                        update({ grain: [...list(form.grain), e.target.value].join(', ') })
                      }
                    >
                      <option value="">Add…</option>
                      {sourceColumns
                        .filter((c) => !list(form.grain).includes(c.name))
                        .map((c) => (
                          <option key={c.name} value={c.name}>
                            {c.name}
                          </option>
                        ))}
                    </select>
                  )}
                </div>
                {unknownGrain.length > 0 && (
                  <p className="hint">
                    Not in the scanned schema: <code>{unknownGrain.join(', ')}</code>. The compiler
                    drops a dimension it cannot resolve, so check the spelling.
                  </p>
                )}
              </div>
              <div className="form-group">
                <label>
                  Additivity
                  <FieldHelp text={HELP.additivity} />
                </label>
                <select
                  value={form.aggregation}
                  onChange={(e) => update({ aggregation: e.target.value as Metric['aggregation'] })}
                >
                  <option value="additive">Additive</option>
                  <option value="semi_additive">Semi-additive (a balance)</option>
                  <option value="non_additive">Non-additive</option>
                </select>
                <p className="hint">{fillUnit(AGGREGATION_HELP[form.aggregation], unit)}</p>
              </div>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label>
                  Time axis column
                  <FieldHelp text="The real date or timestamp column the time grain applies to. A partition string that merely looks like a date must not be used here, or the grain it is meant to guard is unenforceable." />
                </label>
                <ColumnField
                  value={form.time_grain_column}
                  columns={temporalColumns}
                  onChange={(v) => update({ time_grain_column: v })}
                  empty="First temporal column in the grain"
                  placeholder="issued_date"
                  withType
                />
                {sourceColumns.length > 0 && temporalColumns.length === 0 && (
                  <p className="hint">
                    No date or timestamp column in the scanned schema for this table, so nothing can
                    be offered. A partition string that looks like a date is not one.
                  </p>
                )}
              </div>
              <div className="form-group">
                <label>
                  Permitted time grains
                  <FieldHelp text={HELP.timeGrain} />
                </label>
                <input
                  value={form.time_grains}
                  onChange={(e) => update({ time_grains: e.target.value })}
                  placeholder="month, quarter, year"
                />
                <p className="hint">
                  Anything omitted is refused rather than silently approximated.
                </p>
              </div>
            </div>

            <div className="form-row-3">
              <div className="form-group">
                <label>
                  Value type
                  <FieldHelp text="What kind of quantity this is. Presentation only, it never changes the compiled SQL." />
                </label>
                <select
                  value={form.value_type}
                  onChange={(e) => update({ value_type: e.target.value })}
                >
                  {VALUE_TYPES.map((v) => (
                    <option key={v} value={v}>
                      {v}
                    </option>
                  ))}
                </select>
              </div>
              <div className="form-group">
                <label>
                  Unit
                  <FieldHelp text="The unit the figure is in, e.g. GBP or hours. Not decoration: when a metric is composed from others, the compiler warns if their units differ, and it can only do that for metrics that declare one." />
                </label>
                <input
                  className="input-mono"
                  value={form.unit}
                  onChange={(e) => update({ unit: e.target.value })}
                  placeholder="GBP"
                />
              </div>
              <div className="form-group">
                <label>
                  Display format
                  <FieldHelp text="How the figure is written for a reader. Never applied to the stored value." />
                </label>
                <input
                  className="input-mono"
                  value={form.format}
                  onChange={(e) => update({ format: e.target.value })}
                  placeholder="£#,##0"
                />
              </div>
            </div>

            <div className="form-group">
              <label>
                Fixed filters
                <FieldHelp text="Conditions always applied, one per line. These are part of the definition, a caller cannot remove them, so the figure cannot be quietly widened. Name columns unqualified; the compiler assigns the table alias." />
              </label>
              <textarea
                className="input-mono"
                value={form.filters}
                onChange={(e) => update({ filters: e.target.value })}
                placeholder="status = 'ISSUED'"
              />
              {sourceColumns.length > 0 && (
                <p className="hint">
                  Columns on this table: <code>{sourceColumns.map((c) => c.name).join(', ')}</code>
                </p>
              )}
            </div>

            <details
              className="form-section"
              open={showParams}
              onToggle={(e) => setShowParams(e.currentTarget.open)}
            >
              <summary>
                Parameters{form.parameters.length > 0 ? ` (${form.parameters.length})` : ''}
                <FieldHelp text="The columns a caller may filter on when they ask. Declaring any closes the set to exactly these, so a question cannot narrow the figure in a way the definition does not sanction. Declaring none leaves the table's own columns as the filter surface, which is wider but still closed." />
              </summary>
              <div className="metric-rows">
                {form.parameters.map((p, i) => (
                  <div key={i} className="metric-row">
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <ColumnField
                        value={p.column}
                        columns={sourceColumns}
                        onChange={(v) => patchParam(i, { column: v })}
                        empty="Column…"
                        placeholder="column"
                      />
                    </div>
                    <select
                      value={p.operator}
                      onChange={(e) => patchParam(i, { operator: e.target.value })}
                      style={{ width: 92, flexShrink: 0 }}
                    >
                      {OPERATORS.map((o) => (
                        <option key={o} value={o}>
                          {o}
                        </option>
                      ))}
                    </select>
                    <label className="metric-check" title="A question that omits it is refused.">
                      <input
                        type="checkbox"
                        checked={p.required}
                        onChange={(e) => patchParam(i, { required: e.target.checked })}
                      />
                      Required
                    </label>
                    <input
                      value={p.description || ''}
                      onChange={(e) => patchParam(i, { description: e.target.value })}
                      placeholder="What it narrows"
                      style={{ flex: 2 }}
                    />
                    <button
                      className="btn btn-danger btn-sm"
                      onClick={() =>
                        update({ parameters: form.parameters.filter((_, n) => n !== i) })
                      }
                      title="Remove this parameter"
                    >
                      Remove
                    </button>
                  </div>
                ))}
              </div>
              {form.parameters.length === 0 && (
                <p className="hint" style={{ marginTop: 0, marginBottom: 8 }}>
                  None declared, so the filter surface is this table's own columns.
                </p>
              )}
              <button
                className="btn btn-ghost btn-sm"
                style={{ marginTop: form.parameters.length > 0 ? 8 : 0 }}
                onClick={() =>
                  update({
                    parameters: [
                      ...form.parameters,
                      { column: '', operator: '=', required: false, description: '' },
                    ],
                  })
                }
              >
                Add parameter
              </button>
            </details>

            <div className="form-group">
              <label>
                Synonyms
                <FieldHelp text="Alternative wordings that should match this metric. Adding the words your firm actually uses is what keeps questions on the deterministic path instead of falling through to the graph and the documents." />
              </label>
              <input
                value={form.synonyms}
                onChange={(e) => update({ synonyms: e.target.value })}
                placeholder="billings, revenue, turnover"
              />
            </div>

            {Object.keys(form.entity_columns).length > 0 && (
              <div className="form-group">
                <label>
                  Graph entity columns
                  <FieldHelp text="Which result columns resolve to graph entities, as column to node label. Declared rather than guessed: matching warehouse values to node names by similarity would silently join two different clients and produce a confident, cited, wrong answer. This form does not edit them and saving keeps them exactly as they are." />
                </label>
                <p className="hint">
                  {Object.entries(form.entity_columns)
                    .map(([col, label]) => `${col} to ${label}`)
                    .join(', ')}
                </p>
              </div>
            )}

            {preview && (
              <div className="form-group">
                <label>
                  Compiled SQL
                  <FieldHelp text="Compiled from this definition with no model involved, so the same definition always gives this same query. Nothing has been saved." />
                </label>
                <CompilerWarnings warnings={preview.warnings} />
                <pre className="code-block" style={{ margin: 0, maxHeight: 220, overflow: 'auto' }}>
                  {preview.sql}
                </pre>
                <p className="hint">
                  Read this before saving. Once the metric is approved, this is the query that
                  answers the question, and no model rewrites it.
                </p>
              </div>
            )}

            <div className="modal-actions">
              <button
                className="btn btn-ghost"
                style={{ marginRight: 'auto' }}
                onClick={runPreview}
                disabled={previewing || !form.name || !form.expression || !shapeReady}
                title="Compile this definition to SQL without saving it."
              >
                {previewing ? 'Compiling…' : 'Preview SQL'}
              </button>
              <button className="btn btn-ghost" onClick={() => setModal(false)}>
                Cancel
              </button>
              <button
                className="btn btn-primary"
                onClick={save}
                disabled={
                  saving || !form.metric_id || !form.name || !form.expression || !shapeReady
                }
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
