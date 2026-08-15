import { useEffect, useState } from 'react'
import {
  api,
  type Ontology,
  type ResetScope,
  type Source,
  type TenantSettings,
  type TenantUser,
} from '../api'
import { getTenantId } from '../auth'
import { HELP } from '../epistemic'
import ConfidenceBar from '../components/ConfidenceBar'
import FieldHelp from '../components/FieldHelp'
import { ErrorState, Spinner, Toast } from '../components/Shared'
import { fmtDateTime, fmtNum } from '../format'

const RESET_OPTIONS: { key: keyof ResetScope; label: string; rebuild: string }[] = [
  { key: 'graph', label: 'Graph facts', rebuild: 'Rebuilt by Replay' },
  { key: 'vectors', label: 'Search index', rebuild: 'Rebuilt by Replay' },
  { key: 'jobs', label: 'Ingest job history', rebuild: 'Rewritten on the next ingest' },
  { key: 'catalog', label: 'Catalog cache', rebuild: 'Rebuilt by Scan catalog' },
  { key: 'metrics', label: 'Metric definitions', rebuild: 'Nothing rebuilds these' },
]

export default function Admin() {
  const tenant = getTenantId()
  const [settings, setSettings] = useState<TenantSettings | null>(null)
  const [ontology, setOntology] = useState<Ontology | null>(null)
  const [sources, setSources] = useState<Source[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [reloadKey, setReloadKey] = useState(0)
  const [saving, setSaving] = useState<string | null>(null)
  const [toast, setToast] = useState<{ msg: string; type: string } | null>(null)
  const [users, setUsers] = useState<TenantUser[]>([])
  const [newEmail, setNewEmail] = useState('')
  const [newIsAdmin, setNewIsAdmin] = useState(false)
  const [inviting, setInviting] = useState(false)
  const [removing, setRemoving] = useState<string | null>(null)
  const [scope, setScope] = useState<ResetScope>({
    graph: true,
    vectors: true,
    jobs: true,
    catalog: true,
    metrics: false,
  })
  const [confirmMetricLoss, setConfirmMetricLoss] = useState(false)
  const [running, setRunning] = useState<string | null>(null)

  const showToast = (msg: string, type = 'success') => {
    setToast({ msg, type })
    setTimeout(() => setToast(null), type === 'error' ? 9000 : 5000)
  }

  const setScopeFlag = (key: keyof ResetScope, on: boolean) => {
    setScope((s) => ({ ...s, [key]: on }))
    if (key === 'metrics' && !on) setConfirmMetricLoss(false)
  }

  // A partial failure must not read as a success, so errors are surfaced as an error toast
  // even though the request itself returned 200.
  const runOp = async <T extends { errors: string[]; note?: string }>(
    key: string,
    call: () => Promise<T>,
    summarise: (r: T) => string,
  ) => {
    setRunning(key)
    try {
      const r = await call()
      const line = summarise(r)
      if (r.errors.length)
        showToast(`${line}. ${r.errors.length} problem(s): ${r.errors.slice(0, 3).join('; ')}`, 'error')
      else showToast(line)
      setReloadKey((k) => k + 1)
    } catch (e) {
      showToast((e as Error).message.replace(/^\d+:\s*/, ''), 'error')
    } finally {
      setRunning(null)
    }
  }

  const doReset = async () => {
    const parts = (Object.keys(scope) as (keyof ResetScope)[]).filter((k) => scope[k])
    if (!parts.length) return
    if (!confirm(`Remove ${parts.join(', ')} for ${tenant}?`)) return
    await runOp(
      'reset',
      () => api.resetDerived(tenant, scope, confirmMetricLoss),
      (r) =>
        `Reset done: ${fmtNum(r.assertions_dropped)} assertions, ${fmtNum(r.vectors_dropped)} vectors, ` +
        `${fmtNum(r.jobs_dropped)} jobs, ${fmtNum(r.tables_forgotten)} tables removed. ` +
        (r.metrics_dropped
          ? `${fmtNum(r.metrics_dropped)} metric definitions deleted and not recoverable.`
          : `${fmtNum(r.metrics_preserved)} metric definitions kept.`),
    )
    setScopeFlag('metrics', false)
  }

  const resetBlocked = scope.metrics && !confirmMetricLoss
  const busy = running !== null

  const loadUsers = () => {
    // No mock fallback: an empty list is a truthful answer, and inventing colleagues on
    // an admin screen would be actively misleading.
    api
      .listUsers(tenant)
      .then((r) => setUsers(r.users))
      .catch(() => setUsers([]))
  }

  const inviteUser = async () => {
    const email = newEmail.trim().toLowerCase()
    if (!email) return
    setInviting(true)
    try {
      const created = await api.createUser(tenant, email, newIsAdmin)
      showToast(`${created.email} invited. Cognito has emailed a temporary password.`)
      setNewEmail('')
      setNewIsAdmin(false)
      loadUsers()
    } catch (e) {
      showToast((e as Error).message.replace(/^\d+:\s*/, ''), 'error')
    } finally {
      setInviting(false)
    }
  }

  const removeUser = async (email: string) => {
    if (
      !confirm(
        `Delete ${email}?\n\nThe account is removed from the directory and they can no longer sign in. Re-inviting them creates a new account, so anything recorded against the old one keeps naming an identity that no longer exists.`,
      )
    )
      return
    setRemoving(email)
    try {
      const r = await api.deleteUser(tenant, email)
      showToast(`${email} deleted. ${r.note}`)
      loadUsers()
    } catch (e) {
      showToast((e as Error).message.replace(/^\d+:\s*/, ''), 'error')
    } finally {
      setRemoving(null)
    }
  }

  useEffect(() => {
    Promise.all([api.getSettings(tenant), api.listSources(tenant)])
      .then(([s, src]) => {
        setSettings(s)
        setSources(src)
        setError('')
        return api.ontology(s.ontology_domain)
      })
      .then(setOntology)
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false))
    loadUsers()
  }, [tenant, reloadKey])

  const patch = async (key: string, body: Partial<TenantSettings>, message: string) => {
    setSaving(key)
    const before = settings
    setSettings((s) => (s ? { ...s, ...body } : s))
    try {
      const next = await api.updateSettings(tenant, body)
      setSettings(next)
      showToast(message)
    } catch (e) {
      // These are governance policies. A toggle that did not persist must snap back.
      setSettings(before)
      showToast(
        `Could not save that setting: ${(e as Error).message.replace(/^\d+:\s*/, '')}`,
        'error',
      )
    } finally {
      setSaving(null)
    }
  }

  const changeDomain = async (domain: string) => {
    if (
      !confirm(
        `Switch the ontology to "${domain}"?\n\nThe closed list of governing relationships changes with it. Existing facts are not rewritten, any that use a relationship the new pack does not recognise stay in the graph but stop being writable.`,
      )
    )
      return
    await patch('domain', { ontology_domain: domain }, `Ontology set to ${domain}`)
    try {
      setOntology(await api.ontology(domain))
    } catch (e) {
      setOntology(null)
      showToast(
        `Could not load the ${domain} vocabulary: ${(e as Error).message.replace(/^\d+:\s*/, '')}`,
        'error',
      )
    }
  }

  const retry = () => {
    setLoading(true)
    setReloadKey((k) => k + 1)
  }

  if (loading) return <Spinner />
  if (error || !settings)
    return (
      <ErrorState
        title="Could not load tenant settings"
        detail={error}
        onRetry={retry}
      />
    )

  return (
    <>
      <div className="page-header">
        <div className="page-header-row">
          <div>
            <h2>Admin</h2>
            <p>Tenant configuration, the active vocabulary, and the governance controls.</p>
          </div>
        </div>
      </div>

      <div className="card" style={{ marginBottom: 16 }}>
        <div className="card-header">
          <h3>People</h3>
          <span className="card-note">
            Cognito emails the temporary password &middot; tenant is fixed at creation
          </span>
        </div>

        <div className="form-row">
          <div className="toolbar-field" style={{ flex: 1, minWidth: 260 }}>
            <label>Email address</label>
            <input
              type="email"
              placeholder="colleague@firm.example"
              value={newEmail}
              onChange={(e) => setNewEmail(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') inviteUser()
              }}
              style={{ width: '100%' }}
            />
          </div>
          <label className="checkbox-row" style={{ alignSelf: 'end' }}>
            <input
              type="checkbox"
              checked={newIsAdmin}
              onChange={(e) => setNewIsAdmin(e.target.checked)}
            />
            Also make them an administrator
          </label>
          <button
            className="btn btn-primary"
            style={{ alignSelf: 'end' }}
            disabled={inviting || !newEmail.trim()}
            onClick={inviteUser}
          >
            {inviting ? 'Inviting…' : 'Invite'}
          </button>
        </div>

        {users.length === 0 ? (
          <p className="card-note">
            You have not invited anyone yet. A new user receives a temporary password by email and
            must change it at first sign-in. Their tenant cannot be changed afterwards, so an
            address in the wrong firm has to be re-invited rather than edited.
          </p>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>Person</th>
                <th>Email</th>
                <th>Status</th>
                <th>Invited</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {users.map((u) => (
                <tr key={u.user_id}>
                  <td>{u.display_name}</td>
                  <td className="mono">{u.email}</td>
                  <td>
                    <span className="tag">
                      {u.status === 'FORCE_CHANGE_PASSWORD' ? 'Invited' : 'Active'}
                    </span>
                  </td>
                  <td className="muted">{fmtDateTime(u.created_at)}</td>
                  <td style={{ textAlign: 'right' }}>
                    <button
                      className="btn btn-sm btn-danger"
                      disabled={removing === u.email}
                      onClick={() => removeUser(u.email)}
                    >
                      {removing === u.email ? 'Deleting…' : 'Delete'}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="admin-grid">
        <div className="card">
          <div className="card-header">
            <h3>
              Tenant
              <FieldHelp text={HELP.tenant} />
            </h3>
          </div>
          <div className="detail-field">
            <div className="label">Name</div>
            <div className="value">{settings.name}</div>
          </div>
          <div className="detail-field">
            <div className="label">Identifier</div>
            <div className="value">
              <code>{settings.tenant_id}</code>
            </div>
          </div>
          <p className="card-note">
            Every read is filtered to this tenant, and there is no way to express a query that is not.
            Matters are subgraphs within it, filtered by your grants rather than stored separately.
          </p>
        </div>

        <div className="card">
          <div className="card-header">
            <h3>
              Ontology domain
              <FieldHelp text={HELP.ontologyDomain} />
            </h3>
            <span className="tag tag-purple">{settings.ontology_domain}</span>
          </div>
          <div className="form-group">
            <label>Active pack</label>
            <select
              value={settings.ontology_domain}
              onChange={(e) => changeDomain(e.target.value)}
              disabled={saving === 'domain'}
            >
              {settings.available_domains.map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
            <p className="hint">
              The platform is domain-agnostic. The legal pack is the default; the healthcare pack
              exists to keep that claim honest.
            </p>
          </div>
          {ontology && (
            <div className="detail-grid" style={{ marginTop: 4 }}>
              <div className="detail-field" style={{ marginBottom: 0 }}>
                <div className="label">Entity types</div>
                <div className="value">{ontology.entity_types.length}</div>
              </div>
              <div className="detail-field" style={{ marginBottom: 0 }}>
                <div className="label">
                  Governing
                  <FieldHelp text={HELP.governingPredicate} />
                </div>
                <div className="value">{ontology.governing_predicates.length}</div>
              </div>
              <div className="detail-field" style={{ marginBottom: 0 }}>
                <div className="label">
                  Descriptive
                  <FieldHelp text={HELP.descriptivePredicate} />
                </div>
                <div className="value">{ontology.descriptive_predicates.length}</div>
              </div>
              <div className="detail-field" style={{ marginBottom: 0 }}>
                <div className="label">Rules</div>
                <div className="value">{ontology.rules.length}</div>
              </div>
            </div>
          )}
        </div>

        <div className="card">
          <div className="card-header">
            <h3>
              Retrieval trust floor
              <FieldHelp text={HELP.confidenceFloor} />
            </h3>
            <span className="tag tag-blue">{settings.min_confidence.toFixed(2)}</span>
          </div>
          <div className="form-group">
            <label>Minimum confidence</label>
            <input
              type="range"
              min={0.5}
              max={0.99}
              step={0.01}
              value={settings.min_confidence}
              onChange={(e) =>
                setSettings((s) => (s ? { ...s, min_confidence: Number(e.target.value) } : s))
              }
              onMouseUp={(e) =>
                patch(
                  'floor',
                  { min_confidence: Number((e.target as HTMLInputElement).value) },
                  `Trust floor set to ${Number((e.target as HTMLInputElement).value).toFixed(2)}`,
                )
              }
              style={{ width: '100%' }}
            />
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 8 }}>
              <ConfidenceBar
                value={settings.min_confidence}
                floor={settings.min_confidence}
                width={140}
              />
            </div>
            <p className="hint">
              Facts below the floor stay visible in the review queue and the audit trail but never
              shape an answer. Raising it makes the system more likely to say it does not know, which
              is usually the safer failure.
            </p>
          </div>
        </div>

        <div className="card" style={{ borderColor: settings.block_ungoverned_queries ? 'var(--red)' : undefined }}>
          <div className="card-header">
            <h3>
              Ungoverned queries
              <FieldHelp text={HELP.ungovernedKillSwitch} />
            </h3>
            <span className={`tag ${settings.block_ungoverned_queries ? 'tag-red' : 'tag-green'}`}>
              {settings.block_ungoverned_queries ? 'blocked' : 'allowed'}
            </span>
          </div>
          <label className="switch switch-danger" style={{ marginBottom: 12 }}>
            <input
              type="checkbox"
              checked={settings.block_ungoverned_queries}
              disabled={saving === 'kill'}
              onChange={(e) =>
                patch(
                  'kill',
                  { block_ungoverned_queries: e.target.checked },
                  e.target.checked
                    ? 'Ungoverned queries are now refused'
                    : 'Ungoverned queries are now allowed',
                )
              }
            />
            <span className="switch-track" />
            <span>Refuse questions no approved metric can answer</span>
          </label>
          <p className="card-note">
            When on, a question that matches no approved governed metric is refused instead of being
            answered with model-generated SQL, in the web UI and over the API alike. Governed metrics
            are unaffected: they compile deterministically and never depended on a model.
          </p>
          <p className="card-note" style={{ marginTop: 9 }}>
            Refused questions are logged. They are the best available backlog of metrics worth
            defining.
          </p>
        </div>

        <div className="card">
          <div className="card-header">
            <h3>Models</h3>
          </div>
          <div className="form-group">
            <label>
              Extraction
              <FieldHelp text="Reads documents and proposes facts, whichever model this is. Where it quotes words the system can confirm are on the page it names, that much goes live directly; anything it reads into the passage waits for review." />
            </label>
            <select
              value={settings.extraction_model}
              onChange={(e) =>
                patch('extraction', { extraction_model: e.target.value }, 'Extraction model updated')
              }
            >
              {settings.available_models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
          </div>
          <div className="form-group">
            <label>
              Synthesis
              <FieldHelp text="Phrases the final answer from facts already retrieved. It does not decide what is true, it only writes up what the graph and the metrics returned." />
            </label>
            <select
              value={settings.synthesis_model}
              onChange={(e) =>
                patch('synthesis', { synthesis_model: e.target.value }, 'Synthesis model updated')
              }
            >
              {settings.available_models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
          </div>
          <div className="form-group" style={{ marginBottom: 0 }}>
            <label>
              Embeddings
              <FieldHelp text="Turns passages into vectors for search. Changing it means re-indexing, because vectors from different models are not comparable." />
            </label>
            <input
              value={settings.embedding_model}
              onChange={(e) =>
                setSettings((s) => (s ? { ...s, embedding_model: e.target.value } : s))
              }
              onBlur={(e) =>
                patch('embed', { embedding_model: e.target.value }, 'Embedding model updated')
              }
              className="input-mono"
            />
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h3>Structured sources</h3>
          </div>
          <table className="data-table">
            <thead>
              <tr>
                <th>Source</th>
                <th>Kind</th>
                <th className="num">Tables</th>
                <th>Scanned</th>
              </tr>
            </thead>
            <tbody>
              {sources.map((s) => (
                <tr key={s.source_id}>
                  <td>
                    <strong>{s.name}</strong>
                    <div className="dim" style={{ fontSize: 11 }}>
                      {s.database} &middot; {s.region}
                    </div>
                  </td>
                  <td>
                    <span className="tag tag-blue">{s.kind}</span>
                  </td>
                  <td className="num">{fmtNum(s.table_count)}</td>
                  <td className="nowrap dim">{fmtDateTime(s.last_scanned_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="card-note" style={{ marginTop: 10 }}>
            A scan records metadata only. Rows never leave the source.
          </p>
        </div>
      </div>

      <div className="card" style={{ marginBottom: 16 }}>
        <div className="card-header">
          <h3>Maintenance</h3>
          <span className="card-note">
            Nothing here touches S3 &middot; documents and schemas stay where they are
          </span>
        </div>

        <div className="form-group">
          <label>Reset derived data</label>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px 20px', marginBottom: 10 }}>
            {RESET_OPTIONS.map((o) => (
              <label key={o.key} className="checkbox-row" style={{ minWidth: 210 }}>
                <input
                  type="checkbox"
                  checked={scope[o.key]}
                  onChange={(e) => setScopeFlag(o.key, e.target.checked)}
                />
                <span>
                  {o.label}
                  <span className="dim" style={{ display: 'block', fontSize: 11 }}>
                    {o.rebuild}
                  </span>
                </span>
              </label>
            ))}
          </div>

          {scope.metrics && (
            <div className="banner banner-error">
              <div>
                <strong>Metric definitions cannot be rebuilt.</strong> Documents come back from S3
                and schemas come back from Glue, so Replay restores everything else on this list. A
                metric definition was authored in this app and has no upstream source, so deleting
                it is permanent: the expression, grain, filters, synonyms and approval history are
                gone, and every question that resolved through it falls back to model-generated SQL.
                <label className="checkbox-row" style={{ marginTop: 9 }}>
                  <input
                    type="checkbox"
                    checked={confirmMetricLoss}
                    onChange={(e) => setConfirmMetricLoss(e.target.checked)}
                  />
                  I accept that these metric definitions are unrecoverable.
                </label>
              </div>
            </div>
          )}

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            <button
              className="btn btn-danger"
              disabled={busy || resetBlocked}
              title={resetBlocked ? 'Confirm the loss of metric definitions first' : undefined}
              onClick={doReset}
            >
              {running === 'reset' ? 'Resetting…' : 'Reset'}
            </button>
            <button
              className="btn btn-ghost"
              disabled={busy}
              onClick={() =>
                runOp(
                  'replay',
                  () => api.replay(tenant),
                  (r) =>
                    `Replay done: ${fmtNum(r.documents_ingested)} of ${fmtNum(r.documents_found)} documents rebuilt from S3` +
                    (r.documents_failed ? `, ${fmtNum(r.documents_failed)} failed` : ''),
                )
              }
            >
              {running === 'replay' ? 'Replaying…' : 'Replay from S3'}
            </button>
            <button
              className="btn btn-ghost"
              disabled={busy}
              onClick={() =>
                runOp(
                  'sample',
                  () => api.loadSampleData(tenant),
                  (r) =>
                    `Loaded ${fmtNum(r.documents_loaded)} example documents, ${fmtNum(r.chunks)} passages indexed`,
                )
              }
            >
              {running === 'sample' ? 'Loading…' : 'Load sample data'}
            </button>
            <button
              className="btn btn-ghost"
              disabled={busy}
              onClick={() =>
                runOp(
                  'scan',
                  async () => {
                    const r = await api.scanSources(tenant)
                    return {
                      ...r,
                      errors: [...r.scan_errors, ...(r.graph_error ? [r.graph_error] : [])],
                    }
                  },
                  (r) =>
                    `Scanned the catalog: ${fmtNum(r.tables_found)} tables, ${fmtNum(r.assertions_live)} declared facts live`,
                )
              }
            >
              {running === 'scan' ? 'Scanning…' : 'Scan catalog'}
            </button>
          </div>
          <p className="hint">
            Replay and sample loading run inline without model extraction, so they return a report
            rather than a spinner. A large corpus is better replayed by re-uploading.
          </p>
        </div>
      </div>

      {ontology && (
        <div className="card">
          <div className="card-header">
            <h3>
              Governing relationships
              <FieldHelp text={HELP.governingPredicate} />
            </h3>
            <span className="card-note">
              Closed list &middot; validated when a fact is written, not when it is read
            </span>
          </div>
          <table className="data-table">
            <thead>
              <tr>
                <th>Relationship</th>
                <th>From → to</th>
                <th>Meaning</th>
              </tr>
            </thead>
            <tbody>
              {ontology.governing_predicates.map((p) => (
                <tr key={p.id}>
                  <td>
                    <code>{p.id}</code>
                    {p.symmetric && (
                      <span
                        className="tag tag-neutral"
                        style={{ marginLeft: 6 }}
                        title="Holds in both directions. Recording it one way records it both ways, so a check cannot miss it by asking from the wrong end."
                      >
                        symmetric
                      </span>
                    )}
                  </td>
                  <td className="nowrap dim">
                    {p.domain.join(' | ')} → {p.range.join(' | ')}
                  </td>
                  <td className="dim">
                    {p.description}
                    {p.help && <FieldHelp text={p.help} />}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          <div className="card-header" style={{ marginTop: 22 }}>
            <h3>
              Rules
              <FieldHelp text={HELP.proofTree} />
            </h3>
            <span className="card-note">Each conclusion carries its premises</span>
          </div>
          <table className="data-table">
            <thead>
              <tr>
                <th>Rule</th>
                <th>When</th>
                <th>Then</th>
                <th>
                  Minimum premise class
                  <FieldHelp text="The weakest kind of fact the rule will fire on. Conflict checking is set so that it fires only on facts declared by a system of record or confirmed by a check, because a conflict flag resting on a model's guess would be worse than none at all." />
                </th>
              </tr>
            </thead>
            <tbody>
              {ontology.rules.map((r) => (
                <tr key={r.id}>
                  <td>
                    <code>
                      {r.id}@{r.version}
                    </code>
                    <div className="dim" style={{ fontSize: 12, marginTop: 3, maxWidth: 260 }}>
                      {r.description}
                      {r.help && <FieldHelp text={r.help} />}
                    </div>
                  </td>
                  <td>
                    {r.when.map((w) => (
                      <div key={w}>
                        <code style={{ fontSize: 11 }}>{w}</code>
                      </div>
                    ))}
                  </td>
                  <td>
                    <code style={{ fontSize: 11 }}>{r.then}</code>
                  </td>
                  <td>
                    <span className="tag tag-teal">{r.min_premise_class}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <Toast toast={toast} />
    </>
  )
}
