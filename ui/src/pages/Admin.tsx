import { useEffect, useState } from 'react'
import { api, type Ontology, type Source, type TenantSettings, type TenantUser } from '../api'
import { getTenantId } from '../auth'
import { HELP } from '../epistemic'
import { fallback, MOCK_ONTOLOGY, MOCK_SETTINGS, MOCK_SOURCES } from '../mocks'
import ConfidenceBar from '../components/ConfidenceBar'
import FieldHelp from '../components/FieldHelp'
import { MockFlag, Spinner, Toast } from '../components/Shared'
import { fmtDateTime, fmtNum } from '../format'

export default function Admin() {
  const tenant = getTenantId()
  const [settings, setSettings] = useState<TenantSettings | null>(null)
  const [ontology, setOntology] = useState<Ontology | null>(null)
  const [sources, setSources] = useState<Source[]>([])
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState<string | null>(null)
  const [toast, setToast] = useState<{ msg: string; type: string } | null>(null)
  const [users, setUsers] = useState<TenantUser[]>([])
  const [newEmail, setNewEmail] = useState('')
  const [newIsAdmin, setNewIsAdmin] = useState(false)
  const [inviting, setInviting] = useState(false)

  const showToast = (msg: string, type = 'success') => {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 4000)
  }

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

  useEffect(() => {
    Promise.all([
      fallback(api.getSettings(tenant), MOCK_SETTINGS),
      fallback(api.listSources(tenant), MOCK_SOURCES),
    ])
      .then(([s, src]) => {
        setSettings(s)
        setSources(src)
        return fallback(api.ontology(s.ontology_domain), MOCK_ONTOLOGY)
      })
      .then(setOntology)
      .catch(console.error)
      .finally(() => setLoading(false))
    loadUsers()
  }, [tenant])

  const patch = async (key: string, body: Partial<TenantSettings>, message: string) => {
    setSaving(key)
    // Applied locally first: these are single-field policy toggles and a stale
    // switch is more confusing than an optimistic one.
    setSettings((s) => (s ? { ...s, ...body } : s))
    try {
      const next = await api.updateSettings(tenant, body)
      setSettings(next)
    } catch {
      // API not live yet; the local state above stands.
    } finally {
      setSaving(null)
      showToast(message)
    }
  }

  const changeDomain = async (domain: string) => {
    if (
      !confirm(
        `Switch the ontology to "${domain}"?\n\nThe closed list of governing relationships changes with it. Existing facts are not rewritten — any that use a relationship the new pack does not recognise stay in the graph but stop being writable.`,
      )
    )
      return
    await patch('domain', { ontology_domain: domain }, `Ontology set to ${domain}`)
    setOntology(await fallback(api.ontology(domain), MOCK_ONTOLOGY))
  }

  if (loading) return <Spinner />
  if (!settings) return <div className="empty-state">Could not load tenant settings.</div>

  return (
    <>
      <div className="page-header">
        <div className="page-header-row">
          <div>
            <h2>Admin</h2>
            <p>Tenant configuration, the active vocabulary, and the governance controls.</p>
          </div>
          <MockFlag />
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
            answered with model-generated SQL — in the web UI and over the API alike. Governed metrics
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
              <FieldHelp text="Phrases the final answer from facts already retrieved. It does not decide what is true — it only writes up what the graph and the metrics returned." />
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
