/**
 * Platform — the operator's screen, not a firm's.
 *
 * Deliberately separate from Admin, which is a tenant looking at itself. Creating and deleting
 * tenants crosses firms, so putting a destroy button on a page a customer's own admin can reach
 * is exactly the confusion the server-side guard exists to prevent. This page 403s for anyone
 * outside the operator tenant, and the nav entry is hidden for them.
 */

import { useEffect, useState } from 'react'

import { api, type Tenant, type TenantDeleteReport } from '../api'
import { EmptyState, ErrorState, Spinner } from '../components/Shared'
import FieldHelp from '../components/FieldHelp'
import { fmtNum } from '../format'

export default function Platform() {
  const [tenants, setTenants] = useState<Tenant[] | null>(null)
  const [homeTenant, setHomeTenant] = useState('')
  const [domains, setDomains] = useState<string[]>([])
  const [error, setError] = useState('')
  const [busy, setBusy] = useState('')
  const [toast, setToast] = useState('')
  const [report, setReport] = useState<TenantDeleteReport | null>(null)

  const [tenantId, setTenantId] = useState('')
  const [adminEmail, setAdminEmail] = useState('')
  const [name, setName] = useState('')
  const [domain, setDomain] = useState('legal')

  const load = () => {
    api
      .listTenants()
      .then((r) => {
        setTenants(r.tenants)
        setHomeTenant(r.home_tenant)
        setError('')
      })
      .catch((e: Error) => {
        setTenants([])
        setError(e.message)
      })
  }

  useEffect(load, [])
  useEffect(() => {
    // The pack list comes from settings rather than being hardcoded, so a pack added to the
    // image appears here without a UI change.
    api
      .getSettings(homeTenant || 'demo-firm')
      .then((s) => setDomains(s.available_domains))
      .catch(() => setDomains(['legal']))
  }, [homeTenant])

  const create = async () => {
    setBusy('create')
    setReport(null)
    try {
      await api.createTenant({
        tenant_id: tenantId.trim(),
        admin_email: adminEmail.trim(),
        name: name.trim(),
        ontology_domain: domain,
      })
      setToast(`Created ${tenantId}. Cognito has emailed ${adminEmail} a temporary password.`)
      setTenantId('')
      setAdminEmail('')
      setName('')
      load()
    } catch (e) {
      setToast((e as Error).message.replace(/^\d+:\s*/, ''))
    } finally {
      setBusy('')
    }
  }

  const remove = async (t: Tenant) => {
    // Typing the id is the server's requirement too. Asked here as well so the irreversible
    // part is stated in words before the request, not explained by a 400 afterwards.
    const typed = prompt(
      `Delete ${t.tenant_id}?\n\nThis erases every document from S3 including every version, ` +
        `every fact, every user, and the whole audit trail. Nothing replays afterwards.\n\n` +
        `Type ${t.tenant_id} to confirm.`,
    )
    if (typed !== t.tenant_id) return
    setBusy(t.tenant_id)
    try {
      const r = await api.deleteTenant(t.tenant_id)
      setReport(r)
      setToast(r.complete ? `Deleted ${t.tenant_id}` : `Deleted ${t.tenant_id} with errors`)
      load()
    } catch (e) {
      setToast((e as Error).message.replace(/^\d+:\s*/, ''))
    } finally {
      setBusy('')
    }
  }

  if (tenants === null) return <Spinner />

  return (
    <div className="page">
      <div className="page-header">
        <div className="page-header-row">
          <div>
            <h2>Platform</h2>
            <p>
              Every tenant on this deployment. Only admins of{' '}
              <code>{homeTenant || 'the operator tenant'}</code> can see this, because creating and
              deleting tenants crosses firms.
            </p>
          </div>
        </div>
      </div>

      {error && <ErrorState title="Could not load tenants" detail={error} onRetry={load} />}
      {toast && <div className="banner banner-info">{toast}</div>}

      {report && !report.complete && (
        <div className="banner banner-warn">
          <span>
            <strong>The delete did not finish.</strong> {report.errors.join('; ')}. Every step is
            idempotent, so running it again is safe.
          </span>
        </div>
      )}
      {report?.complete && (
        <div className="banner banner-info">
          <span>
            Removed {fmtNum(report.users_deleted)} users, {fmtNum(report.assertions_dropped)} facts,{' '}
            {fmtNum(report.documents_erased)} object versions, {fmtNum(report.grants_dropped)} grants
            and both audit logs.
          </span>
        </div>
      )}

      <div className="card">
        <div className="card-header">
          <h3>
            New tenant
            <FieldHelp text="A tenant is created with its first admin in one act. A tenant with no users cannot be signed in to, so the email is required rather than optional. The address becomes their Cognito username and their tenant is fixed at creation, so an address already in use elsewhere cannot be moved here." />
          </h3>
        </div>
        <div className="form-row">
          <div className="form-group">
            <label>Tenant id</label>
            <input
              value={tenantId}
              onChange={(e) => setTenantId(e.target.value)}
              placeholder="demo-clinic"
            />
            <p className="hint">Lowercase letters, digits and hyphens. Reaches S3 and index names.</p>
          </div>
          <div className="form-group">
            <label>Admin email</label>
            <input
              value={adminEmail}
              onChange={(e) => setAdminEmail(e.target.value)}
              placeholder="admin@clinic.example"
            />
          </div>
          <div className="form-group">
            <label>Display name</label>
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Optional" />
          </div>
          <div className="form-group">
            <label>Ontology pack</label>
            <select value={domain} onChange={(e) => setDomain(e.target.value)}>
              {domains.map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
          </div>
        </div>
        <button
          className="btn btn-primary"
          disabled={!tenantId.trim() || !adminEmail.trim() || busy === 'create'}
          onClick={create}
        >
          {busy === 'create' ? 'Creating…' : 'Create tenant and invite admin'}
        </button>
      </div>

      <div className="card">
        <div className="card-header">
          <h3>Tenants</h3>
          <span className="card-note">
            Deleted ones are listed too &middot; a reused id should read as reused
          </span>
        </div>
        {tenants.length === 0 ? (
          <EmptyState title="No tenants recorded">
            A tenant created before this screen existed has no record, so it will not appear here
            until it is created again.
          </EmptyState>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>Tenant</th>
                <th>Pack</th>
                <th>Created</th>
                <th>State</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {tenants.map((t) => (
                <tr key={t.tenant_id}>
                  <td>
                    <code>{t.tenant_id}</code>
                    {t.name && t.name !== t.tenant_id && (
                      <span className="card-note" style={{ marginLeft: 8 }}>
                        {t.name}
                      </span>
                    )}
                  </td>
                  <td>
                    <span className="tag tag-purple">{t.ontology_domain}</span>
                  </td>
                  <td>{t.created_at ? new Date(t.created_at).toLocaleDateString() : '-'}</td>
                  <td>
                    {t.is_live === false ? (
                      <span className="tag tag-neutral" title={t.deleted_at ?? ''}>
                        deleted
                      </span>
                    ) : (
                      <span className="tag tag-green">live</span>
                    )}
                  </td>
                  <td style={{ textAlign: 'right' }}>
                    {t.is_live !== false && t.tenant_id !== homeTenant && (
                      <button
                        className="btn btn-ghost btn-sm"
                        style={{ color: 'var(--red)' }}
                        disabled={busy === t.tenant_id}
                        onClick={() => remove(t)}
                      >
                        {busy === t.tenant_id ? 'Deleting…' : 'Delete'}
                      </button>
                    )}
                    {t.tenant_id === homeTenant && (
                      <span className="card-note">operator</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
