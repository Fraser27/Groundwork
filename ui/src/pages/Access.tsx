/**
 * Who may read which matter.
 *
 * Matter-centric first, because access is allowlist-primary: a user sees only the
 * matters someone put them on, so the everyday task is staffing a matter rather than
 * auditing a person. The by-user view exists for the other question — "what can this
 * lawyer reach, and why" — which is what a risk reviewer actually asks.
 *
 * A screen beats an assignment and beats the administrator role. Both views render that
 * ordering rather than reporting a simple yes or no, because "screened" needs a
 * conversation with the risk team and "not on the team" usually just needs staffing.
 */

import { useEffect, useMemo, useState } from 'react'
import {
  api,
  type AccessDecision,
  type DirectoryUser,
  type MatterAccessDetail,
  type UserAccess,
} from '../api'
import { getTenantId } from '../auth'
import { ACCESS_DECISIONS, HELP } from '../epistemic'
import AccessAudit from '../components/AccessAudit'
import FieldHelp from '../components/FieldHelp'
import { EmptyState, ErrorState, Spinner, Toast } from '../components/Shared'
import { fmtDate } from '../format'

const ROLES = ['supervising partner', 'associate', 'paralegal', 'trainee', 'support'] as const

type View = 'matter' | 'user'

interface MatterRef {
  matter_id: string
  name: string
}

/** What a confirmation dialogue is about. Kept as one value so only one can be open. */
type Pending =
  | { kind: 'screen'; userId: string; userLabel: string; matterId: string; matterLabel: string }
  | { kind: 'lift'; userId: string; userLabel: string; matterId: string; matterLabel: string }
  | { kind: 'unassign'; userId: string; userLabel: string; matterId: string; matterLabel: string }

export default function Access() {
  const tenant = getTenantId()
  const [view, setView] = useState<View>('matter')
  const [matters, setMatters] = useState<MatterRef[]>([])
  const [users, setUsers] = useState<DirectoryUser[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [reloadKey, setReloadKey] = useState(0)
  const [detailError, setDetailError] = useState('')
  const [matterId, setMatterId] = useState<string | null>(null)
  const [userId, setUserId] = useState<string | null>(null)
  const [matterAccess, setMatterAccess] = useState<MatterAccessDetail | null>(null)
  const [userAccess, setUserAccess] = useState<UserAccess | null>(null)
  const [loadedKey, setLoadedKey] = useState<string | null>(null)
  const [pending, setPending] = useState<Pending | null>(null)
  const [addUser, setAddUser] = useState('')
  const [addRole, setAddRole] = useState<string>(ROLES[1])
  const [refreshKey, setRefreshKey] = useState(0)
  const [toast, setToast] = useState<{ msg: string; type: string } | null>(null)

  const showToast = (msg: string, type = 'success') => {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 4500)
  }

  useEffect(() => {
    Promise.all([api.listMatters(tenant), api.listAccessUsers(tenant)])
      .then(([m, u]) => {
        // A matter screened from the signed-in administrator is still administered here:
        // whoever manages a wall has to be able to see the matter it applies to. The list
        // endpoint does not disclose its name to this caller, so the id stands in until
        // the detail fetch supplies one.
        const refs: MatterRef[] = [
          ...m.matters.map((x) => ({ matter_id: x.matter_id, name: x.name })),
          ...m.withheld.map((w) => ({ matter_id: w.matter_id, name: w.matter_id })),
        ]
        setMatters(refs)
        setUsers(u)
        setMatterId((id) => id ?? refs[0]?.matter_id ?? null)
        setUserId((id) => id ?? u[0]?.user_id ?? null)
        setError('')
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false))
  }, [tenant, reloadKey])

  const userLabels = useMemo(
    () => Object.fromEntries(users.map((u) => [u.user_id, u.display_name ?? u.user_id])),
    [users],
  )
  const matterLabels = useMemo(
    () => Object.fromEntries(matters.map((m) => [m.matter_id, m.name])),
    [matters],
  )

  // What is in `matterAccess` / `userAccess`. Comparing it to the key the page wants is
  // how the spinner is decided, so no state has to be written before a fetch begins.
  const wantKey = view === 'matter' ? `m:${matterId}:${refreshKey}` : `u:${userId}:${refreshKey}`

  useEffect(() => {
    let cancelled = false
    const id = view === 'matter' ? matterId : userId
    if (!id) return
    const p =
      view === 'matter'
        ? api.getMatterAccess(tenant, id).then((d) => {
            if (!cancelled) {
              setMatterAccess(d)
              setDetailError('')
            }
          })
        : api.getUserAccess(tenant, id).then((d) => {
            if (!cancelled) {
              setUserAccess(d)
              setDetailError('')
            }
          })
    p.catch((e: Error) => {
      if (!cancelled) {
        setMatterAccess(null)
        setUserAccess(null)
        setDetailError(e.message)
      }
    }).finally(() => {
      if (!cancelled) setLoadedKey(wantKey)
    })
    return () => {
      cancelled = true
    }
  }, [tenant, view, matterId, userId, wantKey])

  const detailLoading = loadedKey !== wantKey

  const reload = () => setRefreshKey((k) => k + 1)

  const doAssign = async () => {
    if (!matterId || !addUser) return
    try {
      await api.assign(tenant, { user_id: addUser, matter_id: matterId, role: addRole })
      showToast(`${userLabels[addUser] ?? addUser} added to ${matterLabels[matterId] ?? matterId}`)
    } catch {
      showToast('Could not add that person. Nothing was changed.', 'error')
      return
    } finally {
      setAddUser('')
    }
    reload()
  }

  const notOnTeam = useMemo(() => {
    if (!matterAccess) return users
    const on = new Set(matterAccess.team.map((t) => t.user_id))
    return users.filter((u) => !on.has(u.user_id))
  }, [users, matterAccess])

  const retry = () => {
    setLoading(true)
    setReloadKey((k) => k + 1)
  }

  if (loading) return <Spinner />

  return (
    <>
      <div className="page-header">
        <div className="page-header-row">
          <div>
            <h2>Access</h2>
            <p>
              Nobody reads a matter unless someone put them on it. A screen overrides being on
              the team, and it overrides the administrator role — a wall a senior person can read
              through is not a wall. Every change here is added to the record and never removed.
            </p>
          </div>
        </div>
      </div>

      {error && (
        <ErrorState
          title="Could not load matters or the user directory"
          detail={error}
          onRetry={retry}
        />
      )}

      <div className="access-tabs">
        <button
          className={`access-tab${view === 'matter' ? ' active' : ''}`}
          onClick={() => setView('matter')}
        >
          By matter
        </button>
        <button
          className={`access-tab${view === 'user' ? ' active' : ''}`}
          onClick={() => setView('user')}
        >
          By person
        </button>
      </div>

      {view === 'matter' ? (
        <div className="access-layout">
          <div className="card card-tight">
            <div className="card-header">
              <h3>Matters</h3>
            </div>
            <div className="access-picker">
              {matters.length === 0 && !error && (
                <EmptyState title="No matters">Nothing to staff yet.</EmptyState>
              )}
              {matters.map((m) => (
                <button
                  key={m.matter_id}
                  className={`access-picker-row${m.matter_id === matterId ? ' selected' : ''}`}
                  onClick={() => setMatterId(m.matter_id)}
                >
                  {m.name}
                  <span className="access-picker-sub">{m.matter_id}</span>
                </button>
              ))}
            </div>
          </div>

          <div>
            {detailError ? (
              <ErrorState
                title="Could not load access for this matter"
                detail={detailError}
                onRetry={() => setRefreshKey((k) => k + 1)}
              />
            ) : detailLoading ? (
              <Spinner />
            ) : !matterAccess ? (
              <EmptyState title="Pick a matter">
                Choose a matter on the left to see who may read it.
              </EmptyState>
            ) : (
              <>
                <div className="card">
                  <div className="card-header">
                    <h3>
                      Team
                      <FieldHelp text={HELP.matterAssignment} />
                    </h3>
                    <span className="card-note">
                      {matterAccess.team.length} on {matterAccess.matter_name ?? matterAccess.matter_id}
                    </span>
                  </div>

                  {matterAccess.team.length === 0 ? (
                    <EmptyState title="Nobody is on this matter">
                      Until someone is added, this matter is closed to everyone except holders of
                      the administrator role.
                    </EmptyState>
                  ) : (
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Person</th>
                          <th>Role on the matter</th>
                          <th>Added by</th>
                          <th>Added</th>
                          <th />
                        </tr>
                      </thead>
                      <tbody>
                        {matterAccess.team.map((t) => {
                          const screened = matterAccess.screened.some(
                            (s) => s.user_id === t.user_id,
                          )
                          return (
                            <tr key={t.user_id} className={screened ? 'access-row-screened' : ''}>
                              <td>
                                <strong>{t.display_name ?? t.user_id}</strong>
                                <div className="dim" style={{ fontSize: 11.5 }}>
                                  <code>{t.user_id}</code>
                                </div>
                              </td>
                              <td className="dim">{t.role}</td>
                              <td className="dim">{userLabels[t.granted_by] ?? t.granted_by}</td>
                              <td className="nowrap dim">{fmtDate(t.granted_at)}</td>
                              <td className="nowrap" style={{ textAlign: 'right' }}>
                                {screened ? (
                                  <span className="dim" style={{ fontSize: 11.5 }}>
                                    Screened below
                                  </span>
                                ) : (
                                  <>
                                    <button
                                      className="btn btn-ghost btn-sm"
                                      onClick={() =>
                                        setPending({
                                          kind: 'unassign',
                                          userId: t.user_id,
                                          userLabel: t.display_name ?? t.user_id,
                                          matterId: matterAccess.matter_id,
                                          matterLabel:
                                            matterAccess.matter_name ?? matterAccess.matter_id,
                                        })
                                      }
                                    >
                                      Remove
                                    </button>{' '}
                                    <button
                                      className="btn btn-danger btn-sm"
                                      onClick={() =>
                                        setPending({
                                          kind: 'screen',
                                          userId: t.user_id,
                                          userLabel: t.display_name ?? t.user_id,
                                          matterId: matterAccess.matter_id,
                                          matterLabel:
                                            matterAccess.matter_name ?? matterAccess.matter_id,
                                        })
                                      }
                                    >
                                      Screen
                                    </button>
                                  </>
                                )}
                              </td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  )}

                  <div
                    style={{
                      display: 'flex',
                      gap: 10,
                      alignItems: 'flex-end',
                      marginTop: 16,
                      paddingTop: 16,
                      borderTop: '1px solid var(--border)',
                      flexWrap: 'wrap',
                    }}
                  >
                    <div className="form-group" style={{ marginBottom: 0, minWidth: 240, flex: 1 }}>
                      <label>Add someone to this matter</label>
                      <select value={addUser} onChange={(e) => setAddUser(e.target.value)}>
                        <option value="">Choose a person…</option>
                        {notOnTeam.map((u) => (
                          <option key={u.user_id} value={u.user_id}>
                            {u.display_name ?? u.user_id}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div className="form-group" style={{ marginBottom: 0, minWidth: 190 }}>
                      <label>
                        Role
                        <FieldHelp text="What this person does on the matter. It is recorded for the file and shown in the trail; it does not widen or narrow what they can read." />
                      </label>
                      <select value={addRole} onChange={(e) => setAddRole(e.target.value)}>
                        {ROLES.map((r) => (
                          <option key={r} value={r}>
                            {r}
                          </option>
                        ))}
                      </select>
                    </div>
                    <button className="btn btn-primary" disabled={!addUser} onClick={doAssign}>
                      Add to matter
                    </button>
                  </div>
                </div>

                <div className="card">
                  <div className="card-header">
                    <h3>
                      Screened from this matter
                      <FieldHelp text={HELP.ethicalScreen} />
                    </h3>
                    <span className={`tag ${matterAccess.screened.length ? 'tag-red' : 'tag-neutral'}`}>
                      {matterAccess.screened.length}
                    </span>
                  </div>

                  {matterAccess.screened.length === 0 ? (
                    <p className="card-note">
                      No walls on this matter. Screening someone here tells them the matter by
                      name, gives them the reason, and points them at a contact.
                    </p>
                  ) : (
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Person</th>
                          <th>Reason recorded</th>
                          <th>Raised by</th>
                          <th>Raised</th>
                          <th />
                        </tr>
                      </thead>
                      <tbody>
                        {matterAccess.screened.map((s) => (
                          <tr key={s.user_id} className="access-row-screened">
                            <td>
                              <strong>{s.display_name ?? s.user_id}</strong>
                              <div className="dim" style={{ fontSize: 11.5 }}>
                                <code>{s.user_id}</code>
                              </div>
                              {s.overrides_assignment && (
                                <div style={{ marginTop: 4 }}>
                                  <span className="tag tag-red">Overrides their place on the team</span>
                                </div>
                              )}
                            </td>
                            <td>
                              <div className="access-reason">
                                {s.reason}
                                <span className="access-reason-contact">
                                  {s.contact
                                    ? `They are told to contact ${s.contact}.`
                                    : 'No contact given, they are told to ask their risk team.'}
                                </span>
                              </div>
                            </td>
                            <td className="dim">{userLabels[s.screened_by] ?? s.screened_by}</td>
                            <td className="nowrap dim">{fmtDate(s.screened_at)}</td>
                            <td className="nowrap" style={{ textAlign: 'right' }}>
                              <button
                                className="btn btn-ghost btn-sm"
                                onClick={() =>
                                  setPending({
                                    kind: 'lift',
                                    userId: s.user_id,
                                    userLabel: s.display_name ?? s.user_id,
                                    matterId: matterAccess.matter_id,
                                    matterLabel:
                                      matterAccess.matter_name ?? matterAccess.matter_id,
                                  })
                                }
                              >
                                Lift screen
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}

                  <div
                    style={{
                      display: 'flex',
                      gap: 10,
                      alignItems: 'flex-end',
                      marginTop: 16,
                      paddingTop: 16,
                      borderTop: '1px solid var(--border)',
                      flexWrap: 'wrap',
                    }}
                  >
                    <div className="form-group" style={{ marginBottom: 0, minWidth: 240, flex: 1 }}>
                      <label>Screen someone from this matter</label>
                      <select
                        value=""
                        onChange={(e) => {
                          const id = e.target.value
                          if (!id) return
                          setPending({
                            kind: 'screen',
                            userId: id,
                            userLabel: userLabels[id] ?? id,
                            matterId: matterAccess.matter_id,
                            matterLabel: matterAccess.matter_name ?? matterAccess.matter_id,
                          })
                        }}
                      >
                        <option value="">Choose a person…</option>
                        {users
                          .filter((u) => !matterAccess.screened.some((s) => s.user_id === u.user_id))
                          .map((u) => (
                            <option key={u.user_id} value={u.user_id}>
                              {u.display_name ?? u.user_id}
                            </option>
                          ))}
                      </select>
                      <p className="hint">
                        A screen works whether or not the person is on the team, and it applies to
                        holders of the administrator role too.
                      </p>
                    </div>
                  </div>
                </div>

                <AccessAudit
                  matterId={matterAccess.matter_id}
                  matterNames={matterLabels}
                  userNames={userLabels}
                  refreshKey={refreshKey}
                />
              </>
            )}
          </div>
        </div>
      ) : (
        <div className="access-layout">
          <div className="card card-tight">
            <div className="card-header">
              <h3>People</h3>
            </div>
            <div className="access-picker">
              {users.length === 0 && !error && (
                <EmptyState title="No people in the directory">
                  Invite colleagues from Admin.
                </EmptyState>
              )}
              {users.map((u) => (
                <button
                  key={u.user_id}
                  className={`access-picker-row${u.user_id === userId ? ' selected' : ''}`}
                  onClick={() => setUserId(u.user_id)}
                >
                  {u.display_name ?? u.user_id}
                  <span className="access-picker-sub">{u.user_id}</span>
                </button>
              ))}
            </div>
          </div>

          <div>
            {detailError ? (
              <ErrorState
                title="Could not load access for this person"
                detail={detailError}
                onRetry={() => setRefreshKey((k) => k + 1)}
              />
            ) : detailLoading ? (
              <Spinner />
            ) : !userAccess ? (
              <EmptyState title="Pick a person">
                Choose someone on the left to see what they can reach, and why.
              </EmptyState>
            ) : (
              <>
                <div className="card">
                  <div className="card-header">
                    <h3>
                      What {userAccess.display_name ?? userAccess.user_id} can reach
                      <FieldHelp text={HELP.accessDecision} />
                    </h3>
                    {userAccess.is_platform_admin && (
                      <span className="tag tag-orange">Holds the administrator role</span>
                    )}
                  </div>

                  {userAccess.is_platform_admin && (
                    <div className="banner banner-warn">
                      <span>
                        <strong>Every matter below is open to this person by role.</strong>{' '}
                        <span>
                          Nobody staffed them onto these matters — they can read them because they
                          hold the administrator role. A screen still overrides it, so screen them
                          from anything they must not reach.
                        </span>
                        <FieldHelp text={HELP.platformAdminAccess} />
                      </span>
                    </div>
                  )}

                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Matter</th>
                        <th>
                          Can they read it?
                          <FieldHelp text={HELP.accessDecision} />
                        </th>
                        <th>Role on the matter</th>
                        <th>Why</th>
                        <th />
                      </tr>
                    </thead>
                    <tbody>
                      {userAccess.decisions.map((d) => (
                        <tr key={d.matter_id} className={rowClass(d.decision)}>
                          <td>
                            <strong>{d.matter_name ?? d.matter_id}</strong>
                            <div className="dim" style={{ fontSize: 11.5 }}>
                              <code>{d.matter_id}</code>
                            </div>
                          </td>
                          <td>
                            <DecisionBadge decision={d.decision} />
                          </td>
                          <td className="dim">{d.role ?? '-'}</td>
                          <td>
                            <div className="access-reason">
                              {d.decision === 'SCREENED' ? (
                                <>
                                  {d.reason}
                                  <span className="access-reason-contact">
                                    {d.contact
                                      ? `They are told to contact ${d.contact}.`
                                      : 'They are told to ask their risk team.'}
                                  </span>
                                </>
                              ) : (
                                ACCESS_DECISIONS[d.decision].meaning
                              )}
                            </div>
                          </td>
                          <td className="nowrap" style={{ textAlign: 'right' }}>
                            {d.decision === 'SCREENED' ? (
                              <button
                                className="btn btn-ghost btn-sm"
                                onClick={() =>
                                  setPending({
                                    kind: 'lift',
                                    userId: userAccess.user_id,
                                    userLabel: userAccess.display_name ?? userAccess.user_id,
                                    matterId: d.matter_id,
                                    matterLabel: d.matter_name ?? d.matter_id,
                                  })
                                }
                              >
                                Lift screen
                              </button>
                            ) : (
                              <button
                                className="btn btn-danger btn-sm"
                                onClick={() =>
                                  setPending({
                                    kind: 'screen',
                                    userId: userAccess.user_id,
                                    userLabel: userAccess.display_name ?? userAccess.user_id,
                                    matterId: d.matter_id,
                                    matterLabel: d.matter_name ?? d.matter_id,
                                  })
                                }
                              >
                                Screen
                              </button>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>

                  <p className="card-note" style={{ marginTop: 12 }}>
                    “Not on the team” is not a wall — nobody decided anything, and adding them is
                    an ordinary staffing change. “Screened” is a decision someone recorded, with a
                    reason, and it stands until it is lifted with a reason of its own.
                  </p>
                </div>

                <AccessAudit
                  userId={userAccess.user_id}
                  matterNames={matterLabels}
                  userNames={userLabels}
                  refreshKey={refreshKey}
                />
              </>
            )}
          </div>
        </div>
      )}

      {pending?.kind === 'screen' && (
        <ScreenModal
          userLabel={pending.userLabel}
          matterLabel={pending.matterLabel}
          matterId={pending.matterId}
          onCancel={() => setPending(null)}
          onConfirm={async (reason, contact) => {
            try {
              await api.screen(tenant, {
                user_id: pending.userId,
                matter_id: pending.matterId,
                reason,
                contact: contact || undefined,
              })
              showToast(`${pending.userLabel} is screened from ${pending.matterLabel}`)
            } catch {
              showToast('Could not raise the screen. Nothing was changed.', 'error')
            }
            setPending(null)
            reload()
          }}
        />
      )}

      {pending?.kind === 'lift' && (
        <ReasonModal
          title={`Lift the screen on ${pending.userLabel}?`}
          subtitle={`They will be able to read ${pending.matterLabel} again if they are on the team, or if they hold the administrator role.`}
          consequences={[
            `${pending.userLabel} stops being told they are screened from ${pending.matterLabel}.`,
            'The original screen and this lifting both stay on the record. Neither is removed.',
            'If they are not on the team, lifting the screen alone does not give them access.',
          ]}
          reasonLabel="Why is the screen no longer needed?"
          reasonHelp="Required. Lifting a wall is the change most likely to be questioned, so the file has to show what changed and who decided it."
          placeholder="File review completed: the earlier engagement was for an unrelated party."
          confirmLabel="Lift screen"
          danger={false}
          onCancel={() => setPending(null)}
          onConfirm={async (reason) => {
            try {
              await api.liftScreen(tenant, {
                user_id: pending.userId,
                matter_id: pending.matterId,
                reason,
              })
              showToast(`Screen lifted for ${pending.userLabel}`)
            } catch {
              showToast('Could not lift the screen. Nothing was changed.', 'error')
            }
            setPending(null)
            reload()
          }}
        />
      )}

      {pending?.kind === 'unassign' && (
        <ReasonModal
          title={`Remove ${pending.userLabel} from ${pending.matterLabel}?`}
          subtitle="Removing someone from the team is not the same as screening them. Use a screen where there is a conflict."
          consequences={[
            `${pending.userLabel} loses access to the documents, facts and figures on ${pending.matterLabel}.`,
            'They are told they are not on the matter, and to ask the matter owner if they need it.',
            'The record that they were on the matter, and who added them, stays in place.',
            'This is not a wall. Anyone can put them back on the matter without a review.',
          ]}
          reasonLabel="Why are they coming off the matter?"
          reasonHelp="Optional for the API, asked for here because a removal with no explanation is indistinguishable from a mistake six months later."
          placeholder="Moved to another matter and no longer working on this one."
          confirmLabel="Remove from matter"
          danger
          reasonRequired={false}
          onCancel={() => setPending(null)}
          onConfirm={async (reason) => {
            try {
              await api.unassign(tenant, {
                user_id: pending.userId,
                matter_id: pending.matterId,
                reason,
              })
              showToast(`${pending.userLabel} removed from ${pending.matterLabel}`)
            } catch {
              showToast('Could not remove that person. Nothing was changed.', 'error')
            }
            setPending(null)
            reload()
          }}
        />
      )}

      <Toast toast={toast} />
    </>
  )
}

function rowClass(d: AccessDecision): string {
  if (d === 'PLATFORM_ADMIN') return 'access-row-by-role'
  if (d === 'SCREENED') return 'access-row-screened'
  return ''
}

function DecisionBadge({ decision }: { decision: AccessDecision }) {
  const meta = ACCESS_DECISIONS[decision]
  const modifier =
    decision === 'SCREENED'
      ? ' is-screened'
      : decision === 'NOT_ASSIGNED'
        ? ' is-not-assigned'
        : ''
  return (
    <span
      className={`access-decision${modifier}`}
      style={{ ['--decision-colour' as string]: meta.colour }}
      title={`${meta.meaning} ${meta.action}`}
    >
      <span className="access-decision-dot" aria-hidden="true" />
      {meta.label}
    </span>
  )
}

/**
 * Raising a wall. The reason is required in the interface as well as in the API:
 * belt-and-braces, because a blank reason makes the wall indefensible when someone
 * asks about it, and by then the person who raised it may have left.
 */
function ScreenModal({
  userLabel,
  matterLabel,
  matterId,
  onCancel,
  onConfirm,
}: {
  userLabel: string
  matterLabel: string
  matterId: string
  onCancel: () => void
  onConfirm: (reason: string, contact: string) => Promise<void>
}) {
  const [reason, setReason] = useState('')
  const [contact, setContact] = useState('')
  const [saving, setSaving] = useState(false)
  const ready = reason.trim().length > 0

  const submit = async () => {
    if (!ready) return
    setSaving(true)
    await onConfirm(reason.trim(), contact.trim())
    setSaving(false)
  }

  return (
    <div className="modal-overlay" onClick={onCancel}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h3>Screen {userLabel} from {matterLabel}?</h3>
        <p className="modal-sub">
          A screen overrides their place on the team and overrides the administrator role. It
          takes effect immediately, not at their next sign-in.
        </p>

        <div className="consequence">
          <div className="consequence-title">What this person will be told</div>
          <ul>
            <li>
              They will be told, by name, that they are screened from{' '}
              <strong>{matterLabel}</strong>.
            </li>
            <li>They will see the reason you write below, word for word.</li>
            <li>
              They will be sent to the contact you give, or to their risk team if you leave it
              blank.
            </li>
            <li>
              This is deliberate. Hiding the matter instead would let a conflict check come back
              clean because the matching matter was invisible.
            </li>
          </ul>
          {ready && (
            <div className="consequence-preview">
              <span className="consequence-preview-label">They will read</span>
              You are screened from {matterId}. Reason recorded: {reason.trim()}{' '}
              {contact.trim() ? `Contact ${contact.trim()} to discuss.` : 'Contact your risk team.'}
            </div>
          )}
        </div>

        <div className="form-group">
          <label>
            Reason — required
            <FieldHelp text="Written for whoever reads the file in a year, including the person screened. Name the conflict rather than describing the outcome: “acted for the counterparty in 2024” explains itself, “risk decision” does not." />
          </label>
          <textarea
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            placeholder="Acted for the counterparty, Brannigan Aggregates Ltd, in 2024."
            autoFocus
          />
          {!ready && (
            <p className="hint">
              A screen cannot be saved without a reason. An unexplained wall cannot be defended
              when someone asks why it exists.
            </p>
          )}
        </div>

        <div className="form-group">
          <label>
            Who should they contact?
            <FieldHelp text="Shown to them exactly as you type it. A named person or team gets the conversation started; leaving it blank sends them to their risk team, which is slower." />
          </label>
          <input
            value={contact}
            onChange={(e) => setContact(e.target.value)}
            placeholder="r.okonjo@thornevaux.example (Risk)"
          />
        </div>

        <div className="modal-actions">
          <button className="btn btn-ghost" onClick={onCancel}>
            Cancel
          </button>
          <button className="btn btn-danger" disabled={!ready || saving} onClick={submit}>
            {saving ? 'Saving…' : 'Raise screen'}
          </button>
        </div>
      </div>
    </div>
  )
}

/** Confirmation for a change that states its consequence and records why it was made. */
function ReasonModal({
  title,
  subtitle,
  consequences,
  reasonLabel,
  reasonHelp,
  placeholder,
  confirmLabel,
  danger,
  reasonRequired = true,
  onCancel,
  onConfirm,
}: {
  title: string
  subtitle: string
  consequences: string[]
  reasonLabel: string
  reasonHelp: string
  placeholder: string
  confirmLabel: string
  danger: boolean
  reasonRequired?: boolean
  onCancel: () => void
  onConfirm: (reason: string) => Promise<void>
}) {
  const [reason, setReason] = useState('')
  const [saving, setSaving] = useState(false)
  const ready = !reasonRequired || reason.trim().length > 0

  const submit = async () => {
    if (!ready) return
    setSaving(true)
    await onConfirm(reason.trim())
    setSaving(false)
  }

  return (
    <div className="modal-overlay" onClick={onCancel}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h3>{title}</h3>
        <p className="modal-sub">{subtitle}</p>

        <div className="consequence">
          <div className="consequence-title">What happens</div>
          <ul>
            {consequences.map((c) => (
              <li key={c}>{c}</li>
            ))}
          </ul>
        </div>

        <div className="form-group">
          <label>
            {reasonLabel}
            {reasonRequired ? ', required' : ', recommended'}
            <FieldHelp text={reasonHelp} />
          </label>
          <textarea
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            placeholder={placeholder}
            autoFocus
          />
        </div>

        <div className="modal-actions">
          <button className="btn btn-ghost" onClick={onCancel}>
            Cancel
          </button>
          <button
            className={`btn ${danger ? 'btn-danger' : 'btn-primary'}`}
            disabled={!ready || saving}
            onClick={submit}
          >
            {saving ? 'Saving…' : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  )
}
