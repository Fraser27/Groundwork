/**
 * The access trail: who changed whose access to which matter, why, and when.
 *
 * Read-only by construction — there is no mutating call in this file, and nothing is
 * passed in that could make one. That is the point: the trail is the compliance
 * artifact, so it is a first-class view rather than a debug panel, and an interface
 * that could edit it would make it worth less than the log it replaced.
 */

import { useEffect, useMemo, useState } from 'react'
import { api, type AccessEvent } from '../api'
import { getTenantId } from '../auth'
import { ACCESS_ACTION_LABEL, HELP } from '../epistemic'
import { fallback, mockAccessAudit } from '../mocks'
import { fmtDateTime } from '../format'
import FieldHelp from './FieldHelp'
import { EmptyState, Spinner } from './Shared'

const ACTION_COLOUR: Record<AccessEvent['action'], string> = {
  ASSIGN: 'var(--green)',
  UNASSIGN: 'var(--text-dim)',
  SCREEN: 'var(--red)',
  LIFT_SCREEN: 'var(--orange)',
}

/** A screen and its lifting are the entries a reviewer is looking for. */
const ACTIONS: (AccessEvent['action'] | 'ALL')[] = ['ALL', 'SCREEN', 'LIFT_SCREEN', 'ASSIGN', 'UNASSIGN']

export default function AccessAudit({
  matterId,
  userId,
  matterNames = {},
  userNames = {},
  /** Bumped by the parent after a change, so the trail reflects it without a reload. */
  refreshKey = 0,
}: {
  matterId?: string
  userId?: string
  matterNames?: Record<string, string>
  userNames?: Record<string, string>
  refreshKey?: number
}) {
  const tenant = getTenantId()
  const [events, setEvents] = useState<AccessEvent[]>([])
  // What the rows in state describe. Comparing it to the current scope is how the
  // spinner is decided, so nothing has to be set before the fetch starts.
  const [loadedFor, setLoadedFor] = useState<string | null>(null)
  const [action, setAction] = useState<(typeof ACTIONS)[number]>('ALL')

  const scopeKey = `${matterId ?? ''}|${userId ?? ''}|${refreshKey}`

  useEffect(() => {
    let cancelled = false
    const scope = { matter_id: matterId, user_id: userId }
    fallback(api.accessAudit(tenant, scope), mockAccessAudit(scope))
      .then((e) => {
        if (!cancelled) setEvents(e)
      })
      .catch(() => {
        if (!cancelled) setEvents([])
      })
      .finally(() => {
        if (!cancelled) setLoadedFor(scopeKey)
      })
    return () => {
      cancelled = true
    }
  }, [tenant, matterId, userId, scopeKey])

  const loading = loadedFor !== scopeKey

  const shown = useMemo(() => {
    const filtered = action === 'ALL' ? events : events.filter((e) => e.action === action)
    return [...filtered].sort((a, b) => b.at.localeCompare(a.at))
  }, [events, action])

  const scopeLabel = matterId
    ? `${matterNames[matterId] ?? matterId}`
    : userId
      ? `${userNames[userId] ?? userId}`
      : 'the whole firm'

  return (
    <div className="card">
      <div className="card-header">
        <h3>
          Access trail
          <FieldHelp text={HELP.accessAudit} />
        </h3>
        <div className="access-tabs" style={{ margin: 0, border: 'none' }}>
          {ACTIONS.map((a) => (
            <button
              key={a}
              className={`access-tab${action === a ? ' active' : ''}`}
              onClick={() => setAction(a)}
            >
              {a === 'ALL' ? 'Everything' : ACCESS_ACTION_LABEL[a]}
            </button>
          ))}
        </div>
      </div>

      <p className="card-note" style={{ marginBottom: 12 }}>
        Every change to who may read what, for {scopeLabel}. Entries are added and never
        altered, so removing someone and lifting a screen both stay on the record.
      </p>

      {loading ? (
        <Spinner />
      ) : shown.length === 0 ? (
        <EmptyState title="No changes recorded">
          Nothing has been added, removed or screened here.
        </EmptyState>
      ) : (
        <table className="data-table">
          <thead>
            <tr>
              <th>When</th>
              <th>Change</th>
              <th>Person affected</th>
              {!matterId && <th>Matter</th>}
              <th>
                Made by
                <FieldHelp text="The signed-in person who made the change. Recorded from the verified session, not typed in." />
              </th>
              <th>
                Reason given
                <FieldHelp text="Required when a screen is raised or lifted. A blank reason would make the wall impossible to defend when someone asks about it later." />
              </th>
            </tr>
          </thead>
          <tbody>
            {shown.map((e) => (
              <tr key={e.event_id}>
                <td className="nowrap dim">{fmtDateTime(e.at)}</td>
                <td>
                  <span
                    className="audit-action"
                    style={{ ['--audit-colour' as string]: ACTION_COLOUR[e.action] }}
                  >
                    <span className="audit-action-bar" aria-hidden="true" />
                    {ACCESS_ACTION_LABEL[e.action] ?? e.action}
                  </span>
                  {typeof e.detail?.role === 'string' && (
                    <div className="dim" style={{ fontSize: 11, marginTop: 2 }}>
                      as {e.detail.role}
                    </div>
                  )}
                </td>
                <td className="audit-actor">
                  {userNames[e.subject_user] ?? e.subject_user}
                  {userNames[e.subject_user] && (
                    <div>
                      <code>{e.subject_user}</code>
                    </div>
                  )}
                </td>
                {!matterId && (
                  <td>
                    {matterNames[e.matter_id] ?? e.matter_id}
                    <div className="dim" style={{ fontSize: 11 }}>
                      <code>{e.matter_id}</code>
                    </div>
                  </td>
                )}
                <td className="audit-actor">
                  {userNames[e.actor] ?? e.actor}
                  {userNames[e.actor] && (
                    <div>
                      <code>{e.actor}</code>
                    </div>
                  )}
                </td>
                <td>
                  {e.reason ? (
                    <div className="audit-reason">
                      {e.reason}
                      {typeof e.detail?.contact === 'string' && (
                        <span className="access-reason-contact">
                          Contact given: {e.detail.contact}
                        </span>
                      )}
                    </div>
                  ) : (
                    <span className="audit-reason audit-reason-missing">
                      {e.action === 'ASSIGN' ? 'Not required for adding someone' : 'None recorded'}
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
