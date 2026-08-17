/** Small presentational pieces used on more than one page. */

import type { ReactNode } from 'react'
import type { IngestState } from '../api'
import { INGEST_STATES } from '../api'
import { INGEST_STEP_HELP, INGEST_STEP_LABEL, failureStep, ingestPhase, tierMeta } from '../epistemic'
import { isPlatformAdmin } from '../auth'

export function Spinner() {
  return (
    <div className="loading">
      <div className="spinner" />
    </div>
  )
}

export function Toast({ toast }: { toast: { msg: string; type: string } | null }) {
  if (!toast) return null
  return (
    <div className={`toast toast-${toast.type}`} role="status">
      {toast.msg}
    </div>
  )
}

/** A failed request. Says nothing loaded, so an empty page is never read as no data. */
export function ErrorState({
  title = 'Could not load this page',
  detail,
  onRetry,
}: {
  title?: string
  detail?: string
  onRetry?: () => void
}) {
  return (
    <div className="banner banner-error">
      <span>
        <strong>{title}.</strong> {detail ? `${detail}. ` : ''}Nothing is shown below because
        nothing was loaded.{' '}
        {onRetry && (
          <button className="btn btn-ghost btn-sm" onClick={onRetry} style={{ marginLeft: 4 }}>
            Try again
          </button>
        )}
      </span>
    </div>
  )
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="empty-state">
      <strong>{title}</strong>
      {children}
    </div>
  )
}

/**
 * Renders `children` only for a platform admin. Presentation, not enforcement: the
 * routes underneath answer 403 regardless, and this only spares someone who reached the
 * URL directly a page of failed requests.
 */
export function AdminOnly({ children }: { children: ReactNode }) {
  if (isPlatformAdmin()) return <>{children}</>
  return (
    <EmptyState title="You do not have access to this page">
      <p>
        Administration requires the platform-admin role. Ask an administrator at your firm if you
        need it.
      </p>
    </EmptyState>
  )
}

export function IngestPill({ state }: { state: IngestState }) {
  const phase = ingestPhase(state)
  const failedAt = failureStep(state)
  const label = failedAt
    ? `${INGEST_STEP_LABEL[failedAt]} failed`
    : INGEST_STEP_LABEL[state] || state
  const help = failedAt
    ? `The document failed while ${INGEST_STEP_LABEL[failedAt].toLowerCase()}. Nothing from it has entered the graph.`
    : INGEST_STEP_HELP[state]

  return (
    <span className={`ingest-pill state-${phase}`} title={help}>
      {phase === 'running' && <span className="spin-dot" aria-hidden="true" />}
      {label}
    </span>
  )
}

/** The ingest state machine drawn as a track, with the current step marked. */
export function Pipeline({ state }: { state: IngestState }) {
  const failedAt = failureStep(state)
  const currentIdx = failedAt
    ? INGEST_STATES.indexOf(failedAt as (typeof INGEST_STATES)[number])
    : INGEST_STATES.indexOf(state as (typeof INGEST_STATES)[number])

  return (
    <div className="pipeline">
      {INGEST_STATES.map((s, i) => {
        const done = currentIdx > i
        const current = currentIdx === i && !failedAt
        const failed = currentIdx === i && !!failedAt
        const cls = failed ? 'failed' : current ? 'current' : done ? 'done' : ''
        return (
          <div key={s} style={{ display: 'flex', alignItems: 'center', flex: i === 0 ? '0 0 auto' : 1 }}>
            {i > 0 && <div className={`pipeline-connector${done || current ? ' done' : ''}`} />}
            <div className={`pipeline-step ${cls}`} title={INGEST_STEP_HELP[s]}>
              <div className="pipeline-node">
                {failed ? '✕' : done ? '✓' : i + 1}
              </div>
              <div className="pipeline-label">{INGEST_STEP_LABEL[s]}</div>
            </div>
          </div>
        )
      })}
    </div>
  )
}

/** `number`, not `ResolutionTier`: the question log is append-only and holds retired tiers. */
export function TierBadge({ tier }: { tier: number }) {
  const meta = tierMeta(tier)
  if (meta === null) return <span className="tag tag-neutral">Tier {tier}</span>
  return (
    <span
      className="tag"
      style={{
        color: meta.colour,
        background: `color-mix(in srgb, ${meta.colour} 12%, transparent)`,
      }}
      title={meta.detail}
    >
      Tier {tier} &middot; {meta.label}
    </span>
  )
}
