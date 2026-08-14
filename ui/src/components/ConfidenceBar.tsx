/**
 * ConfidenceBar — a score with the retrieval trust floor marked on the track.
 *
 * Showing the floor matters: "0.72" means nothing on its own, but "0.72, below
 * the 0.8 floor" tells a reviewer that this fact is currently not shaping any
 * answer. Without the tick a reader has to remember the threshold.
 */

import type { CSSProperties } from 'react'

export default function ConfidenceBar({
  value,
  floor,
  width = 68,
  showValue = true,
}: {
  value: number
  /** Retrieval trust floor. Drawn as a tick, and the value turns red below it. */
  floor?: number
  width?: number
  showValue?: boolean
}) {
  const pct = Math.max(0, Math.min(1, value)) * 100
  const band = value >= 0.85 ? 'high' : value >= 0.6 ? 'mid' : 'low'
  const below = floor !== undefined && value < floor

  const title = [
    `Confidence ${value.toFixed(2)}`,
    floor !== undefined
      ? below
        ? `below the ${floor.toFixed(2)} trust floor — visible for review, but excluded from answers`
        : `at or above the ${floor.toFixed(2)} trust floor`
      : null,
  ]
    .filter(Boolean)
    .join(' — ')

  return (
    <span className="conf" title={title}>
      <span
        className="conf-track"
        style={{ '--conf-width': `${width}px` } as CSSProperties}
        role="meter"
        aria-valuenow={Number(value.toFixed(2))}
        aria-valuemin={0}
        aria-valuemax={1}
        aria-label={title}
      >
        <span className={`conf-fill ${band}`} style={{ width: `${pct}%` }} />
        {floor !== undefined && (
          <span className="conf-floor" style={{ left: `${floor * 100}%` }} aria-hidden="true" />
        )}
      </span>
      {showValue && (
        <span className={`conf-value${below ? ' conf-below' : ''}`}>{value.toFixed(2)}</span>
      )}
    </span>
  )
}
