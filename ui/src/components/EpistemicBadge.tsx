/**
 * EpistemicBadge — the single most repeated element in the UI.
 *
 * Every fact shown to a user carries one, in a colour that is consistent across
 * the dashboard, the review queue, the graph and the audit view. The tooltip is
 * not decoration: it says what the class means and how much weight to give it,
 * because "EXTRACTED_MODEL" tells an unprepared reader nothing.
 */

import type { CSSProperties } from 'react'
import type { EpistemicClass } from '../api'
import { EPISTEMIC } from '../epistemic'

export default function EpistemicBadge({
  epistemicClass,
  size = 'md',
  showLabel = true,
  tipPlacement = 'below',
  tipAlign = 'left',
}: {
  epistemicClass: EpistemicClass
  size?: 'sm' | 'md'
  showLabel?: boolean
  tipPlacement?: 'below' | 'above'
  tipAlign?: 'left' | 'right'
}) {
  const meta = EPISTEMIC[epistemicClass]
  const style = { '--epi-colour': meta.colour } as CSSProperties

  return (
    <span
      className={[
        'epi-badge',
        size === 'sm' ? 'epi-badge-sm' : '',
        epistemicClass === 'PREDICTED' ? 'epi-badge-predicted' : '',
      ]
        .filter(Boolean)
        .join(' ')}
      style={style}
      tabIndex={0}
      role="note"
      aria-label={`${meta.label}. ${meta.meaning} ${meta.trust}`}
    >
      <span className="epi-badge-dot" aria-hidden="true" />
      {showLabel && meta.label}
      <span
        className={[
          'epi-badge-tip',
          tipPlacement === 'above' ? 'epi-badge-tip-above' : '',
          tipAlign === 'right' ? 'epi-badge-tip-right' : '',
        ]
          .filter(Boolean)
          .join(' ')}
        role="tooltip"
      >
        <span className="epi-tip-title">
          {meta.label} &middot; {epistemicClass}
        </span>
        {meta.meaning}
        <span className="epi-tip-trust">{meta.trust}</span>
      </span>
    </span>
  )
}
