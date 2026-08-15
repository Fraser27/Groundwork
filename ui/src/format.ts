/** Formatting helpers. Kept out of the component modules so fast refresh works. */

import type { CSSProperties } from 'react'
import type { EpistemicClass } from './api'
import { EPISTEMIC } from './epistemic'

/** Supplies --epi-colour to any element styled by epistemic class. */
export function epiStyle(c: EpistemicClass): CSSProperties {
  return { '--epi-colour': EPISTEMIC[c].colour } as CSSProperties
}

export function fmtBytes(n?: number | null): string {
  if (n == null) return '-'
  const units = ['B', 'KB', 'MB', 'GB']
  let v = n
  let u = 0
  while (v >= 1024 && u < units.length - 1) {
    v /= 1024
    u++
  }
  return `${v.toFixed(u === 0 ? 0 : 1)} ${units[u]}`
}

export function fmtNum(n?: number | null): string {
  if (n == null) return '-'
  return n.toLocaleString()
}

export function fmtDate(s?: string | null): string {
  if (!s) return '-'
  return new Date(s).toLocaleDateString(undefined, {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
  })
}

export function fmtDateTime(s?: string | null): string {
  if (!s) return '-'
  return new Date(s).toLocaleString(undefined, {
    day: '2-digit',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
  })
}
