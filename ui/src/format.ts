/** Formatting helpers. Kept out of the component modules so fast refresh works. */

import type { CSSProperties } from 'react'
import type { EpistemicClass } from './api'
import { EPISTEMIC } from './epistemic'

/** Supplies --epi-colour to any element styled by epistemic class. */
export function epiStyle(c: EpistemicClass): CSSProperties {
  return { '--epi-colour': EPISTEMIC[c].colour } as CSSProperties
}

/** `party:acme-corporation` -> `acme corporation`. An Entity node carries no label property, so
 *  the slug inside the id is the only name there is. Mirrors `_entity_label` server-side. */
export function entityLabel(id: string): string {
  const slug = id.includes(':') ? id.slice(id.indexOf(':') + 1) : id
  return slug.replace(/[-_]/g, ' ')
}

/** The kind before the colon. Empty when the id carries none, rather than guessed. */
export function entityKind(id: string): string {
  return id.includes(':') ? id.slice(0, id.indexOf(':')) : ''
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
