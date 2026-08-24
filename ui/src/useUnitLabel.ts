import { useEffect, useState } from 'react'
import { api } from './api'
import { getTenantId } from './auth'

/**
 * What this tenant's ontology pack calls the unit work is organised by.
 *
 * "Matter" is the legal pack's word for it. Healthcare calls it an Encounter and lending a
 * Facility, so hardcoding it made a whitelabel platform read as a legal one. The scoping key is
 * still `matter_id` throughout the API and the graph -- renaming that would touch Cedar, a Cognito
 * group and a Neptune constraint to change a caption.
 *
 * Cached per tenant at module scope because it changes only when an admin switches pack, and every
 * page that renders a heading would otherwise refetch settings. Falls back to Matter, which is what
 * the default pack calls it, so a page never renders an empty heading while the fetch is in flight.
 */
export type UnitLabel = { singular: string; plural: string }

const FALLBACK: UnitLabel = { singular: 'Matter', plural: 'Matters' }

const cache = new Map<string, UnitLabel>()
const inFlight = new Map<string, Promise<UnitLabel>>()

/** Drop the cached label, so the next read reflects a pack the admin just switched. */
export function forgetUnitLabel(tenant: string = getTenantId()): void {
  cache.delete(tenant)
  inFlight.delete(tenant)
}

function load(tenant: string): Promise<UnitLabel> {
  const running = inFlight.get(tenant)
  if (running) return running
  const p = api
    .getSettings(tenant)
    .then((s) => {
      const label = s.unit_label ?? FALLBACK
      cache.set(tenant, label)
      return label
    })
    .catch(() => FALLBACK)
    .finally(() => inFlight.delete(tenant))
  inFlight.set(tenant, p)
  return p
}

export function useUnitLabel(): UnitLabel {
  const tenant = getTenantId()
  // Tagged with the tenant it belongs to, so a cached value is read during render rather than
  // pushed in from an effect -- and a tenant switch reads as "not loaded" instead of briefly
  // showing the previous tenant's wording.
  const [loaded, setLoaded] = useState<{ tenant: string; label: UnitLabel } | null>(() => {
    const hit = cache.get(tenant)
    return hit ? { tenant, label: hit } : null
  })

  useEffect(() => {
    let live = true
    load(tenant).then((label) => {
      if (live) setLoaded({ tenant, label })
    })
    return () => {
      live = false
    }
  }, [tenant])

  const fresh = loaded?.tenant === tenant ? loaded.label : cache.get(tenant)
  return fresh ?? FALLBACK
}
