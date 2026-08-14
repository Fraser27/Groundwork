import { useEffect, useState } from 'react'
import { api, type Provenance } from './api'
import { fallback, mockProvenance } from './mocks'

/**
 * Load full provenance for one assertion.
 *
 * The loaded value is tagged with the id it belongs to, and a mismatch reads as
 * null. That is what stops the previous assertion's proof tree flashing up under a
 * newly selected one, without needing to clear state on the way out.
 */
export function useProvenance(tenant: string, assertionId: string | null): Provenance | null {
  const [loaded, setLoaded] = useState<{ id: string; data: Provenance } | null>(null)

  useEffect(() => {
    if (!assertionId) return
    let live = true
    fallback(api.getProvenance(tenant, assertionId), mockProvenance(assertionId))
      .then((data) => {
        if (live) setLoaded({ id: assertionId, data })
      })
      .catch(() => {})
    return () => {
      live = false
    }
  }, [tenant, assertionId])

  return assertionId && loaded?.id === assertionId ? loaded.data : null
}
