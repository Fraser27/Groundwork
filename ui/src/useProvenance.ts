import { useEffect, useState } from 'react'
import { api, type Provenance } from './api'

/**
 * Load full provenance for one assertion.
 *
 * The loaded value is tagged with the id it belongs to, and a mismatch reads as
 * null. That is what stops the previous assertion's proof tree flashing up under a
 * newly selected one, without needing to clear state on the way out.
 */
export function useProvenance(
  tenant: string,
  assertionId: string | null,
): { provenance: Provenance | null; error: string | null } {
  const [loaded, setLoaded] = useState<{ id: string; data: Provenance | null; error: string | null }>(
    { id: '', data: null, error: null },
  )

  useEffect(() => {
    if (!assertionId) return
    let live = true
    api
      .getProvenance(tenant, assertionId)
      .then((data) => {
        if (live) setLoaded({ id: assertionId, data, error: null })
      })
      .catch((e: Error) => {
        if (live) setLoaded({ id: assertionId, data: null, error: e.message })
      })
    return () => {
      live = false
    }
  }, [tenant, assertionId])

  const current = assertionId && loaded.id === assertionId
  return {
    provenance: current ? loaded.data : null,
    error: current ? loaded.error : null,
  }
}
