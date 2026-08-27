import { useEffect, useState } from 'react'
import { api } from './api'
import { getTenantId, isAuthenticated } from './auth'

/**
 * What this tenant's ontology pack calls things: the organising unit's name, and the questions
 * worth asking of its data.
 *
 * Both live here because both come from `/settings`, both change at exactly one moment -- an admin
 * switching pack -- and both were hardcoded English before. One fetch, one cache, one invalidation
 * path. A second module would mean a second round trip on every page and a second thing to
 * remember to invalidate.
 *
 * "Matter" is the legal pack's word for it. Healthcare calls it an Encounter, lending a Facility and
 * retail a Case, so hardcoding it made a whitelabel platform read as a legal one. The scoping key is
 * still `matter_id` throughout the API and the graph -- renaming that would touch Cedar, a Cognito
 * group and a Neptune constraint to change a caption.
 *
 * **This is the only place the word may come from.** The pack declares it, `_unit_label` projects it
 * onto `/settings`, and every caption, placeholder and sentence in the UI reads it from here. A page
 * that spells it out instead is a page that lies the moment an admin switches pack -- which is
 * exactly how the navigation came to say Facilities while the Documents page still asked for a
 * matter, on the same screen, from the same tenant.
 *
 * Cached per tenant at module scope because it changes only when an admin switches pack, and every
 * page that renders a heading would otherwise refetch settings.
 */
export type UnitLabel = {
  /** Title case, for headings and buttons: `Facility`. */
  singular: string
  /** Title case plural, for navigation and tab labels: `Facilities`. */
  plural: string
  /** Mid-sentence: `... inherits its facility`. */
  lower: string
  /** Mid-sentence plural: `... groups facts by facilities`. */
  lowerPlural: string
}

/** What the server sends. The lowercase forms are derived rather than transmitted. */
type Wording = { singular: string; plural: string }

const FALLBACK: Wording = { singular: 'Matter', plural: 'Matters' }

/** Lowercased rather than pluralised. Case folding is safe in a way adding `s` is not: a pack
 *  declaring `Care Episode` gets `care episode`, while a naive pluraliser gets `Facilitys`. */
function derive(w: Wording): UnitLabel {
  return {
    singular: w.singular,
    plural: w.plural,
    lower: w.singular.toLowerCase(),
    lowerPlural: w.plural.toLowerCase(),
  }
}

/**
 * Substitute the unit noun into a string written at module scope.
 *
 * A hook cannot be called from a `const`, and the help text, glossary and action labels in
 * `epistemic.ts` are all consts read from a hundred call sites. Rewriting them as functions of
 * `UnitLabel` would mean threading the label through every one. Placeholders keep the strings
 * declarative and put the substitution in the few components that render them.
 *
 * `{unit}` facility, `{units}` facilities, `{Unit}` Facility, `{Units}` Facilities.
 */
export function fillUnit(text: string, unit: UnitLabel): string {
  // Single pass, so a substituted word is never itself rescanned.
  return text.replace(/\{(unit|units|Unit|Units)\}/g, (m) =>
    m === '{unit}'
      ? unit.lower
      : m === '{units}'
        ? unit.lowerPlural
        : m === '{Unit}'
          ? unit.singular
          : unit.plural,
  )
}

/** `fillUnit` bound to this tenant's wording, for components rendering stored text. */
export function useUnitText(): (text: string) => string {
  const unit = useUnitLabel()
  return (text: string) => fillUnit(text, unit)
}

/** Everything `/settings` tells us about how this pack words itself. */
type Pack = { unit: UnitLabel; questions: string[] }

const cache = new Map<string, Pack>()
const inFlight = new Map<string, Promise<Pack>>()

// Persisted, so a reload paints the right word on the first frame. The module cache is empty on
// every page load, so without this the fallback rendered first and the heading changed from Matters
// to Facilities a few hundred milliseconds later -- read, reasonably, as the pack not having taken
// effect. Wording is not a secret and not authoritative: it is re-fetched on mount regardless, and a
// stale entry is corrected within one round trip.
const KEY = 'groundwork.unitLabel'

function persisted(tenant: string): Pack | null {
  try {
    const raw = localStorage.getItem(`${KEY}.${tenant}`)
    if (!raw) return null
    const parsed = JSON.parse(raw) as Partial<Wording> & { questions?: unknown }
    if (typeof parsed?.singular !== 'string' || typeof parsed?.plural !== 'string') return null
    const questions = Array.isArray(parsed.questions)
      ? parsed.questions.filter((q): q is string => typeof q === 'string')
      : []
    return { unit: derive({ singular: parsed.singular, plural: parsed.plural }), questions }
  } catch {
    // A quota error, a disabled store, or wording somebody hand-edited. None of it is worth
    // failing a render over when the fetch is already in flight.
    return null
  }
}

function persist(tenant: string, w: Wording, questions: string[]): void {
  try {
    localStorage.setItem(`${KEY}.${tenant}`, JSON.stringify({ ...w, questions }))
  } catch {
    /* see `persisted` */
  }
}

// Every mounted hook, so switching pack repaints the whole app rather than the one page that
// happens to remount next. Dropping the cache alone was not enough: the navigation had already
// read "Matter" into its own state and had no reason to re-run, so the new wording appeared only
// after a reload -- the second half of "Facilities is only seen when I refreshed".
const listeners = new Set<() => void>()

// Bumped by `forgetUnitLabel`. A fetch started before a pack switch resolves after it, and its
// `.then` would write the old pack's wording straight back into the cache and localStorage that
// were just cleared -- so the admin switches pack, the app repaints, and one round trip later the
// previous pack's word returns with nothing left to invalidate it.
let generation = 0

/** Drop the cached wording and re-read it everywhere, for a pack the admin just switched. */
export function forgetUnitLabel(tenant: string = getTenantId()): void {
  generation += 1
  cache.delete(tenant)
  inFlight.delete(tenant)
  try {
    localStorage.removeItem(`${KEY}.${tenant}`)
  } catch {
    /* see `persisted` */
  }
  // Copied, because a listener that unsubscribes while being notified would otherwise mutate
  // the set mid-iteration.
  for (const notify of [...listeners]) notify()
}

function load(tenant: string): Promise<Pack> {
  const running = inFlight.get(tenant)
  if (running) return running
  const started = generation
  const p = api
    .getSettings(tenant)
    .then((s) => {
      const wording = s.unit_label ?? FALLBACK
      // Empty rather than a hardcoded default. A pack declaring no question should show none: the
      // fallbacks here would be legal ones, which is the whole problem being fixed.
      const questions = s.example_questions ?? []
      const pack: Pack = { unit: derive(wording), questions }
      if (started === generation) {
        cache.set(tenant, pack)
        persist(tenant, wording, questions)
      }
      // Returned either way. The caller is `usePack`, which discards a resolution it no longer
      // wants; what must not happen is this reply outliving the switch in shared state.
      return pack
    })
    .catch(() => ({ unit: derive(FALLBACK), questions: [] }))
    // Only if this is still the registered fetch. `forgetUnitLabel` dropped ours and a
    // replacement may already sit under this tenant, which an unconditional delete would evict
    // and so let a third fetch start against the same pack.
    .finally(() => {
      if (inFlight.get(tenant) === p) inFlight.delete(tenant)
    })
  inFlight.set(tenant, p)
  return p
}

function seed(tenant: string): Pack | null {
  const hit = cache.get(tenant)
  if (hit) return hit
  const stored = persisted(tenant)
  if (stored) cache.set(tenant, stored)
  return stored
}

/**
 * Shared machinery for both hooks: a cached read during render, a fetch on mount, and a
 * re-fetch when `forgetUnitLabel` fires.
 *
 * Tagged with the tenant it belongs to, so a cached value is read during render rather than
 * pushed in from an effect -- and a tenant switch reads as "not loaded" instead of briefly
 * showing the previous tenant's wording.
 */
function usePack(): Pack | null {
  const tenant = getTenantId()
  const [loaded, setLoaded] = useState<{ tenant: string; pack: Pack } | null>(() => {
    const hit = seed(tenant)
    return hit ? { tenant, pack: hit } : null
  })
  // Bumped by `forgetUnitLabel`, which is the only thing that invalidates wording. Carried in the
  // effect's dependencies so a pack switch re-fetches here without this hook knowing who changed it.
  const [epoch, setEpoch] = useState(0)

  useEffect(() => {
    const bump = () => {
      setLoaded(null)
      setEpoch((n) => n + 1)
    }
    listeners.add(bump)
    return () => {
      listeners.delete(bump)
    }
  }, [])

  useEffect(() => {
    // Never before there is a token. This hook renders in the app chrome, which mounts before the
    // auth guard decides anything, so an unconditional fetch here 401s -- and `request()` treats a
    // 401 as "session over", clears localStorage and redirects to `/`. That threw away the
    // `?code=` of an in-progress Cognito callback, so signing in could never complete.
    if (!isAuthenticated()) return
    let live = true
    load(tenant).then((pack) => {
      if (live) setLoaded({ tenant, pack })
    })
    return () => {
      live = false
    }
  }, [tenant, epoch])

  return loaded?.tenant === tenant ? loaded.pack : seed(tenant)
}

export function useUnitLabel(): UnitLabel {
  return usePack()?.unit ?? derive(FALLBACK)
}

/**
 * Questions worth asking of this pack's data, for the Ask and Retrieval pages.
 *
 * Empty until settings arrive, and empty for a pack that declares none. Callers render nothing
 * rather than a placeholder: the affordance exists to show what the system *can* answer, so an
 * example that returns nothing is worse than no example at all.
 */
export function useExampleQuestions(): string[] {
  return usePack()?.questions ?? []
}
