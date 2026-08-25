/**
 * Normalising a query response into the four steps of the trace diagram.
 *
 * Kept out of the component module so fast refresh works, and separate from the page because
 * `/query` and `/query/compose` describe the same run in different shapes: one tier that
 * answered, or several lanes that ran. The diagram should not know which it is looking at.
 */

import type {
  AnswerPart,
  CatalogSchemaRef,
  ComposedResult,
  GeneratedSQLResult,
  Lane,
  QueryAnswer,
  QueryHit,
  QueryPassage,
  QueryResult,
  QueryRows,
  RouterTrace,
} from './api'
import { LANES } from './epistemic'

/**
 * `answer` is shaped by whichever tier answered, so it is narrowed rather than rendered.
 *
 * Rendering it directly is what blanked the Ask page: tier 2 puts a list of assertions there, and
 * a list of objects as a React child throws. Narrowing in one place beats guessing at each use.
 */
export function asRows(answer: QueryAnswer): QueryRows | null {
  if (answer && typeof answer === 'object' && 'columns' in answer && Array.isArray(answer.rows)) {
    return answer as QueryRows
  }
  return null
}

export function asHits(answer: QueryAnswer): QueryHit[] {
  if (Array.isArray(answer)) return answer
  if (answer && typeof answer === 'object' && 'related' in answer) return answer.related ?? []
  return []
}

export function asPassages(answer: QueryAnswer): QueryPassage[] {
  if (answer && typeof answer === 'object' && 'passages' in answer) return answer.passages ?? []
  return []
}

/**
 * The passages a composed answer quoted, gathered from whichever parts carry them.
 *
 * `asPassages` reads the single-tier `answer` shape, where everything sits under one key. A
 * composed answer nests each lane's output in `parts[].content`, so the same data is one level
 * deeper and reading it the old way found nothing.
 */
export function passagesFromComposed(composed: ComposedResult): QueryPassage[] {
  const out: QueryPassage[] = []
  for (const part of composed.parts ?? []) {
    if (part.lane !== 'passages' || !Array.isArray(part.content)) continue
    for (const row of part.content) {
      if (row && typeof row === 'object' && 'document_id' in row) out.push(row as QueryPassage)
    }
  }
  return out
}

/** The verified relationships a composed answer walked, from the graph part. */
export function factsFromComposed(composed: ComposedResult): QueryHit[] {
  const out: QueryHit[] = []
  for (const part of composed.parts ?? []) {
    if (part.lane !== 'graph' || !Array.isArray(part.content)) continue
    for (const row of part.content) {
      // `assertion_id` is what makes a row citable: without one there is no provenance to open,
      // so a row lacking it is not a fact this can offer to explain.
      if (row && typeof row === 'object' && 'assertion_id' in row) out.push(row as QueryHit)
    }
  }
  return out
}

/** Tier 3's AI-written query, or null when none was written. Absent on every older response. */
export function asGenerated(answer: QueryAnswer): GeneratedSQLResult | null {
  if (!answer || typeof answer !== 'object' || !('generated' in answer)) return null
  const generated = answer.generated
  if (!generated || typeof generated.sql !== 'string') return null
  return generated
}

/** `content` is free-form per lane, so every field is checked before it is read. */
function asSchema(content: unknown): CatalogSchemaRef[] {
  if (!Array.isArray(content)) return []
  return content
    .filter((row): row is Record<string, unknown> => !!row && typeof row === 'object')
    .filter((row) => typeof row.full_name === 'string')
    .map((row) => ({
      full_name: row.full_name as string,
      description: typeof row.description === 'string' ? row.description : null,
      columns: Array.isArray(row.columns) ? row.columns.filter((c) => typeof c === 'string') : [],
    }))
}

function asFacts(content: unknown): QueryHit[] {
  if (!Array.isArray(content)) return []
  return content.filter(
    (row): row is QueryHit =>
      !!row && typeof row === 'object' && typeof (row as QueryHit).assertion_id === 'string',
  )
}

function asPassageList(content: unknown): QueryPassage[] {
  if (!Array.isArray(content)) return []
  return content.filter(
    (row): row is QueryPassage =>
      !!row && typeof row === 'object' && typeof (row as QueryPassage).document_id === 'string',
  )
}

/** One lane of step 3: what ran, or what did not and why. */
export interface TraceLane {
  key: string
  label: string
  colour: string
  tier: number
  ran: boolean
  /**
   * Why there is nothing here. Verbatim from the response — never paraphrased.
   *
   * Set when the lane did not run, and also when it ran and failed: the SQL lane can produce a
   * query that errors at Athena, and a lane that failed must not render as one that found nothing.
   */
  reason?: string
  /** `deterministic` | `verbatim` | `inferred`, when the response says. */
  provenance?: string
  sql?: string | null
  rows?: QueryRows | null
  facts?: QueryHit[]
  passages?: QueryPassage[]
  schema?: CatalogSchemaRef[]
}

/** How many items a lane returned. Zero is a real answer and reads as one. */
export function laneCount(lane: TraceLane): number {
  return (
    (lane.rows?.rows.length ?? 0) +
    (lane.facts?.length ?? 0) +
    (lane.passages?.length ?? 0) +
    (lane.schema?.length ?? 0)
  )
}

/** Fallback only: a part carries its own tier. Catalogue and the SQL lane are tier 3, alongside
 *  the passages — the catalogue finds the schema and the SQL lane is what writes over it. */
const LANE_TIER: Record<Lane, number> = { metric: 1, graph: 2, passages: 3, catalog: 3, sql: 3 }

/**
 * The single tier that answered, as lanes.
 *
 * Tier 3 becomes two lanes rather than one: it retrieves passages and then walks the graph
 * around them, and collapsing that into "hybrid" loses which half supplied what.
 */
export function lanesFromResult(result: QueryResult): TraceLane[] {
  const rows = asRows(result.answer)
  const facts = asHits(result.answer)
  const passages = asPassages(result.answer)

  if (result.tier === 1) {
    return [{ ...laneShell('metric'), tier: 1, ran: true, rows, sql: result.sql ?? null }]
  }
  if (result.tier === 2) {
    return [{ ...laneShell('graph'), tier: 2, ran: true, facts }]
  }
  if (result.tier === 3) {
    const lanes: TraceLane[] = [{ ...laneShell('passages'), tier: 3, ran: true, passages }]
    if (facts.length > 0) lanes.push({ ...laneShell('graph'), tier: 3, ran: true, facts })
    const generated = asGenerated(result.answer)
    if (generated) {
      lanes.push({
        ...laneShell('sql'),
        tier: 3,
        ran: true,
        provenance: 'model_written',
        sql: generated.sql,
        rows: generated.rows ?? undefined,
        reason: generated.error ?? undefined,
      })
    }
    return lanes
  }
  // A tier this build does not know, which today means a retired one. No lane is claimed for it:
  // drawing a passage lane would assert that passages were searched, and nothing here knows that.
  return []
}

function laneShell(lane: Lane): TraceLane {
  return {
    key: lane,
    label: LANES[lane].label,
    colour: LANES[lane].colour,
    tier: LANE_TIER[lane],
    ran: true,
  }
}

/** The lanes a composed answer ran, and the ones it named as skipped. */
export function lanesFromComposed(composed: ComposedResult): TraceLane[] {
  const lanes: TraceLane[] = (composed.parts ?? []).map((part) => fromPart(part))
  for (const [lane, reason] of Object.entries(composed.lanes_skipped ?? {})) {
    // A lane can be skipped *and* have produced a part in an older response. The part wins:
    // whatever it says about being skipped, something came back.
    if (lanes.some((l) => l.key === lane)) continue
    const known = lane in LANES ? (lane as Lane) : null
    lanes.push({
      key: lane,
      label: known ? LANES[known].label : lane,
      colour: known ? LANES[known].colour : 'var(--text-dim)',
      tier: known ? LANE_TIER[known] : 0,
      ran: false,
      reason,
    })
  }
  return lanes
}

function fromPart(part: AnswerPart): TraceLane {
  const known = part.lane in LANES ? part.lane : null
  const base: TraceLane = {
    key: part.lane,
    label: known ? LANES[known].label : String(part.lane),
    colour: known ? LANES[known].colour : 'var(--text-dim)',
    tier: typeof part.tier === 'number' ? part.tier : (known ? LANE_TIER[known] : 0),
    ran: true,
    provenance: part.provenance,
    sql: part.sql ?? null,
  }
  if (part.lane === 'metric') return { ...base, rows: asRows(part.content as QueryAnswer) }
  if (part.lane === 'graph') return { ...base, facts: asFacts(part.content) }
  if (part.lane === 'passages') return { ...base, passages: asPassageList(part.content) }
  if (part.lane === 'catalog') return { ...base, schema: asSchema(part.content) }
  // `content` is null when the query errored, and the error is what there is to show. Falling
  // through to an empty rows table would read as "no data" for a query that never ran.
  if (part.lane === 'sql') {
    return { ...base, rows: asRows(part.content as QueryAnswer), reason: part.error ?? undefined }
  }
  return base
}

/**
 * Whether a dropped tier was refused by policy rather than out-scored.
 *
 * Read from `tiers_forbidden`, which the router sends as data. It was briefly inferred from the
 * wording of the reason, which worked and was the wrong shape: "your administrator turned this
 * off" and "this did not look relevant" are different facts about the system, and a UI that
 * decides which one it is by pattern-matching prose gets it wrong the first time anybody
 * rewords a sentence.
 *
 * Falls back to the reason text only when the field is absent, so a response from an older
 * backend still reads sensibly rather than reporting every refusal as a low score.
 */
export function isForbidden(router: RouterTrace | null | undefined, tier: number, reason = ''): boolean {
  const declared = router?.tiers_forbidden
  if (declared != null) return declared.map(Number).includes(tier)
  return /not permitted|not allowed|not enabled|forbidden|disabled for/i.test(reason)
}

/**
 * Whether the router's scores decided anything, or were only recorded.
 *
 * Two ways they decide nothing and they are not the same story: `degraded` is the router failing
 * to choose, `applied: false` is the caller choosing not to act on a decision it did get. Compose
 * is the second — it runs every permitted lane on purpose. Either way `selected` on a layer is a
 * score, not an outcome, and the diagram must not render it as one.
 */
export function routerDecided(router: RouterTrace | null | undefined): boolean {
  return router != null && !router.degraded && router.applied !== false
}

/** The score a layer had to reach to stay within the margin of the best one. */
export function marginCutoff(router: RouterTrace): number {
  const best = router.best_score ?? 0
  const margin = router.margin ?? 0
  return best > 0 ? best * (1 - margin) : 0
}

/** Tier numbers as numbers. `tiers_dropped` keys are JSON object keys, so they arrive as strings. */
export function droppedTiers(router: RouterTrace): { tier: number; reason: string }[] {
  return Object.entries(router.tiers_dropped ?? {})
    .map(([tier, reason]) => ({ tier: Number(tier), reason }))
    .filter((d) => Number.isFinite(d.tier))
    .sort((a, b) => a.tier - b.tier)
}
