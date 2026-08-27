/**
 * QueryTrace — the four steps that produced an answer, each one openable.
 *
 * The ethical wall is step 4 of a route the reader otherwise cannot see, and a wall nobody can
 * inspect is indistinguishable from no wall. So the whole route is drawn: what the question was
 * matched against, which tiers that chose, what each tier returned, and what the wall then did
 * with it.
 *
 * Vertical rather than left-to-right. A step expands to a table of passages or a page of SQL,
 * and four columns wide enough to hold that are too narrow to read; a single column keeps every
 * step the full width of the page and the rail keeps the order legible.
 *
 * Nothing here is asserted. `router` and `gate` are absent in deployments with no vector store,
 * and a step with no data says so rather than drawing an empty box.
 */

import { useState, type CSSProperties, type ReactNode } from 'react'
import type {
  GateTrace,
  QueryBlock,
  QueryHit,
  QueryPassage,
  RouterLayer,
  RouterTrace,
  ResolutionTier,
} from '../api'
import { HELP, ROUTER_LAYERS, PART_PROVENANCE_LABEL, TIERS } from '../epistemic'
import { fillUnit, useUnitLabel } from '../useUnitLabel'
import {
  droppedTiers,
  isForbidden,
  laneCount,
  marginCutoff,
  routerDecided,
  type TraceLane,
} from '../trace'
import ConfidenceBar from './ConfidenceBar'
import EpistemicBadge from './EpistemicBadge'
import FieldHelp from './FieldHelp'
import { TierBadge } from './Shared'
import { epiStyle } from '../format'

const TIER_NUMBERS: ResolutionTier[] = [1, 2, 3]

/** A layer's `tier` is nullable: a kind the response names but this build does not map has none. */
function isTier(n: number | null | undefined): n is number {
  return typeof n === 'number' && n > 0
}

export default function QueryTrace({
  router,
  gate,
  lanes,
  blocks,
  floor,
  usedFactCount = 0,
  onOpenPassage,
}: {
  router?: RouterTrace | null
  gate?: GateTrace | null
  lanes: TraceLane[]
  /** The blocks on the answer itself. Used when `gate` is absent, so a refusal is never lost. */
  blocks: QueryBlock[]
  floor: number
  /** Facts the answer did use, so a refusal can say where to look instead of nowhere. */
  usedFactCount?: number
  onOpenPassage?: (p: QueryPassage) => void
}) {
  const gateBlocks = gate?.blocks ?? blocks
  // Absent `effect` reads as withhold, matching the pack default and older responses.
  const withheldBlocks = gateBlocks.filter((b) => (b.effect ?? 'withhold') === 'withhold')
  const advisoryBlocks = gateBlocks.filter((b) => b.effect === 'notify')
  const ran = lanes.filter((l) => l.ran)
  const skipped = lanes.filter((l) => !l.ran)

  return (
    <div className="card">
      <div className="card-header">
        <h3>
          How this answer was reached
          <FieldHelp text={HELP.ethicalGate} />
        </h3>
        <span className="card-note">Open a step to see exactly what it returned.</span>
      </div>

      <div className="qtrace">
        <Step
          n={1}
          title="Disambiguate"
          summary={routerSummary(router)}
          tag={
            router == null ? null : router.degraded ? (
              <span className="tag tag-orange">degraded</span>
            ) : !router.enabled ? (
              <span className="tag tag-neutral">off</span>
            ) : router.applied === false ? (
              <span
                className="tag tag-blue"
                title="Scored, and recorded rather than acted on: every permitted lane ran."
              >
                scored
              </span>
            ) : (
              <span className="tag tag-green">chose</span>
            )
          }
        >
          <RouterStep router={router} />
        </Step>

        <Step n={2} title="Tiers chosen" summary={tiersSummary(router, ran)}>
          <TiersStep router={router} lanes={lanes} />
        </Step>

        <Step n={3} title={searchTitle(ran)} summary={lanesSummary(ran, skipped)}>
          <LanesStep lanes={lanes} floor={floor} onOpenPassage={onOpenPassage} />
        </Step>

        {/* Open from the start when something was refused, and when the wall failed open. A
            refusal that needs a click to find is the failure this step exists to prevent, and a
            check that did not run is the same failure wearing a green tag. */}
        <Step
          n={4}
          title="Ethical wall"
          summary={gateSummary(gate, gateBlocks)}
          tag={
            withheldBlocks.length > 0 ? (
              <span className="tag tag-red">
                {withheldBlocks.length} {withheldBlocks.length === 1 ? 'refusal' : 'refusals'}
              </span>
            ) : advisoryBlocks.length > 0 ? (
              // Orange, not red: nothing was suppressed. But not green either — a conflict the
              // reader is not told about is the failure this step exists to prevent.
              <span className="tag tag-orange">
                {advisoryBlocks.length} to review
              </span>
            ) : gate?.degraded ? (
              // Never green here. "Nothing refused" over a check that did not run is the exact
              // sentence this step exists to stop the system from saying.
              <span className="tag tag-orange" title={gate.degraded}>
                not fully checked
              </span>
            ) : (
              <span className="tag tag-green">nothing refused</span>
            )
          }
          defaultOpen={gateBlocks.length > 0 || Boolean(gate?.degraded)}
          // Advisories are inside `gateBlocks`, so they open the step too — a conflict that needs
          // a click to find is the same failure as a refusal that does.
          last
        >
          <GateStep gate={gate} blocks={gateBlocks} usedFactCount={usedFactCount} />
        </Step>
      </div>
    </div>
  )
}

// ── The rail ────────────────────────────────────────────────────────────────

function Step({
  n,
  title,
  summary,
  tag,
  last,
  defaultOpen = false,
  children,
}: {
  n: number
  title: string
  summary: string
  tag?: ReactNode
  last?: boolean
  defaultOpen?: boolean
  children: ReactNode
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className={`qtrace-step${last ? ' is-last' : ''}`}>
      <div className="qtrace-rail" aria-hidden="true">
        <div className="qtrace-node">{n}</div>
        {!last && <div className="qtrace-line" />}
      </div>
      <div className="qtrace-body">
        <button
          type="button"
          className="qtrace-head"
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
        >
          <span className="qtrace-title">{title}</span>
          {tag}
          <span className="qtrace-summary">{summary}</span>
          <span className="qtrace-chevron" aria-hidden="true">
            {open ? '−' : '+'}
          </span>
        </button>
        {open && <div className="qtrace-detail">{children}</div>}
      </div>
    </div>
  )
}

// ── Step 1: the router ──────────────────────────────────────────────────────

function routerSummary(router?: RouterTrace | null): string {
  if (router == null) return 'No routing trace was recorded for this question.'
  if (!router.enabled) return 'The router is switched off, so every permitted tier was tried.'
  if (router.degraded) {
    return router.reason
      ? `Could not choose, so every permitted tier ran. ${router.reason}`
      : 'Could not choose, so every permitted tier ran.'
  }
  const layers = router.layers ?? []
  const chosen = layers.filter((l) => l.selected).length
  const noun = layers.length === 1 ? 'layer' : 'layers'
  if (router.applied === false) {
    return `${layers.length} ${noun} scored, ${chosen} above the margin. Every permitted lane ran.`
  }
  return `${layers.length} ${noun} scored, ${chosen} searched.`
}

function RouterStep({ router }: { router?: RouterTrace | null }) {
  if (router == null) {
    return (
      <p className="qtrace-note">
        This deployment records no routing trace, either because the router is not wired or
        because there is no vector index to route against. The tiers below still ran and the wall
        below still applied, so the rest of this trace is complete. What is missing is the record
        of <em>why</em> those tiers and not others.
      </p>
    )
  }

  const layers = router.layers ?? []
  const cutoff = marginCutoff(router)
  const decided = routerDecided(router)

  return (
    <>
      {!router.degraded && router.applied === false && (
        <div className="banner banner-info" style={{ marginBottom: 12 }}>
          <span>
            <strong>Scored, but not acted on.</strong> This view searches everything the firm
            permits, so these scores explain where the question resembles the system rather than
            deciding where it looked. A layer below the margin was still searched, and its results
            are in step 3.
            <FieldHelp text={HELP.routerRecorded} />
          </span>
        </div>
      )}

      {router.degraded && (
        <div className="banner banner-warn" style={{ marginBottom: 12 }}>
          <span>
            <strong>The router did not choose.</strong> {sentence(router.reason)} Every tier this
            tenant permits was run instead. That is not a worse answer, but it is a different one
            to defend: nothing below was selected on the strength of its match.
            <FieldHelp text={HELP.routerDegraded} />
          </span>
        </div>
      )}

      <dl className="qtrace-facts">
        <div>
          <dt>
            Best score
            <FieldHelp text={HELP.similarity} />
          </dt>
          <dd>{fmtScore(router.best_score)}</dd>
        </div>
        <div>
          <dt>
            Margin
            <FieldHelp text={HELP.routerMargin} />
          </dt>
          <dd>
            {fmtScore(router.margin)} <span className="dim">· cutoff {fmtScore(cutoff)}</span>
          </dd>
        </div>
        <div>
          <dt>
            Similarity floor
            <FieldHelp text={HELP.routerMinSimilarity} />
          </dt>
          <dd>{fmtScore(router.min_similarity)}</dd>
        </div>
        <div>
          <dt>
            Metric boost
            <FieldHelp text={HELP.routerMetricBoost} />
          </dt>
          <dd>{router.metric_boost ? `+${fmtScore(router.metric_boost)}` : 'none'}</dd>
        </div>
      </dl>

      {layers.length === 0 ? (
        <p className="qtrace-note">
          No layer was scored. Nothing in this tenant's routing index resembled the question
          closely enough to report.
        </p>
      ) : (
        <div className="qtrace-layers">
          {layers.map((layer) => (
            // `selected` is only an outcome when the decision was acted on. Degraded, or recorded
            // on compose, every tier ran -- so labelling one layer "searched" and another "not
            // searched" would describe a decision that was never taken.
            <Layer key={layer.kind} layer={layer} cutoff={cutoff} decided={decided} />
          ))}
        </div>
      )}
      <p className="qtrace-note">
        Scores measure resemblance between your question and a description of each thing. They are
        not probabilities and not relevance percentages, which is why a layer is chosen by how it
        compares with the best layer rather than by clearing a fixed bar.
        <FieldHelp text={HELP.vectorRouter} />
      </p>
    </>
  )
}

function Layer({
  layer,
  cutoff,
  decided,
}: {
  layer: RouterLayer
  cutoff: number
  /** False when the scores decided nothing: the router degraded, or the caller only recorded it. */
  decided: boolean
}) {
  const [open, setOpen] = useState(false)
  const unit = useUnitLabel()
  const meta = layer.kind in ROUTER_LAYERS ? ROUTER_LAYERS[layer.kind] : null
  const items = layer.items ?? []

  return (
    <div
      className={`qtrace-layer${layer.selected && decided ? ' is-selected' : ''}`}
      style={{ '--layer-colour': meta?.colour ?? 'var(--text-dim)' } as CSSProperties}
    >
      <div className="qtrace-layer-head">
        <span className="qtrace-layer-name">{meta?.label ?? layer.kind}</span>
        {isTier(layer.tier) && <TierBadge tier={layer.tier} />}
        {!decided ? (
          <span className="tag tag-orange" title="This score decided nothing: every tier ran.">
            scored only
          </span>
        ) : (
          <span className={`tag ${layer.selected ? 'tag-green' : 'tag-neutral'}`}>
            {layer.selected ? 'searched' : 'not searched'}
          </span>
        )}
        <span className="qtrace-score">
          <span className="qtrace-score-track">
            <span
              className="qtrace-score-fill"
              style={{ width: `${Math.max(0, Math.min(1, layer.score)) * 100}%` }}
            />
            {cutoff > 0 && cutoff <= 1 && (
              <span
                className="qtrace-score-cutoff"
                style={{ left: `${cutoff * 100}%` }}
                title={`Margin cutoff ${fmtScore(cutoff)}. A layer below this was not searched.`}
              />
            )}
          </span>
          <span className="qtrace-score-value">{fmtScore(layer.score)}</span>
        </span>
      </div>
      <p className="qtrace-layer-reason">{layer.reason}</p>
      {layer.boost > 0 && (
        <p className="qtrace-layer-reason dim">
          {fmtScore(layer.raw_score)} on resemblance alone, plus a {fmtScore(layer.boost)} boost
          for being a governed metric.
        </p>
      )}
      {items.length > 0 ? (
        <>
          <button type="button" className="link-button qtrace-more" onClick={() => setOpen((o) => !o)}>
            {open
              ? 'Hide what matched'
              : `Show ${items.length} of ${layer.hit_count} ${layer.hit_count === 1 ? 'match' : 'matches'}`}
          </button>
          {open && (
            <table className="data-table qtrace-items">
              <thead>
                <tr>
                  <th>Matched</th>
                  <th>Detail</th>
                  <th className="num">
                    Similarity
                    <FieldHelp text={HELP.similarity} />
                  </th>
                </tr>
              </thead>
              <tbody>
                {items.map((item) => (
                  <tr key={item.item_id}>
                    <td>
                      <strong>{item.label || item.item_id}</strong>
                      <div className="dim mono" style={{ fontSize: 11 }}>
                        {item.item_id}
                      </div>
                    </td>
                    <td className="dim">{fmtDetail(item.detail)}</td>
                    <td className="num">{fmtScore(item.similarity)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </>
      ) : (
        <p className="qtrace-layer-reason dim">
          {layer.hit_count > 0
            ? `${layer.hit_count} matched, but the trace carries no detail of them.`
            : `Nothing here matched. ${meta ? fillUnit(meta.meaning, unit) : ''}`}
        </p>
      )}
    </div>
  )
}

// ── Step 2: which tiers that chose ──────────────────────────────────────────

function tiersSummary(router: RouterTrace | null | undefined, ran: TraceLane[]): string {
  // Only when the decision was acted on. `tiers_selected` on compose is what scored well, not what
  // ran, and reporting it as "ran" is the bug this whole distinction exists to prevent.
  const selected = routerDecided(router) ? (router?.tiers_selected ?? []) : []
  if (selected.length > 0) {
    const dropped = droppedTiers(router as RouterTrace)
    const forbidden = dropped.filter((d) => isForbidden(router, d.tier, d.reason)).length
    return (
      `Tier ${selected.join(', ')} ran` +
      (dropped.length > 0
        ? `. ${dropped.length} dropped` + (forbidden > 0 ? `, ${forbidden} not permitted here.` : '.')
        : '.')
    )
  }
  const tiers = [...new Set(ran.map((l) => l.tier))].filter((t) => t > 0).sort()
  const forbidden = (router?.tiers_forbidden ?? []).length
  if (tiers.length === 0) {
    return forbidden > 0
      ? `No tier ran. ${forbidden} ${forbidden === 1 ? 'is' : 'are'} not permitted here.`
      : 'No tier is recorded as having run.'
  }
  const tail = forbidden > 0
    ? ` ${forbidden} ${forbidden === 1 ? 'tier is' : 'tiers are'} not permitted here.`
    : router == null
      ? ' Nothing records what was dropped.'
      : ''
  return `Tier ${tiers.join(', ')} ran.${tail}`
}

function TiersStep({ router, lanes }: { router?: RouterTrace | null; lanes: TraceLane[] }) {
  // What ran, not what scored well. On compose the router's `tiers_selected` is advice and every
  // permitted lane ran anyway, so trusting it here would grey out a tier whose results are visible
  // in the step below.
  const selected = new Set(
    routerDecided(router)
      ? (router?.tiers_selected ?? lanes.filter((l) => l.ran).map((l) => l.tier))
      : lanes.filter((l) => l.ran).map((l) => l.tier),
  )
  // Selected is the router's decision; answered is the outcome. A tier can be searched and come
  // back empty, and calling that "searched" without more reads as though it contributed.
  const answered = new Set(lanes.filter((l) => l.ran && laneCount(l) > 0).map((l) => l.tier))
  // A low score is only a reason a tier did not run when the decision was acted on. Otherwise the
  // tier ran regardless and the only real refusals left are the tenant's.
  const dropped = (router ? droppedTiers(router) : []).filter(
    (d) => routerDecided(router) || isForbidden(router, d.tier, d.reason),
  )
  const reasonFor = new Map(dropped.map((d) => [d.tier, d.reason]))
  // A lane the planner skipped is a fifth story: the tier was permitted and chosen, and its
  // collaborator was missing. Kept beside the tier reasons rather than merged into them.
  const laneSkips = lanes.filter((l) => !l.ran && !reasonFor.has(l.tier))

  return (
    <>
      <div className="tier-ladder">
        {TIER_NUMBERS.map((t) => {
          const reason = reasonFor.get(t)
          const forbidden = isForbidden(router, t, reason ?? '')
          const on = selected.has(t)
          return (
            <div
              key={t}
              className={`tier-ladder-row${on ? ' active' : ''}${forbidden ? ' forbidden' : ''}`}
              style={{ '--tier-colour': TIERS[t].colour } as CSSProperties}
            >
              <span className="tier-ladder-num">{t}</span>
              <span>
                <strong>{TIERS[t].label}.</strong> {TIERS[t].detail}
                <span className="qtrace-tier-verdict">
                  {on ? (
                    answered.has(t) ? (
                      <span className="tag tag-green">searched</span>
                    ) : (
                      <span className="tag tag-neutral" title="Searched, and it returned nothing.">
                        searched, empty
                      </span>
                    )
                  ) : forbidden ? (
                    <>
                      <span className="tag tag-red">not permitted</span>
                      <FieldHelp text={HELP.tierForbidden} />
                    </>
                  ) : reason != null ? (
                    <span className="tag tag-neutral">not selected</span>
                  ) : (
                    <span className="tag tag-neutral">not reached</span>
                  )}
                  {reason != null && <span className="qtrace-tier-reason">{reason}</span>}
                </span>
              </span>
            </div>
          )
        })}
      </div>

      {dropped.some((d) => isForbidden(router, d.tier, d.reason)) && (
        <p className="qtrace-note">
          A tier marked <strong>not permitted</strong> was never tried: an administrator has
          switched it off for this firm, and nothing about the question was measured against it.
          That is a different statement from <strong>not selected</strong>, which means it was
          measured and came back too far behind the best layer to be worth searching.
        </p>
      )}

      {laneSkips.length > 0 && (
        <>
          <p className="qtrace-note">
            These were permitted, and still did not run, because the part of the system they need
            is not available in this deployment:
          </p>
          <ul className="qtrace-list">
            {laneSkips.map((l) => (
              <li key={l.key}>
                <strong>{l.label}.</strong> {l.reason}
              </li>
            ))}
          </ul>
        </>
      )}
    </>
  )
}

// ── Step 3: what each lane returned ─────────────────────────────────────────

/** Lanes run in sequence, never concurrently. Said plainly because the reader is being asked to
 *  audit an answer, and "in parallel" would have claimed the lanes were independent when the
 *  graph lane's input is the passage lane's output. */
function searchTitle(ran: TraceLane[]): string {
  return ran.length > 1 ? 'Searched in sequence' : 'Searched'
}

/** True when the graph lane walked out from retrieved passages rather than only term-matching. */
function chainedRetrieval(ran: TraceLane[]): boolean {
  return ran.some((l) => l.key === 'passages') && ran.some((l) => l.key === 'graph')
}

function lanesSummary(ran: TraceLane[], skipped: TraceLane[]): string {
  if (ran.length === 0) {
    return skipped.length > 0
      ? `Nothing ran. ${skipped.length} ${skipped.length === 1 ? 'lane was' : 'lanes were'} skipped.`
      : 'Nothing ran.'
  }
  const parts = ran.map((l) => `${l.label} ${laneCount(l)}`)
  return parts.join(' · ')
}

function LanesStep({
  lanes,
  floor,
  onOpenPassage,
}: {
  lanes: TraceLane[]
  floor: number
  onOpenPassage?: (p: QueryPassage) => void
}) {
  const ran = lanes.filter((l) => l.ran)
  if (ran.length === 0) {
    return (
      <p className="qtrace-note">
        No lane returned anything. That is not the same as nothing existing: step 2 says which
        tiers were permitted to look.
      </p>
    )
  }
  return (
    <div className="qtrace-lanes">
      {/* The cards read graph first because a verified relationship is the stronger claim, which
          is the opposite of the order they ran in. Say so, or the reader infers the graph found
          these independently. */}
      {chainedRetrieval(ran) && (
        <p className="qtrace-note">
          Documents were searched first. The graph lane then walked out from the passages below, so
          a fact marked with hops was reached because a passage cited it, not found on its own.
        </p>
      )}
      {ran.map((lane) => (
        <LaneCard key={lane.key} lane={lane} floor={floor} onOpenPassage={onOpenPassage} />
      ))}
    </div>
  )
}

/** The facts a lane returned, split by how each was reached.
 *
 * These arrive in one list because `Planner._graph_part` concatenates the term search with the
 * walk out from retrieved passages, but they are not the same kind of claim: a term match is a
 * fact the question named, while a walked edge is one the graph reached from a cited passage. The
 * API has always distinguished them -- `expand()` sets `hops` for exactly this reason and
 * `matched_on` stays empty on a walked edge -- and the panel used to discard it, so a reader could
 * not tell a fact stated in the document from one two steps away.
 */
function FactList({ facts, floor }: { facts: QueryHit[]; floor: number }) {
  const matched = facts.filter((h) => h.hops == null)
  const walked = facts.filter((h) => h.hops != null)

  const row = (h: QueryHit) => (
    <div key={h.assertion_id} className="path-hop" style={epiStyle(h.epistemic_class)}>
      <EpistemicBadge epistemicClass={h.epistemic_class} size="sm" showLabel={false} />
      <span>
        <strong>{entityLabel(h.subject_id)}</strong>{' '}
        <span className="prov-pred">{h.predicate}</span>{' '}
        <strong>{entityLabel(h.object_id)}</strong>
        {h.hops != null && (
          <span className="dim" style={{ marginLeft: 6, fontSize: 11 }}>
            {h.hops} {h.hops === 1 ? 'hop' : 'hops'} from a cited passage
          </span>
        )}
      </span>
      <ConfidenceBar value={h.confidence} floor={floor} width={54} />
    </div>
  )

  // One list when everything arrived the same way, which is the common case. Splitting then would
  // add a heading that says nothing.
  if (!walked.length || !matched.length) {
    return <div className="path-chain">{facts.map(row)}</div>
  }
  return (
    <>
      <p className="qtrace-note dim">
        Matched by name in the question, the graph was asked about these directly.
      </p>
      <div className="path-chain">{matched.map(row)}</div>
      <p className="qtrace-note dim" style={{ marginTop: 10 }}>
        Reached by walking out from a retrieved passage. Nothing in the question named these; the
        hybrid tier found them next to a document it cited.
      </p>
      <div className="path-chain">{walked.map(row)}</div>
    </>
  )
}

function LaneCard({
  lane,
  floor,
  onOpenPassage,
}: {
  lane: TraceLane
  floor: number
  onOpenPassage?: (p: QueryPassage) => void
}) {
  const [open, setOpen] = useState(false)
  const count = laneCount(lane)
  const provenance = lane.provenance
    ? (PART_PROVENANCE_LABEL[lane.provenance] ?? lane.provenance)
    : null

  return (
    <div className="qtrace-lane" style={{ '--layer-colour': lane.colour } as CSSProperties}>
      <button
        type="button"
        className="qtrace-lane-head"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
      >
        <span className="qtrace-layer-name">{lane.label}</span>
        {isTier(lane.tier) && <TierBadge tier={lane.tier} />}
        {provenance && <span className="tag tag-neutral">{provenance}</span>}
        <span className="qtrace-lane-count">
          {count} {count === 1 ? 'item' : 'items'}
        </span>
        <span className="qtrace-chevron" aria-hidden="true">
          {open ? '−' : '+'}
        </span>
      </button>

      {open && (
        <div className="qtrace-lane-body">
          {/* A lane that ran and failed is not a lane that found nothing. The SQL lane can do
              either: the firewall validates tables but not columns, so a query naming a column
              that does not exist errors at Athena, and calling that an empty result would read as
              "no data" for a question nothing ever answered. */}
          {count === 0 && lane.reason && (
            <p className="qtrace-note qtrace-note-warn">
              This lane ran and did not return a result: {lane.reason}
            </p>
          )}
          {count === 0 && !lane.reason && (
            <p className="qtrace-note">
              This lane ran and returned nothing. Read that as an empty result from a real search,
              not as a search that did not happen.
            </p>
          )}

          {lane.rows && lane.rows.rows.length > 0 && (
            <div style={{ overflowX: 'auto' }}>
              <table className="data-table">
                <thead>
                  <tr>
                    {lane.rows.columns.map((c) => (
                      <th key={c}>{c}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {lane.rows.rows.map((r, i) => (
                    <tr key={i}>
                      {r.map((v, j) => (
                        <td key={j} className={typeof v === 'number' ? 'num' : ''}>
                          {typeof v === 'number' ? v.toLocaleString() : (v ?? '-')}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {lane.facts && lane.facts.length > 0 && <FactList facts={lane.facts} floor={floor} />}

          {lane.passages && lane.passages.length > 0 && (
            <>
              {lane.passages.map((p, i) => (
                <div className="citation" key={`${p.document_id}-${p.char_start ?? i}`}>
                  <span className="citation-num">[{i + 1}]</span>
                  <div className="citation-body">
                    {p.text && <div className="citation-quote">{p.text}</div>}
                    <div className="citation-loc">
                      {p.filename ?? p.document_id}
                      {p.page != null ? ` · page ${p.page}` : ''}
                      {p.score != null ? ` · ${fmtScore(p.score)}` : ''}
                    </div>
                  </div>
                  {p.page != null && onOpenPassage && (
                    <button className="btn btn-ghost btn-sm" onClick={() => onOpenPassage(p)}>
                      Open at page {p.page}
                    </button>
                  )}
                </div>
              ))}
            </>
          )}

          {lane.schema && lane.schema.length > 0 && (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Table</th>
                  <th>Columns</th>
                </tr>
              </thead>
              <tbody>
                {lane.schema.map((t) => (
                  <tr key={t.full_name}>
                    <td>
                      <code>{t.full_name}</code>
                      {t.description && (
                        <div className="dim" style={{ fontSize: 11.5 }}>
                          {t.description}
                        </div>
                      )}
                    </td>
                    <td className="dim">{(t.columns ?? []).join(', ') || '-'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          {lane.sql && (
            <>
              <div className="qtrace-sublabel">
                {lane.key === 'sql' ? 'AI-written SQL' : 'Compiled SQL'}
                <FieldHelp
                  text={
                    lane.key === 'sql'
                      ? 'Written by AI for this question, not compiled from an approved metric. It was checked against the tables it was allowed to read and required to aggregate, but nobody approved what it measures. Read it before relying on the figures.'
                      : 'Compiled from the governed metric definition. The same definition always produces this query, with no model involved.'
                  }
                />
              </div>
              <pre className="code-block">{lane.sql}</pre>
            </>
          )}
        </div>
      )}
    </div>
  )
}

// ── Step 4: the wall ────────────────────────────────────────────────────────

function gateSummary(gate: GateTrace | null | undefined, blocks: QueryBlock[]): string {
  if (gate == null) {
    return blocks.length > 0
      ? `${blocks.length} ${blocks.length === 1 ? 'block' : 'blocks'} applied. No count of what cleared was recorded.`
      : 'Nothing was refused. No count of what cleared was recorded.'
  }
  if (gate.degraded) {
    return 'Screens applied, but the graph was not checked for conflicts.'
  }
  // `subjects_flagged` is why this is not just cleared-and-withheld. A notify finding withholds
  // nothing, so "4 of 5 cleared, 0 items withheld" beside a conflict would be true and read as
  // reassurance. Naming the flagged count is what stops the summary being misleading.
  const flagged = gate.subjects_flagged ?? 0
  const parts = [`${gate.subjects_cleared} of ${gate.seeds_considered} cleared`]
  if (flagged > 0) parts.push(`${flagged} flagged for review`)
  parts.push(`${gate.items_withheld} ${gate.items_withheld === 1 ? 'item' : 'items'} withheld`)
  return `${parts.join(', ')}.`
}

function GateStep({
  gate,
  blocks,
  usedFactCount,
}: {
  gate?: GateTrace | null
  blocks: QueryBlock[]
  usedFactCount: number
}) {
  const unit = useUnitLabel()
  const screens = blocks.filter((b) => b.rule === 'ethical_screen')
  const withheld = blocks.filter((b) => (b.effect ?? 'withhold') === 'withhold')
  const degraded = Boolean(gate?.degraded)

  return (
    <>
      <p className="qtrace-note">
        Applied by the graph, not a model, and before anything was summarised. A model never saw
        what was refused here, so it could not have reasoned about it even accidentally.
        <FieldHelp text={HELP.ethicalScreen} />
      </p>

      {gate?.degraded && (
        <p className="qtrace-note qtrace-withheld">
          The ethical screens on your account were applied. The graph was not checked for
          conflicts or other rule-based blocks, so nothing listed below is a clearance, this
          answer may include evidence that would normally be withheld.
          <br />
          <span className="dim">Reported reason: {gate.degraded}</span>
        </p>
      )}

      {gate != null && !degraded && (
        <dl className="qtrace-facts">
          <div>
            <dt>Subjects considered</dt>
            <dd>{gate.seeds_considered}</dd>
          </div>
          <div>
            <dt>Cleared</dt>
            <dd className="qtrace-cleared">{gate.subjects_cleared}</dd>
          </div>
          <div>
            <dt>Withheld</dt>
            <dd className={gate.items_withheld > 0 ? 'qtrace-withheld' : undefined}>
              {gate.items_withheld}
            </dd>
          </div>
        </dl>
      )}

      {gate == null && (
        <p className="qtrace-note dim">
          This response carries no count of what the wall cleared, only what it refused. An answer
          with nothing listed below passed the wall; it does not say how much passed it.
        </p>
      )}

      {/* Before the "nothing was refused" note, because that note is exactly what this
          qualifies. An unreviewed conflict cannot refuse anything, a model's proposal must not
          withhold evidence, but a clean wall over a graph that holds one is a true sentence
          reading as a false one. */}
      {!!gate?.awaiting_review?.length && !degraded && (
        <p className="qtrace-note qtrace-awaiting">
          <span className="tag tag-orange">awaiting review</span> A conflict or other blocking fact
          about {gate.awaiting_review.join(', ')} is in the review queue and has not been signed
          off, so it refused nothing here. Nothing below is a clearance for{' '}
          {gate.awaiting_review.length === 1 ? 'that subject' : 'those subjects'} until someone
          reviews it.
        </p>
      )}

      {blocks.length === 0 ? (
        !degraded && (
          <p className="qtrace-note">
            Nothing was refused for this question. That is a result the wall produced, not an
            absence of one: had a screened {unit.lower} matched, it would be named here rather
            than quietly left out.
          </p>
        )
      ) : (
        <div className="withheld-block" style={{ marginBottom: 0 }}>
          <div className="withheld-block-head">
            <h3>
              {blocks.length} {blocks.length === 1 ? 'finding' : 'findings'}
              {/* Not `ethicalScreen`: this heading covers rule findings too, and a
                  stale-authority one is a quality problem rather than a conduct one. */}
              <FieldHelp text={HELP.blockKinds} />
            </h3>
            {withheld.length > 0 ? (
              <span className="tag tag-red">
                {screens.length === blocks.length ? 'Screened' : 'Withheld'}
              </span>
            ) : (
              <span className="tag tag-orange">To review</span>
            )}
          </div>
          <p className="withheld-block-note">
            {withheld.length > 0
              ? 'Named on purpose. An answer that looks complete because the inconvenient part was invisible is the failure a screen exists to prevent.'
              : 'Nothing was withheld. The evidence below is complete, and whether these findings matter is a judgement for you, which is why they are named rather than acted on.'}
            {blocks.some((b) => (b.premise_count ?? 0) > 2) && (
              <>
                {' '}
                Most indirect first: a conflict the graph had to derive from several documents is
                the one least likely to be already known.
              </>
            )}
          </p>
          <div className="withheld-list">
            {blocks.map((b, i) => (
              <div className="withheld-item" key={`${b.rule}-${b.subject}-${i}`}>
                <div className="withheld-item-head">
                  <strong>{b.matter_id ?? entityLabel(b.subject)}</strong>
                  <code>{b.rule || 'blocked'}</code>
                  {/* Per item, because one list can hold both: a screen that walled a matter off
                      and a conflict that only wants a lawyer's eye are not the same news. */}
                  {(b.effect ?? 'withhold') === 'withhold' ? (
                    <span className="tag tag-red" title="Evidence about this was withheld.">
                      withheld
                    </span>
                  ) : (
                    <span
                      className="tag tag-orange"
                      title="Nothing was withheld. Reported for you to weigh."
                    >
                      to review
                    </span>
                  )}
                  {(b.premise_count ?? 0) > 2 && (
                    <span
                      className="tag tag-red"
                      title={
                        `Derived from ${b.premise_count} separate signed-off facts. A conflict ` +
                        'this indirect is not visible in any one document, so it is unlikely ' +
                        'anyone would have found it by reading the file.'
                      }
                    >
                      indirect ({b.premise_count} facts)
                    </span>
                  )}
                </div>
                <div className="withheld-field">
                  <span className="withheld-field-label">Reason recorded</span>
                  {b.reason}
                </div>
                {b.rule === 'ethical_screen' && (
                  <div className="withheld-field">
                    <span className="withheld-field-label">Who to contact</span>
                    {b.contact ?? (
                      <span className="dim">
                        No contact was given. Ask your risk team about this {unit.lower}.
                      </span>
                    )}
                  </div>
                )}
                <div className="withheld-field">
                  <span className="withheld-field-label">In the graph</span>
                  {/* Stated rather than linked, on purpose. `?highlight=` takes assertion ids
                      and a block has none, a screen is a grant, not a fact. A link filtered to
                      a screened matter would draw an empty canvas reading as "this matter holds
                      nothing", which is the silent failure this card exists to prevent. */}
                  <span className="dim">
                    Nothing to open.{' '}
                    {b.rule === 'ethical_screen'
                      ? 'A screen is a recorded instruction, not an assertion, so there is no ' +
                        'subgraph to draw.'
                      : 'A rule finding names what it is about, not an edge you can open.'}
                    {usedFactCount > 0 && ' What the answer did use opens in the graph below.'}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </>
  )
}

// ── Formatting ──────────────────────────────────────────────────────────────

/** Entity ids are `kind:slug`. The slug is what a reader recognises; the kind is noise here. */
function entityLabel(id: string): string {
  const slug = id.includes(':') ? id.slice(id.indexOf(':') + 1) : id
  return slug.replace(/[-_]/g, ' ')
}

function fmtScore(n: number | null | undefined): string {
  return typeof n === 'number' && Number.isFinite(n) ? n.toFixed(2) : '-'
}

/** A server reason is a fragment, and it is concatenated with prose. Capitalised and stopped so
 *  it does not run into the next sentence. Never reworded. */
function sentence(s: string | null | undefined): string {
  const text = (s ?? '').trim()
  if (!text) return 'No reason was recorded.'
  const lead = text.charAt(0).toUpperCase() + text.slice(1)
  return /[.!?]$/.test(lead) ? lead : `${lead}.`
}

/** `detail` is free-form per kind, so it is flattened rather than read field by field. */
function fmtDetail(detail: Record<string, unknown> | null | undefined): string {
  if (detail == null || typeof detail !== 'object') return '-'
  const parts = Object.entries(detail)
    .filter(([, v]) => v != null && v !== '')
    .map(([k, v]) => `${k.replace(/_/g, ' ')}: ${Array.isArray(v) ? v.join(', ') : String(v)}`)
  return parts.length > 0 ? parts.join(' · ') : '-'
}
