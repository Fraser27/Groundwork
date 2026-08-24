/**
 * Entities — is anything in this graph forked?
 *
 * Entity fragmentation is the ceiling on conflict detection. If one company exists as three nodes,
 * a check walks to one of them and comes back clean, and a missed merge is indistinguishable from
 * no conflict. This page is where a fork gets found and, by a human act, closed.
 *
 * Nothing here merges automatically at any confidence. The blocking key that catches a spelling
 * variant also catches a genuine sibling company, and merging those two would convert an affiliate
 * conflict into a false direct one. So a group is presented as a question.
 */

import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'

import { api, type Assertion, type DuplicateGroup, type MergeResult } from '../api'
import { getTenantId } from '../auth'
import FieldHelp from '../components/FieldHelp'
import MergeDialog from '../components/MergeDialog'
import { EmptyState, ErrorState, Spinner, Toast } from '../components/Shared'
import { entityKind, entityLabel } from '../format'

interface MergeTarget {
  candidates: string[]
  winner: string
  loser: string
}

export default function Entities() {
  const tenant = getTenantId()
  const [groups, setGroups] = useState<DuplicateGroup[]>([])
  const [assertions, setAssertions] = useState<Assertion[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [duplicatesError, setDuplicatesError] = useState('')
  const [reloadKey, setReloadKey] = useState(0)
  const [filter, setFilter] = useState('')
  const [merging, setMerging] = useState<MergeTarget | null>(null)
  const [toast, setToast] = useState<{ msg: string; type: string } | null>(null)

  const showToast = (msg: string, type = 'success') => {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 7000)
  }

  useEffect(() => {
    // Two requests, settled independently. `/entities/duplicates` answers 503 when no ontology
    // pack is wired, and that must not blank the entity list -- a manual merge is still possible
    // when nothing can suggest a candidate.
    Promise.allSettled([
      api.listAssertions(tenant, { review_state: 'ALL', limit: 500 }),
      api.entityDuplicates(tenant),
    ])
      .then(([a, d]) => {
        if (a.status === 'fulfilled') {
          setAssertions(a.value)
          setError('')
        } else setError((a.reason as Error).message)

        if (d.status === 'fulfilled') {
          setGroups(d.value.groups)
          setDuplicatesError('')
        } else setDuplicatesError((d.reason as Error).message.replace(/^\d+:\s*/, ''))
      })
      .finally(() => setLoading(false))
  }, [tenant, reloadKey])

  const retry = () => {
    setLoading(true)
    setReloadKey((k) => k + 1)
  }

  /** Every id currently named by a live claim, with how many claims name it. */
  const entities = useMemo(() => {
    const counts = new Map<string, number>()
    for (const a of assertions) {
      for (const id of [a.subject_id, a.object_id]) {
        if (id) counts.set(id, (counts.get(id) ?? 0) + 1)
      }
    }
    return counts
  }, [assertions])

  const factsFor = (id: string) => entities.get(id)

  const allIds = useMemo(() => [...entities.keys()].sort(), [entities])

  const visibleGroups = useMemo(() => {
    if (!filter.trim()) return groups
    const q = filter.toLowerCase()
    return groups.filter(
      (g) => g.key.includes(q) || g.entity_ids.some((id) => id.toLowerCase().includes(q)),
    )
  }, [groups, filter])

  const onMerged = (r: MergeResult) => {
    setMerging(null)
    showToast(
      `${r.rewritten.length} claim${r.rewritten.length === 1 ? '' : 's'} restated about ` +
        `${r.winning_id}${
          r.cascaded.length > 0
            ? `, and ${r.cascaded.length} conclusion${r.cascaded.length === 1 ? '' : 's'} withdrawn`
            : ''
        }. The Audit page records who merged them and why.`,
    )
    retry()
  }

  if (loading) return <Spinner />

  return (
    <>
      <div className="page-header">
        <div className="page-header-row">
          <div>
            <h2>Entities</h2>
            <p>
              One company held as two nodes is the quietest failure in the system: a conflict check
              walks to one of them and comes back clean. Ids that may name one thing are grouped
              here so somebody can decide.
            </p>
          </div>
          <button
            className="btn btn-ghost btn-sm"
            onClick={() =>
              setMerging({
                candidates: allIds,
                winner: allIds[0] ?? '',
                loser: allIds[1] ?? allIds[0] ?? '',
              })
            }
            disabled={allIds.length < 2}
            title={
              allIds.length < 2
                ? 'A merge needs two ids, and fewer than two are named by any claim'
                : 'Pick both sides by hand'
            }
          >
            Merge two ids
          </button>
        </div>
      </div>

      {error && <ErrorState title="Could not load the entities in this graph" detail={error} onRetry={retry} />}

      <div className="stats-grid">
        <div className="stat-card">
          <div className="label">Entity ids in use</div>
          <div className="value accent">{entities.size}</div>
          <div className="sub">Named by at least one claim</div>
        </div>
        <div className="stat-card">
          <div className="label">
            Possible forks
            <FieldHelp text="Groups of ids sharing a name shape once legal-form words like Ltd or GmbH are set aside. A group is a question, not a finding: Calder Shipping AG and Calder Shipping Ltd are routinely a parent and its subsidiary, which is a relationship rather than a duplicate." />
          </div>
          <div className={`value ${groups.length > 0 ? 'orange' : 'green'}`}>{groups.length}</div>
          <div className="sub">Each needs a human decision</div>
        </div>
      </div>

      {duplicatesError && (
        <div className="banner banner-warn">
          <span>
            <strong>No fork candidates could be computed.</strong> {duplicatesError}. Ids are still
            listed below and a merge can still be done by hand — this only means nothing is
            suggesting one.
          </span>
        </div>
      )}

      <div className="banner banner-info">
        <span>
          A merge is never automatic. The same name shape that catches a variant spelling also
          catches a genuine sibling company, and merging those would turn an affiliate conflict into
          a false direct one. Nothing on this page acts until somebody chooses which id survives.
        </span>
      </div>

      {groups.length > 0 && (
        <div className="search-bar">
          <input
            placeholder="Filter groups by id or key…"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          />
          {filter && (
            <span className="search-count">
              {visibleGroups.length} of {groups.length}
            </span>
          )}
        </div>
      )}

      <div className="card">
        <div className="card-header">
          <h3>
            Ids that may name one thing
            <FieldHelp text="Grouped by blocking key. Words that are legal form rather than name — ag, gmbh, ltd, llp, plc and the rest — are set aside when computing the key. Holdings, group, partners and trading deliberately are not: Acme Corp and Acme Holdings are two companies." />
          </h3>
          <Link to="/provenance" className="btn btn-ghost btn-sm">
            Audit view
          </Link>
        </div>

        {visibleGroups.length === 0 ? (
          <EmptyState title={groups.length === 0 ? 'No forks found' : 'No groups match'}>
            {groups.length === 0
              ? 'No two ids in this graph share a name shape. That is a clean result rather than an unanswered question: the check ran over every id named by a claim.'
              : 'Clear the filter to see every group.'}
          </EmptyState>
        ) : (
          <div className="merge-groups">
            {visibleGroups.map((g) => (
              <GroupRow
                key={g.key}
                group={g}
                factsFor={factsFor}
                onMerge={(winner, loser) =>
                  setMerging({ candidates: g.entity_ids, winner, loser })
                }
              />
            ))}
          </div>
        )}
      </div>

      {merging && (
        <MergeDialog
          tenant={tenant}
          candidates={merging.candidates}
          initialWinner={merging.winner}
          initialLoser={merging.loser}
          factsFor={factsFor}
          onCancel={() => setMerging(null)}
          onMerged={onMerged}
        />
      )}

      <Toast toast={toast} />
    </>
  )
}

/**
 * One blocking-key group.
 *
 * The claim counts are the only evidence on offer for which id should survive, so they are on the
 * row rather than a click away. They are a hint and not an answer: the id with more claims is
 * usually the established one, but a fresh import can outnumber it.
 */
function GroupRow({
  group,
  factsFor,
  onMerge,
}: {
  group: DuplicateGroup
  factsFor: (id: string) => number | undefined
  onMerge: (winner: string, loser: string) => void
}) {
  const ranked = [...group.entity_ids].sort((a, b) => (factsFor(b) ?? 0) - (factsFor(a) ?? 0))
  const kinds = [...new Set(group.entity_ids.map(entityKind))]

  return (
    <div className="merge-group">
      <div className="merge-group-head">
        <div>
          <strong>{entityLabel(ranked[0] ?? group.key)}</strong>
          <span className="dim"> · {group.entity_ids.length} ids share this name shape</span>
        </div>
        <div className="merge-group-badges">
          {kinds.map((k) => (
            <span key={k} className="tag tag-neutral tag-mono">
              {k || 'no kind'}
            </span>
          ))}
          {kinds.length > 1 && (
            <span className="tag tag-red" title="A merge across kinds is refused by the server">
              Different kinds
            </span>
          )}
        </div>
      </div>

      <table className="data-table">
        <thead>
          <tr>
            <th>Entity id</th>
            <th className="num">Claims</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {ranked.map((id) => (
            <tr key={id}>
              <td>
                <code>{id}</code>
              </td>
              <td className="num">{factsFor(id) ?? 0}</td>
              <td className="num">
                {id === ranked[0] ? (
                  <span className="dim" title="Most claims name this id, so it is offered as the survivor">
                    most claims
                  </span>
                ) : (
                  <button
                    className="btn btn-ghost btn-sm"
                    onClick={() => onMerge(ranked[0], id)}
                    title={`Preview merging ${id} into ${ranked[0]}`}
                  >
                    Merge into {ranked[0]}
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
