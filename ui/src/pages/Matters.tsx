import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  api,
  type Assertion,
  type DocumentSummary,
  type Matter,
  type WithheldMatter,
} from '../api'
import { getTenantId } from '../auth'
import { useUnitLabel } from '../useUnitLabel'
import { HELP } from '../epistemic'
import ConfidenceBar from '../components/ConfidenceBar'
import EpistemicBadge from '../components/EpistemicBadge'
import FieldHelp from '../components/FieldHelp'
import { EmptyState, ErrorState, IngestPill, Spinner, Toast } from '../components/Shared'
import WipeDialog from '../components/WipeDialog'
import { fmtDate, fmtNum } from '../format'

export default function Matters() {
  const unit = useUnitLabel()
  const tenant = getTenantId()
  const [matters, setMatters] = useState<Matter[]>([])
  // Kept in its own piece of state, never merged into `matters`. A screened matter must
  // not be able to reach the readable list through a filter or a sort.
  const [withheld, setWithheld] = useState<WithheldMatter[]>([])
  const [docs, setDocs] = useState<DocumentSummary[]>([])
  const [assertions, setAssertions] = useState<Assertion[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [reloadKey, setReloadKey] = useState(0)
  const [filter, setFilter] = useState('')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [wiping, setWiping] = useState<string | null>(null)
  const [wipingBusy, setWipingBusy] = useState(false)
  const [toast, setToast] = useState<{ msg: string; type: string } | null>(null)

  const showToast = (msg: string, type = 'success') => {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 5000)
  }

  useEffect(() => {
    Promise.all([
      api.listMatters(tenant),
      api.listDocuments(tenant),
      // review_state null, not omitted: the endpoint defaults to PENDING, so omitting it loaded
      // only unreviewed facts -- which is why a matter with ten approved facts showed zero.
      api.listAssertions(tenant, { review_state: 'ALL', limit: 500 }),
    ])
      .then(([m, d, a]) => {
        setMatters(m.matters)
        setWithheld(m.withheld)
        setDocs(d)
        setAssertions(a)
        setError('')
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false))
  }, [tenant, reloadKey])

  const filtered = useMemo(() => {
    if (!filter.trim()) return matters
    const q = filter.toLowerCase()
    // Everything but matter_id is optional here. A matter is derived from the assertions
    // filed under it, so there is no record carrying a name until something names it, and
    // `m.name.toLowerCase()` threw the moment anyone typed in this box.
    return matters.filter((m) =>
      [m.matter_id, m.name].some((v) => (v ?? '').toLowerCase().includes(q)),
    )
  }, [matters, filter])

  const selected = matters.find((m) => m.matter_id === selectedId) ?? null

  /**
   * Per-matter counts, derived from the documents and facts already loaded.
   *
   * The detail panel was fixed to do this and the list row was not, so a matter with a document
   * and ten approved facts still read 0 and "-" across every column: the row was reading
   * `m.counts?.assertions`, and `counts` is not a field this API has ever sent. Same lying-type
   * bug as the detail panel had, one component lower.
   */
  const countsFor = useMemo(() => {
    const byMatter = new Map<
      string,
      { documents: number; assertions: number; pending: number; conflicts: number }
    >()
    const bump = (id: string | null | undefined) => {
      if (!id) return null
      let c = byMatter.get(id)
      if (!c) {
        c = { documents: 0, assertions: 0, pending: 0, conflicts: 0 }
        byMatter.set(id, c)
      }
      return c
    }
    // Guarded, not asserted with `!`. A document uploaded before a matter was required carries
    // none, so `bump` returns null and the non-null assertion crashed the whole page -- which is
    // what a `!` costs when the value can genuinely be absent.
    for (const d of docs) {
      const c = bump(d.matter_id)
      if (c) c.documents += 1
    }
    for (const a of assertions) {
      const c = bump(a.matter_id)
      if (!c) continue
      c.assertions += 1
      if (a.review_state === 'PENDING') c.pending += 1
      // The predicate is the conflict, not a separate flag: the ontology names the conclusion a
      // rule draws, so counting it here needs no extra field on the response.
      if (a.predicate === 'POTENTIAL_CONFLICT') c.conflicts += 1
    }
    return byMatter
  }, [docs, assertions])

  const retry = () => {
    setLoading(true)
    setReloadKey((k) => k + 1)
  }

  if (loading) return <Spinner />

  if (selected) {
    const matterDocs = docs.filter((d) => d.matter_id === selected.matter_id)
    const matterAssertions = assertions.filter((a) => a.matter_id === selected.matter_id)
    // Derived from the facts actually loaded rather than a `counts` object the API never sent.
    // Every stat card read `selected.counts?...`, which was always undefined, so each silently
    // fell through to a default -- and one of those defaults was a hardcoded zero.
    const matterPending = matterAssertions.filter((a) => a.review_state === 'PENDING').length
    // Was `selected.counts?.conflicts ?? 0` -- a field the API does not send, so this card read a
    // hardcoded zero and a matter with a live conflict displayed green.
    const matterConflicts = matterAssertions.filter(
      (a) => a.predicate === 'POTENTIAL_CONFLICT',
    ).length
    return (
      <>
        <button className="back-link btn-ghost" style={{ border: 'none', background: 'none', cursor: 'pointer' }} onClick={() => setSelectedId(null)}>
          ← Back to matters
        </button>

        <div className="page-header">
          <div className="page-header-row">
            <div>
              <h2>{selected.name || selected.matter_id}</h2>
              <p>
                <code>{selected.matter_id}</code>
                {selected.created_by && ` · opened by ${selected.created_by}`}
              </p>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <button
                className="btn btn-danger btn-sm"
                onClick={() => setWiping(selected.matter_id)}
                title="Withdraw every fact read out of this matter's documents"
              >
                Withdraw facts
              </button>
            </div>
          </div>
        </div>

        <div className="stats-grid">
          <div className="stat-card">
            <div className="label">Documents</div>
            <div className="value accent">{fmtNum(matterDocs.length)}</div>
          </div>
          <div className="stat-card">
            <div className="label">
              Facts
              <FieldHelp text={HELP.epistemicClass} />
            </div>
            <div className="value purple">
              {fmtNum(selected.assertion_count ?? matterAssertions.length)}
            </div>
          </div>
          <div className="stat-card">
            <div className="label">
              Pending review
              <FieldHelp text={HELP.reviewState} />
            </div>
            <div className={`value ${matterPending > 0 ? 'orange' : 'green'}`}>
              {fmtNum(matterPending)}
            </div>
            <div className="sub">
              <Link to="/review">Review queue</Link>
            </div>
          </div>
          <div className="stat-card">
            <div className="label">
              Potential conflicts
              <FieldHelp text="Inferred where the firm both acts for and opposes the same party. Fires only on facts declared by a system of record or confirmed by a check, a conflict flag resting on a model's guess would be worse than none." />
            </div>
            <div className={`value ${matterConflicts > 0 ? 'red' : 'green'}`}>
              {fmtNum(matterConflicts)}
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h3>Documents</h3>
            <Link to="/documents" className="btn btn-ghost btn-sm">
              Ingest pipeline
            </Link>
          </div>
          <table className="data-table">
            <thead>
              <tr>
                <th>File</th>
                <th>
                  State
                  <FieldHelp text={HELP.ingestState} />
                </th>
                <th className="num">Facts</th>
                <th className="num">Pending</th>
                <th>Uploaded</th>
              </tr>
            </thead>
            <tbody>
              {matterDocs.map((d) => (
                <tr key={d.document_id}>
                  <td>{d.filename}</td>
                  <td>
                    <IngestPill state={d.state} />
                  </td>
                  <td className="num">{fmtNum(d.assertion_count)}</td>
                  <td className="num">
                    {d.pending_review_count > 0 ? (
                      <span className="tag tag-orange">{d.pending_review_count}</span>
                    ) : (
                      '-'
                    )}
                  </td>
                  <td className="nowrap dim">{fmtDate(d.uploaded_at)}</td>
                </tr>
              ))}
              {matterDocs.length === 0 && (
                <tr>
                  <td colSpan={5} className="empty-state">
                    No documents on this matter yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        <div className="card">
          <div className="card-header">
            <h3>
              Facts on this matter
              <FieldHelp text={HELP.epistemicClass} />
            </h3>
            <Link to="/provenance" className="btn btn-ghost btn-sm">
              Audit view
            </Link>
          </div>
          <table className="data-table">
            <thead>
              <tr>
                <th>Claim</th>
                <th>How reached</th>
                <th>
                  Confidence
                  <FieldHelp text={HELP.confidence} />
                </th>
                <th>State</th>
              </tr>
            </thead>
            <tbody>
              {matterAssertions.map((a) => (
                <tr key={a.assertion_id}>
                  <td>
                    <strong>{a.subject_label || a.subject_id}</strong>{' '}
                    <span className="prov-pred">{a.predicate}</span>{' '}
                    <strong>{a.object_label || a.object_id}</strong>
                  </td>
                  <td>
                    <EpistemicBadge epistemicClass={a.epistemic_class} size="sm" />
                  </td>
                  <td>
                    <ConfidenceBar value={a.confidence} floor={0.8} />
                  </td>
                  <td className="nowrap dim">{a.review_state.replace('_', '-').toLowerCase()}</td>
                </tr>
              ))}
              {matterAssertions.length === 0 && (
                <tr>
                  <td colSpan={4} className="empty-state">
                    No facts recorded on this matter yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {wiping && (
          <WipeDialog
            scope="matter"
            target={wiping}
            count={matterAssertions.length}
            busy={wipingBusy}
            onCancel={() => setWiping(null)}
            onSubmit={async (reason) => {
              setWipingBusy(true)
              try {
                const r = await api.wipeMatter(tenant, wiping, reason)
                showToast(
                  `${r.assertions_superseded} facts withdrawn across ${r.documents.length} ` +
                    'documents. The Audit page records who withdrew them and why.',
                )
                setWiping(null)
                retry()
              } catch (e) {
                showToast((e as Error).message.replace(/^\d+:\s*/, ''), 'error')
              } finally {
                setWipingBusy(false)
              }
            }}
          />
        )}

        <Toast toast={toast} />
      </>
    )
  }

  return (
    <>
      <div className="page-header">
        <div className="page-header-row">
          <div>
            <h2>{unit.plural}</h2>
            <p>
              {unit.plural} are subgraphs of one tenant-wide graph, not separate graphs. A conflict
              check is by definition cross-{unit.singular.toLowerCase()}, and a shared party is the
              signal it reads.
            </p>
          </div>
        </div>
      </div>

      {error && (
        <ErrorState
          title={`Could not load ${unit.plural.toLowerCase()}`}
          detail={error}
          onRetry={retry}
        />
      )}

      {withheld.length > 0 && (
        <div className="withheld-block">
          <div className="withheld-block-head">
            <h3>
              {withheld.length} matter{withheld.length === 1 ? '' : 's'} withheld from you
              <FieldHelp text={HELP.ethicalScreen} />
            </h3>
            <span className="tag tag-red">Screened</span>
          </div>
          <p className="withheld-block-note">
            You cannot read these matters, their documents, or anything recorded on them. They are
            named here on purpose: if they were simply hidden, a conflict check could come back
            clean because the matching matter was invisible, and someone would proceed on it.
            Nothing here can be opened, and none of it appears in the list below.
          </p>
          <div className="withheld-list">
            {withheld.map((w) => (
              <div className="withheld-item" key={w.matter_id}>
                <div className="withheld-item-head">
                  <strong>{w.matter_id}</strong>
                  <code>withheld</code>
                </div>
                <div className="withheld-field">
                  <span className="withheld-field-label">Reason recorded</span>
                  {w.reason}
                </div>
                <div className="withheld-field">
                  <span className="withheld-field-label">Who to contact</span>
                  {w.contact ? (
                    w.contact
                  ) : (
                    <span className="dim">
                      No contact was given. Ask your risk team about this matter.
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="search-bar">
        <input
          placeholder={`Filter by ${unit.singular.toLowerCase()} id, name or client…`}
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        />
        {filter && (
          <span className="search-count">
            {filtered.length} of {matters.length}
          </span>
        )}
      </div>

      <div className="card">
        <table className="data-table data-table-hover">
          <thead>
            <tr>
              {/* No Client or Status column. A matter record carries a reference, a name and its
                  timestamps -- nothing sends either of those, so both rendered as an empty tag and
                  a dash on every row. A column that can never hold a value is worse than absent:
                  it reads as missing data rather than as a field that does not exist. */}
              <th>Matter</th>
              <th className="num">Documents</th>
              <th className="num">Facts</th>
              <th className="num">
                Pending
                <FieldHelp text={HELP.reviewState} align="right" />
              </th>
              <th className="num">Conflicts</th>
              <th>Created</th>
            </tr>
          </thead>
          <tbody>
            {/* Only readable matters reach here, a screened one never enters `matters`. */}
            {filtered.map((m) => (
              <tr key={m.matter_id} onClick={() => setSelectedId(m.matter_id)}>
                <td>
                  <strong>{m.name || m.matter_id}</strong>
                  <div className="dim" style={{ fontSize: 11.5 }}>
                    <code>{m.matter_id}</code>
                  </div>
                </td>
                <td className="num">{fmtNum(countsFor.get(m.matter_id)?.documents ?? 0)}</td>
                <td className="num">
                  {fmtNum(m.assertion_count ?? countsFor.get(m.matter_id)?.assertions ?? 0)}
                </td>
                <td className="num">
                  {countsFor.get(m.matter_id)?.pending ? (
                    <span className="tag tag-orange">{countsFor.get(m.matter_id)!.pending}</span>
                  ) : (
                    '-'
                  )}
                </td>
                <td className="num">
                  {countsFor.get(m.matter_id)?.conflicts ? (
                    <span className="tag tag-red">{countsFor.get(m.matter_id)!.conflicts}</span>
                  ) : (
                    '-'
                  )}
                </td>
                <td className="nowrap dim">{fmtDate(m.created_at)}</td>
              </tr>
            ))}
            {filtered.length === 0 && (
              <tr>
                <td colSpan={6}>
                  <EmptyState title={matters.length === 0 ? 'No matters yet' : 'No matters match'}>
                    {matters.length === 0
                      ? 'Matters arrive from the case management system as declared records. None have been loaded for this tenant.'
                      : 'Clear the filter to see every matter you can read.'}
                  </EmptyState>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </>
  )
}
