/**
 * Merge two entity ids that turned out to name one thing.
 *
 * Two steps, because the API's `dry_run` defaults to true for a reason: a merge restates every
 * claim about the losing id and withdraws the conclusions resting on them. The reviewer sees that
 * list before deciding, not after.
 *
 * Direction is stated three times over — in the selects, in the arrow, and on the commit button —
 * because getting it backwards closes the node that should have survived and there is no undo.
 */

import { useState } from 'react'

import { api, type MergeResult } from '../api'
import { entityKind, entityLabel } from '../format'
import FieldHelp from './FieldHelp'

export default function MergeDialog({
  tenant,
  candidates,
  initialWinner,
  initialLoser,
  factsFor,
  onCancel,
  onMerged,
}: {
  tenant: string
  /** Ids offered for either side. A group's members, or every id in the graph for a manual pick. */
  candidates: string[]
  initialWinner: string
  initialLoser: string
  /** Current claims naming an id, so the direction is chosen knowing which side holds more. */
  factsFor?: (id: string) => number | undefined
  onCancel: () => void
  onMerged: (result: MergeResult) => void
}) {
  const [winner, setWinner] = useState(initialWinner)
  const [loser, setLoser] = useState(initialLoser)
  const [reason, setReason] = useState('')
  const [preview, setPreview] = useState<MergeResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  // Mirrors `_refuse_cross_kind` in src/documents/merge.py. Checked here only so a reviewer is not
  // shown a preview that plan_merge will happily produce and the commit will then refuse: the
  // dry run does not run these guards.
  const sameId = winner === loser
  const crossKind =
    !sameId && entityKind(winner).toLowerCase() !== entityKind(loser).toLowerCase()
  const refusal = sameId
    ? 'The same id is on both sides, so there is nothing to merge.'
    : crossKind
      ? `${entityKind(loser)} and ${entityKind(winner)} are different kinds of thing, so they are not one entity. A merge across kinds would move a fact onto a node of the wrong type.`
      : ''

  const ready = reason.trim().length > 0 && !refusal
  const nothingToDo = preview !== null && preview.affected.length === 0

  const run = async (dryRun: boolean) => {
    setBusy(true)
    setError('')
    try {
      const result = await api.mergeEntities(tenant, {
        losing_id: loser,
        winning_id: winner,
        reason: reason.trim(),
        dry_run: dryRun,
      })
      if (dryRun) setPreview(result)
      else onMerged(result)
    } catch (e) {
      setError(serverDetail(e))
    } finally {
      setBusy(false)
    }
  }

  const swap = () => {
    setWinner(loser)
    setLoser(winner)
    setPreview(null)
  }

  return (
    <div className="modal-overlay" onClick={onCancel}>
      <div className="modal modal-wide" onClick={(e) => e.stopPropagation()}>
        <h3>Merge two ids that name one thing</h3>
        <p className="modal-sub">
          One id survives and the other closes. Every claim about the closing id is restated about
          the surviving one, and conclusions that rested on those claims are withdrawn with them.
        </p>

        <div className="merge-direction">
          <div className="merge-side">
            <label>
              Closes
              <FieldHelp text="The fork. Nothing is deleted: each claim naming this id is closed and restated about the surviving id, so a dated read from before the merge still shows the graph as it was." />
            </label>
            <select
              value={loser}
              onChange={(e) => {
                setLoser(e.target.value)
                setPreview(null)
              }}
            >
              {candidates.map((id) => (
                <option key={id} value={id}>
                  {id}
                </option>
              ))}
            </select>
            <EntityLine id={loser} facts={factsFor?.(loser)} />
          </div>

          <div className="merge-arrow">
            <span aria-hidden="true">→</span>
            <button className="btn btn-ghost btn-sm" onClick={swap} disabled={busy}>
              Swap
            </button>
          </div>

          <div className="merge-side merge-side-winner">
            <label>
              Survives
              <FieldHelp text="The canonical id. Every restated claim points here afterwards, and this is the id a conflict check will walk to." />
            </label>
            <select
              value={winner}
              onChange={(e) => {
                setWinner(e.target.value)
                setPreview(null)
              }}
            >
              {candidates.map((id) => (
                <option key={id} value={id}>
                  {id}
                </option>
              ))}
            </select>
            <EntityLine id={winner} facts={factsFor?.(winner)} />
          </div>
        </div>

        <p className="merge-sentence">
          Merging <code>{loser}</code> into <code>{winner}</code>.{' '}
          <strong>{loser}</strong> will no longer name anything.
        </p>

        {refusal && (
          <div className="banner banner-warn">
            <span>{refusal}</span>
          </div>
        )}

        <div className="form-group">
          <label>
            Reason, required
            <FieldHelp text="Read by whoever asks later why two nodes became one. Say what established that they are the same company: “same registration number on the engagement letter” explains itself, “duplicate” does not." />
          </label>
          <textarea
            value={reason}
            onChange={(e) => {
              setReason(e.target.value)
              setPreview(null)
            }}
            placeholder="Both spellings appear on the same engagement letter, company number 04215567."
            autoFocus
          />
          {reason.trim().length === 0 && (
            <p className="hint">
              A merge collapses a distinction the graph was holding. Without a reason there is no
              record of what justified it.
            </p>
          )}
        </div>

        {error && (
          <div className="banner banner-error">
            <span>
              <strong>The server refused this merge.</strong> {error}
            </span>
          </div>
        )}

        {preview && (
          <div className="consequence">
            <div className="consequence-title">
              {nothingToDo
                ? 'Nothing would change'
                : `What a merge would do: ${preview.affected.length + preview.cascaded.length} fact${
                    preview.affected.length + preview.cascaded.length === 1 ? '' : 's'
                  } touched`}
            </div>
            {nothingToDo ? (
              <p className="merge-nothing">
                No current claim names <code>{loser}</code>, so there is nothing to restate. Either
                it was merged already or the fork only ever existed in a claim that has since been
                closed.
              </p>
            ) : (
              <>
                <ul>
                  <li>
                    <strong>
                      {preview.affected.length} claim
                      {preview.affected.length === 1 ? '' : 's'} restated.
                    </strong>{' '}
                    Each is closed and rewritten to name <code>{winner}</code>. The assertion id
                    changes, because an id hashes its endpoints, so this is a supersession with
                    its own audit event rather than an edit.
                  </li>
                  <li>
                    <strong>
                      {preview.cascaded.length} conclusion
                      {preview.cascaded.length === 1 ? '' : 's'} withdrawn.
                    </strong>{' '}
                    {preview.cascaded.length === 0
                      ? 'None rests on a restated claim without naming the merged id itself. A conflict flag about the closing id is restated above rather than withdrawn here.'
                      : 'They rest on a claim that is closing. Leaving them standing would cite a premise that no longer exists.'}
                  </li>
                </ul>
                <IdList label="Claims to be restated" ids={preview.affected} />
                {preview.cascaded.length > 0 && (
                  <IdList label="Conclusions to be withdrawn" ids={preview.cascaded} />
                )}
              </>
            )}
          </div>
        )}

        <div className="modal-actions">
          <button className="btn btn-ghost" onClick={onCancel} disabled={busy}>
            Cancel
          </button>
          {!preview ? (
            <button className="btn btn-primary" disabled={!ready || busy} onClick={() => run(true)}>
              {busy ? 'Checking…' : 'Preview the merge'}
            </button>
          ) : (
            <button
              className="btn btn-danger"
              disabled={!ready || busy || nothingToDo}
              onClick={() => run(false)}
            >
              {busy ? 'Merging…' : `Merge ${loser} into ${winner}`}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

function EntityLine({ id, facts }: { id: string; facts?: number }) {
  const kind = entityKind(id)
  return (
    <div className="merge-entity">
      <strong>{entityLabel(id)}</strong>
      <span className="merge-entity-meta">
        {kind && <span className="tag tag-neutral tag-mono">{kind}</span>}
        {typeof facts === 'number' && (
          <span>
            {facts} claim{facts === 1 ? '' : 's'}
          </span>
        )}
      </span>
    </div>
  )
}

/** Assertion ids, shown in full. They are what a reviewer looks up on the Audit page afterwards. */
function IdList({ label, ids }: { label: string; ids: string[] }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="consequence-preview">
      <span className="consequence-preview-label">
        {label} ({ids.length})
      </span>
      {open ? (
        <div className="merge-id-list">
          {ids.map((id) => (
            <code key={id}>{id}</code>
          ))}
        </div>
      ) : (
        <button className="btn btn-ghost btn-sm" onClick={() => setOpen(true)}>
          Show the {ids.length === 1 ? 'id' : `${ids.length} ids`}
        </button>
      )}
    </div>
  )
}

/** `request` throws `"409: {"detail":"..."}"`. The detail is the MergeError's own sentence, which
 *  names which id was refused and why, so it is unwrapped rather than shown as JSON. */
function serverDetail(e: unknown): string {
  const raw = (e as Error).message.replace(/^\d+:\s*/, '')
  try {
    const parsed: unknown = JSON.parse(raw)
    const detail = (parsed as { detail?: unknown }).detail
    if (typeof detail === 'string') return detail
    if (Array.isArray(detail)) {
      const first = detail[0] as { msg?: unknown } | undefined
      if (typeof first?.msg === 'string') return first.msg
    }
  } catch {
    // Not JSON, so the message is already the sentence.
  }
  return raw
}
