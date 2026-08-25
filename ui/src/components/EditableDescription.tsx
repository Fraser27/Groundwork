/**
 * A table or column description: what it says, where it came from, and what a model has proposed.
 *
 * The provenance tag is the load-bearing part. A description reaches the model that writes SQL for
 * ungoverned questions, so "a colleague wrote this" and "a model guessed this from the column name"
 * are different grounds for trusting a generated query, and rendering them identically would hide
 * the one distinction worth acting on.
 *
 * A proposal is shown **beside** the live text, never in place of it. That is what makes the review
 * gate legible: something has been suggested, it is not in force, and approving it is a decision
 * somebody takes rather than a state the page drifts into.
 */

import { useState } from 'react'

import type { DescriptionSource, PendingDescription } from '../api'

const SOURCE_LABEL: Record<Exclude<DescriptionSource, ''>, string> = {
  glue: 'from the catalog',
  human: 'written here',
  model: 'model, approved',
}

/** Green only for a person's words. A model's approved guess is legitimate but not the same thing. */
const SOURCE_TONE: Record<Exclude<DescriptionSource, ''>, string> = {
  glue: 'tag-neutral',
  human: 'tag-green',
  model: 'tag-orange',
}

export default function EditableDescription({
  value,
  source = '',
  pending,
  canEdit,
  canApprove,
  onSave,
  onApprove,
  placeholder = 'No description recorded.',
}: {
  value: string
  source?: DescriptionSource
  /** A model's unreviewed proposal, or null. Shown alongside, not instead of, `value`. */
  pending?: PendingDescription | null
  canEdit: boolean
  canApprove: boolean
  onSave: (text: string) => Promise<void> | void
  onApprove: () => Promise<void> | void
  placeholder?: string
}) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(value)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const save = async () => {
    const text = draft.trim()
    if (!text) {
      setError('Say what it means, or cancel. An empty description is not stored.')
      return
    }
    setBusy(true)
    setError('')
    try {
      await onSave(text)
      setEditing(false)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const approve = async () => {
    setBusy(true)
    setError('')
    try {
      await onApprove()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  if (editing) {
    return (
      <div className="desc-edit">
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          rows={3}
          maxLength={2000}
          placeholder="What does this hold, in one sentence?"
          autoFocus
        />
        {error && <p className="hint desc-error">{error}</p>}
        <div className="desc-actions">
          <button className="btn btn-primary btn-sm" onClick={save} disabled={busy}>
            {busy ? 'Saving' : 'Save'}
          </button>
          <button
            className="btn btn-ghost btn-sm"
            onClick={() => {
              setEditing(false)
              setDraft(value)
              setError('')
            }}
            disabled={busy}
          >
            Cancel
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="desc-read">
      <div className="desc-line">
        <span className={value ? undefined : 'dim'}>{value || placeholder}</span>
        {source && <span className={`tag ${SOURCE_TONE[source]}`}>{SOURCE_LABEL[source]}</span>}
        {canEdit && (
          <button className="btn btn-ghost btn-sm" onClick={() => setEditing(true)}>
            {value ? 'Edit' : 'Describe'}
          </button>
        )}
      </div>

      {pending && (
        <div className="desc-pending">
          <span className="tag tag-orange">proposed</span>
          <span className="dim">
            A model has suggested a description for this. It is not in use until approved.
          </span>
          {canApprove && (
            <button className="btn btn-approve btn-sm" onClick={approve} disabled={busy}>
              {busy ? 'Approving' : 'Approve'}
            </button>
          )}
        </div>
      )}

      {error && <p className="hint desc-error">{error}</p>}
    </div>
  )
}
