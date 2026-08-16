/**
 * Create a matter, and file documents under one.
 *
 * A matter is a record now rather than a side effect of grouping facts, which is what makes both
 * of these possible. Before, a matter existed only because a document referred to it, so an empty
 * matter could not be created and a mistyped reference silently became a second matter that
 * nothing queried.
 *
 * The reference is the firm's own, not generated here: a matter already has one in the firm's
 * systems, and inventing a second guarantees the two diverge.
 */

import { useState } from 'react'

import FieldHelp from './FieldHelp'

export function CreateMatterDialog({
  busy,
  onCancel,
  onSubmit,
}: {
  busy: boolean
  onCancel: () => void
  onSubmit: (matterId: string, name: string) => void
}) {
  const [matterId, setMatterId] = useState('')
  const [name, setName] = useState('')
  const ready = matterId.trim().length > 0 && name.trim().length > 0

  return (
    <div className="modal-overlay" onClick={onCancel}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h3>New matter</h3>
        <p className="modal-sub">
          The matter exists as soon as you create it, before any document is filed under it. That
          is the order real work happens in: a team is staffed and an ethical screen raised before
          the first document arrives.
        </p>

        <div className="form-group">
          <label>
            Reference
            <FieldHelp text="Your firm's own reference, exactly as it appears in your systems. Not generated here: a second reference invented by this system would drift from the real one, and then two records describe one matter." />
          </label>
          <input
            value={matterId}
            onChange={(e) => setMatterId(e.target.value)}
            placeholder="NTL-2026-0114"
            autoFocus
          />
        </div>

        <div className="form-group">
          <label>Name</label>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Northwind Trading Ltd v Calder Shipping AG"
          />
          {!ready && (
            <p className="hint">
              Both are required. A list of bare references is unreadable, and a matter with no
              reference cannot be filed against.
            </p>
          )}
        </div>

        <div className="modal-actions">
          <button className="btn btn-ghost" onClick={onCancel}>
            Cancel
          </button>
          <button
            className="btn btn-primary"
            disabled={!ready || busy}
            onClick={() => onSubmit(matterId.trim(), name.trim())}
          >
            {busy ? 'Creating…' : 'Create matter'}
          </button>
        </div>
      </div>
    </div>
  )
}

export function LinkDocumentsDialog({
  count,
  matters,
  busy,
  onCancel,
  onSubmit,
}: {
  /** How many documents are selected, so the confirmation names a number. */
  count: number
  matters: { matter_id: string; name?: string }[]
  busy: boolean
  onCancel: () => void
  onSubmit: (matterId: string, reason: string) => void
}) {
  const [matterId, setMatterId] = useState('')
  const [reason, setReason] = useState('')

  return (
    <div className="modal-overlay" onClick={onCancel}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h3>
          File {count} document{count === 1 ? '' : 's'} under a matter
        </h3>
        <p className="modal-sub">
          Every fact read out of these documents moves with them. Citations are untouched: the
          matter is not part of what identifies a fact, so nothing is re-read and no page or quote
          changes.
        </p>

        <div className="consequence">
          <div className="consequence-title">This changes who can read these facts</div>
          <ul>
            <li>
              Matter access is by assignment, so a document moved into a matter somebody is not
              staffed on becomes invisible to them.
            </li>
            <li>
              Moved out of a matter they are screened from, it becomes visible. A screen follows
              the matter, not the document.
            </li>
            <li>
              Recorded on the Audit page with your name, the time, and the matter each document
              came from, because an access change made through a data operation is still an access
              change.
            </li>
          </ul>
        </div>

        <div className="form-group">
          <label>Matter</label>
          <select value={matterId} onChange={(e) => setMatterId(e.target.value)}>
            <option value="">Choose a matter…</option>
            {matters.map((m) => (
              <option key={m.matter_id} value={m.matter_id}>
                {m.matter_id}
                {m.name && m.name !== m.matter_id ? ` - ${m.name}` : ''}
              </option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label>
            Reason, optional
            <FieldHelp text="Filing a document is ordinary work rather than a withdrawal, so this is not required. It is kept when given, and it is what explains an unexpected move to whoever reads the trail later." />
          </label>
          <input
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            placeholder="Filed under the wrong matter on upload."
          />
        </div>

        <div className="modal-actions">
          <button className="btn btn-ghost" onClick={onCancel}>
            Cancel
          </button>
          <button
            className="btn btn-primary"
            disabled={!matterId || busy}
            onClick={() => onSubmit(matterId, reason.trim())}
          >
            {busy ? 'Filing…' : `File ${count} document${count === 1 ? '' : 's'}`}
          </button>
        </div>
      </div>
    </div>
  )
}
