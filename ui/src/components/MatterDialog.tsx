/**
 * Create an organising unit, and file documents under one.
 *
 * The unit is a record now rather than a side effect of grouping facts, which is what makes both of
 * these possible. Before, it existed only because a document referred to it, so an empty one could
 * not be created and a mistyped reference silently became a second record that nothing queried.
 *
 * The reference is the firm's own, not generated here: the unit already has one in the firm's
 * systems, and inventing a second guarantees the two diverge.
 *
 * Every caption reads its noun from `useUnitLabel` -- Matter for law, Encounter for care, Facility
 * for lending, Case for retail. The file keeps its name because `matter_id` is the scoping key
 * everywhere in the API, the graph, Cedar and a Cognito group; renaming that to relabel a dialog
 * would be the wrong trade.
 */

import { useState } from 'react'

import FieldHelp from './FieldHelp'
import { useUnitLabel } from '../useUnitLabel'

export function CreateMatterDialog({
  busy,
  onCancel,
  onSubmit,
}: {
  busy: boolean
  onCancel: () => void
  onSubmit: (matterId: string, name: string) => void
}) {
  const unit = useUnitLabel()
  const [matterId, setMatterId] = useState('')
  const [name, setName] = useState('')
  const ready = matterId.trim().length > 0 && name.trim().length > 0

  return (
    <div className="modal-overlay" onClick={onCancel}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h3>New {unit.lower}</h3>
        <p className="modal-sub">
          The {unit.lower} exists as soon as you create it, before any document is filed under it.
          That is the order real work happens in: a team is staffed and an ethical screen raised
          before the first document arrives.
        </p>

        <div className="form-group">
          <label>
            Reference
            <FieldHelp
              text={`Your firm's own reference, exactly as it appears in your systems. Not generated here: a second reference invented by this system would drift from the real one, and then two records describe one ${unit.lower}.`}
            />
          </label>
          {/* Placeholders are deliberately generic. A worked example would have to be a legal
              reference or a retail one, and whichever was hardcoded would read as wrong in every
              other pack -- the same leak that put "Choose a matter" under a Facilities heading. */}
          <input
            value={matterId}
            onChange={(e) => setMatterId(e.target.value)}
            placeholder="Your reference"
            autoFocus
          />
        </div>

        <div className="form-group">
          <label>Name</label>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="A short description somebody will recognise"
          />
          {!ready && (
            <p className="hint">
              Both are required. A list of bare references is unreadable, and a {unit.lower} with no
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
            {busy ? 'Creating…' : `Create ${unit.lower}`}
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
  const unit = useUnitLabel()
  const [matterId, setMatterId] = useState('')
  const [reason, setReason] = useState('')

  return (
    <div className="modal-overlay" onClick={onCancel}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h3>
          File {count} document{count === 1 ? '' : 's'} under a {unit.lower}
        </h3>
        <p className="modal-sub">
          Every fact read out of these documents moves with them. Citations are untouched: the{' '}
          {unit.lower} is not part of what identifies a fact, so nothing is re-read and no page or
          quote changes.
        </p>

        <div className="consequence">
          <div className="consequence-title">This changes who can read these facts</div>
          <ul>
            <li>
              {unit.singular} access is by assignment, so a document moved into a {unit.lower}{' '}
              somebody is not staffed on becomes invisible to them.
            </li>
            <li>
              Moved out of a {unit.lower} they are screened from, it becomes visible. A screen
              follows the {unit.lower}, not the document.
            </li>
            <li>
              Recorded on the Audit page with your name, the time, and the {unit.lower} each document
              came from, because an access change made through a data operation is still an access
              change.
            </li>
          </ul>
        </div>

        <div className="form-group">
          <label>{unit.singular}</label>
          <select value={matterId} onChange={(e) => setMatterId(e.target.value)}>
            <option value="">Choose a {unit.lower}…</option>
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
            placeholder={`Filed under the wrong ${unit.lower} on upload.`}
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
