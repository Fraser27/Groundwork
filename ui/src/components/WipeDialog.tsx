/**
 * Confirm withdrawing the facts read out of a document, or a whole organising unit.
 *
 * The dialogue exists to correct a wrong mental model rather than to add friction. "Delete" makes
 * a lawyer think the record is gone, and it is not: the facts are closed, an as-of read before
 * this moment still reconstructs them, and the Audit page keeps who withdrew them and why. Saying
 * that here is the difference between someone using this confidently and avoiding it.
 *
 * The reason is mandatory for the same purpose it is mandatory on a screen: it is what somebody
 * reads months later when they ask why the graph disagrees with the file.
 */

import { useState } from 'react'

import FieldHelp from './FieldHelp'
import { useUnitLabel } from '../useUnitLabel'

export default function WipeDialog({
  scope,
  target,
  count,
  busy,
  onCancel,
  onSubmit,
}: {
  scope: 'document' | 'matter'
  target: string
  /** Facts currently attributed to it, so the confirmation names a number rather than a vague
   *  "everything". Undefined when the caller does not know, which is honest rather than zero. */
  count?: number
  busy: boolean
  onCancel: () => void
  onSubmit: (reason: string) => void
}) {
  const unit = useUnitLabel()
  const [reason, setReason] = useState('')
  const ready = reason.trim().length > 0
  const isMatter = scope === 'matter'

  return (
    <div className="modal-overlay" onClick={onCancel}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h3>
          Withdraw the facts from {isMatter ? unit.lower : ''} {target}?
        </h3>
        <p className="modal-sub">
          {typeof count === 'number'
            ? `${count} fact${count === 1 ? '' : 's'} will stop shaping answers.`
            : 'Every fact read out of it will stop shaping answers.'}{' '}
          {isMatter
            ? `This covers every document filed under the ${unit.lower}.`
            : 'The file itself stays where it is.'}
        </p>

        <div className="consequence">
          <div className="consequence-title">What actually happens</div>
          <ul>
            <li>
              <strong>The facts are closed, not deleted.</strong> They leave the current graph, so
              nothing answers from them again. A dated read from before now still shows them.
            </li>
            <li>
              <strong>It is recorded.</strong> Your name, the time and the reason go on the Audit
              page, so a withdrawal is part of the record rather than a gap in it.
            </li>
            <li>
              <strong>The document stays in storage.</strong> It is the original, so this can be
              read again afterwards and will produce whatever the current extractor finds.
            </li>
            <li>
              <strong>Conclusions drawn earlier are left standing.</strong> They were true when
              drawn and can still be traced to what they rested on. Withdrawing them too would
              claim the firm never held a belief it did hold.
            </li>
          </ul>
        </div>

        <div className="form-group">
          <label>
            Reason, required
            <FieldHelp text="Written for whoever reads the file in a year. Say why these facts should not stand: “extracted before the model was corrected” or “this {unit} was loaded by mistake” both explain themselves; “cleanup” does not." />
          </label>
          <textarea
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            placeholder={
              isMatter
                ? `This ${unit.lower} was loaded into the wrong tenant.`
                : 'Re-reading with the corrected extraction model.'
            }
            autoFocus
          />
          {!ready && (
            <p className="hint">
              A withdrawal cannot be saved without a reason. An unexplained one cannot be defended
              when somebody asks why the facts changed.
            </p>
          )}
        </div>

        <div className="modal-actions">
          <button className="btn btn-ghost" onClick={onCancel}>
            Cancel
          </button>
          <button
            className="btn btn-danger"
            disabled={!ready || busy}
            onClick={() => onSubmit(reason.trim())}
          >
            {busy ? 'Withdrawing…' : 'Withdraw the facts'}
          </button>
        </div>
      </div>
    </div>
  )
}
