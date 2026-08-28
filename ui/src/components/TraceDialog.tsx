/**
 * One turn's trace, given room.
 *
 * The trace stays rendered inline as well. A wall you have to click to find is a wall people stop
 * reading, and step 4 is the ethical wall, so this is a roomier second view rather than the only
 * way to reach it.
 *
 * Composes the same `EvidenceFlow` / `QueryTrace` / `PassagesCited` / `FactsUsed` set the page
 * renders inline, rather than a second copy of them. A citation drawn two ways would be two claims
 * about what a citation is.
 */

import { useEffect } from 'react'

import type { QueryPassage } from '../api'
import type { TraceView } from '../trace'
import EvidenceFlow from './EvidenceFlow'
import { FactsUsed, PassagesCited } from './EvidencePanels'
import QueryTrace from './QueryTrace'

export default function TraceDialog({
  trace,
  tool,
  onClose,
  onOpenPassage,
  onExplain,
}: {
  trace: TraceView
  /** Which tool produced this. Named because `ask` and `compose` answer differently. */
  tool: string
  onClose: () => void
  onOpenPassage: (passage: QueryPassage) => void
  onExplain: (assertionId: string) => void
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div
        className="modal modal-trace"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label="Full trace"
      >
        <h3>How this answer was reached</h3>
        <p className="modal-sub">
          From <code>{tool}</code>. The same trace shown in the turn below, with room for the
          whole chain.
        </p>

        <EvidenceFlow
          passages={trace.passages}
          facts={trace.facts}
          blocks={trace.blocks}
          onOpenPassage={onOpenPassage}
          onExplain={onExplain}
        />

        <QueryTrace
          router={trace.router}
          gate={trace.gate}
          lanes={trace.lanes}
          blocks={trace.blocks}
          floor={trace.floor}
          warnings={trace.warnings}
          onOpenPassage={onOpenPassage}
        />

        <PassagesCited passages={trace.passages} onOpen={onOpenPassage} />
        <FactsUsed facts={trace.facts} floor={trace.floor} onExplain={onExplain} />

        <div className="modal-actions">
          <button className="btn btn-secondary" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  )
}
