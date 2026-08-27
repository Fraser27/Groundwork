/**
 * FieldHelp — a small "(?)" icon that reveals a tooltip on hover or keyboard focus.
 *
 * Used everywhere a term would be obvious to a data engineer and opaque to a lawyer.
 * If a label needs a glossary entry, it gets one of these.
 *
 * Positioning is measured rather than declared. The previous version was CSS-only and
 * always opened upward from a `position: absolute` box, which clipped in two ways: a
 * tooltip near the top of a scroll container was cut off by the container's edge, and
 * one inside a modal was cut off by the modal itself. CSS cannot see how much room is
 * available, so this measures on open and renders to a portal — escaping every
 * ancestor's `overflow` and stacking context — then flips or shifts to fit.
 */

import { useCallback, useEffect, useId, useLayoutEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'

import { useUnitText } from '../useUnitLabel'

const GAP = 8
const MARGIN = 10
const MAX_WIDTH = 320

interface Position {
  top: number
  left: number
  placement: 'above' | 'below'
}

export default function FieldHelp({
  text: rawText,
  title: rawTitle,
}: {
  /**
   * May contain `{unit}` / `{units}` / `{Unit}` / `{Units}`, substituted with whatever this
   * tenant's pack calls the unit work is organised by. Help text is written as a `const` in
   * `epistemic.ts` and cannot call a hook, so the placeholder is how it stays pack-neutral.
   */
  text: string
  /** Optional bolded lead-in, for terms that need naming as well as explaining. */
  title?: string
  /**
   * Retained so existing call sites keep compiling. Alignment is now measured, so
   * the hint is ignored.
   */
  align?: 'center' | 'right'
}) {
  const fill = useUnitText()
  const text = fill(rawText)
  const title = rawTitle ? fill(rawTitle) : rawTitle
  const label = title ? `${title}: ${text}` : text
  const anchorRef = useRef<HTMLSpanElement>(null)
  const tipRef = useRef<HTMLSpanElement>(null)
  const [open, setOpen] = useState(false)
  const [pos, setPos] = useState<Position | null>(null)
  const tipId = useId()

  const place = useCallback(() => {
    const anchor = anchorRef.current
    const tip = tipRef.current
    if (!anchor || !tip) return

    const a = anchor.getBoundingClientRect()
    const t = tip.getBoundingClientRect()

    // Flip below when there is not enough room above — the case that clipped inside
    // the provenance drawer.
    const roomAbove = a.top
    const placement: Position['placement'] =
      roomAbove < t.height + GAP + MARGIN ? 'below' : 'above'

    const top = placement === 'above' ? a.top - t.height - GAP : a.bottom + GAP

    // Centre on the icon, then pull back inside the viewport on either side.
    let left = a.left + a.width / 2 - t.width / 2
    left = Math.max(MARGIN, Math.min(left, window.innerWidth - t.width - MARGIN))

    setPos({ top, left, placement })
  }, [])

  // Measure after paint: the tooltip must be in the DOM to have a height, so the
  // first frame renders it hidden and off-screen.
  useLayoutEffect(() => {
    if (open) place()
  }, [open, place])

  useEffect(() => {
    if (!open) return
    const close = () => setOpen(false)
    // Scrolling or resizing invalidates a measured position, and re-measuring while
    // the page moves under the cursor looks broken. Closing is honest.
    window.addEventListener('scroll', close, true)
    window.addEventListener('resize', close)
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    window.addEventListener('keydown', onKey)
    return () => {
      window.removeEventListener('scroll', close, true)
      window.removeEventListener('resize', close)
      window.removeEventListener('keydown', onKey)
    }
  }, [open])

  return (
    <span
      ref={anchorRef}
      className="field-help"
      tabIndex={0}
      role="note"
      aria-label={label}
      aria-describedby={open ? tipId : undefined}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
    >
      <span className="field-help-icon" aria-hidden="true">
        ?
      </span>
      {open &&
        createPortal(
          <span
            ref={tipRef}
            id={tipId}
            role="tooltip"
            className={`field-help-tip is-open place-${pos?.placement ?? 'above'}`}
            style={{
              top: pos?.top ?? 0,
              left: pos?.left ?? 0,
              maxWidth: MAX_WIDTH,
              // Hidden until measured, so the first frame does not flash at 0,0.
              visibility: pos ? 'visible' : 'hidden',
            }}
          >
            {title && <strong>{title}</strong>}
            {text}
          </span>,
          document.body,
        )}
    </span>
  )
}
