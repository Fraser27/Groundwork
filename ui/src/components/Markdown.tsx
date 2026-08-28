/**
 * Markdown for model prose, rendered rather than shown as source.
 *
 * A model asked to analyse something writes headings and bullets whether or not anyone asked it
 * to, and the answer card was rendering all of it into one wall of `##` and `**`. The structure is
 * the model's own reasoning made legible, so discarding it costs the reader real information.
 *
 * Indentation is honoured for the same reason rather than for tidiness. Flattened, a sub-bullet
 * under "this account shows:" reads as a finding in its own right instead of a detail of the one
 * above it, and on a page whose whole job is showing what rests on what, that is the wrong claim.
 *
 * Deliberately a small subset, and deliberately not a dependency. What renders here is untrusted
 * text a model wrote, and the interesting surface of every off-the-shelf renderer is its HTML
 * passthrough. This builds React elements and never touches `dangerouslySetInnerHTML`, so there is
 * no path at all from model output to markup.
 */

import type { ReactNode } from 'react'

const HEADING = /^(#{1,6})\s+(.+)$/
const BULLET = /^(\s*)[-*+]\s+(.+)$/
const ORDERED = /^(\s*)\d+[.)]\s+(.+)$/

/**
 * `**bold**`, `` `code` `` and `*italic*`.
 *
 * Underscores are deliberately not emphasis. This prose is full of field names -- `matter_id`,
 * `review_state`, `assertion_id` -- and pairing their underscores italicises the middle of an
 * identifier the reader may need to type.
 */
const INLINE = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/g

type Item = { text: string; indent: number; ordered: boolean }

type Block =
  | { kind: 'heading'; level: number; text: string }
  | { kind: 'para'; lines: string[] }
  | { kind: 'list'; items: Item[] }

function inline(text: string): ReactNode[] {
  return text.split(INLINE).map((part, i) => {
    if (part.length > 4 && part.startsWith('**') && part.endsWith('**')) {
      return <strong key={i}>{part.slice(2, -2)}</strong>
    }
    if (part.length > 2 && part.startsWith('`') && part.endsWith('`')) {
      return <code key={i}>{part.slice(1, -1)}</code>
    }
    if (part.length > 2 && part.startsWith('*') && part.endsWith('*')) {
      return <em key={i}>{part.slice(1, -1)}</em>
    }
    return part
  })
}

function parse(source: string): Block[] {
  const blocks: Block[] = []
  let para: string[] = []
  let items: Item[] = []

  const closeParagraph = () => {
    if (para.length) blocks.push({ kind: 'para', lines: para })
    para = []
  }
  const closeList = () => {
    if (items.length) blocks.push({ kind: 'list', items })
    items = []
  }

  for (const raw of source.split('\n')) {
    const line = raw.trimEnd()
    if (!line.trim()) {
      closeParagraph()
      closeList()
      continue
    }

    const heading = HEADING.exec(line)
    if (heading) {
      closeParagraph()
      closeList()
      blocks.push({ kind: 'heading', level: heading[1].length, text: heading[2] })
      continue
    }

    // Bullets and numbers accumulate into one block. Which of them nests under which is decided
    // at render time from the indent, so a numbered sub-list under a bullet stays subordinate
    // instead of being cut off into a sibling list by the change of marker.
    const bullet = BULLET.exec(line)
    const ordered = bullet ? null : ORDERED.exec(line)
    const match = bullet ?? ordered
    if (match) {
      closeParagraph()
      items.push({ text: match[2], indent: match[1].length, ordered: Boolean(ordered) })
      continue
    }

    closeList()
    para.push(line)
  }
  closeParagraph()
  closeList()
  return blocks
}

/**
 * A flat run of indent-tagged items as nested lists.
 *
 * Consecutive items sharing an indent and a marker form one list; anything indented further
 * belongs inside the item above it, recursively.
 */
function lists(items: Item[]): ReactNode[] {
  const out: ReactNode[] = []
  let i = 0
  while (i < items.length) {
    const { indent, ordered } = items[i]
    const entries: { item: Item; children: Item[] }[] = []
    while (i < items.length && items[i].indent === indent && items[i].ordered === ordered) {
      const parent = items[i]
      let end = i + 1
      while (end < items.length && items[end].indent > indent) end++
      entries.push({ item: parent, children: items.slice(i + 1, end) })
      i = end
    }
    const rendered = entries.map((entry, n) => (
      <li key={n}>
        {inline(entry.item.text)}
        {entry.children.length > 0 && lists(entry.children)}
      </li>
    ))
    const key = `${indent}-${out.length}`
    out.push(
      ordered ? (
        <ol key={key} className="md-list">
          {rendered}
        </ol>
      ) : (
        <ul key={key} className="md-list">
          {rendered}
        </ul>
      ),
    )
  }
  return out
}

export default function Markdown({ text }: { text: string }) {
  return (
    <div className="md">
      {parse(text).map((block, i) => {
        if (block.kind === 'heading') {
          // Six levels collapse to two. The card around this already owns an `h3` saying what the
          // prose is, and a model's `#` must not outrank it.
          return block.level <= 2 ? (
            <h4 key={i} className="md-heading">
              {inline(block.text)}
            </h4>
          ) : (
            <h5 key={i} className="md-heading md-heading-sub">
              {inline(block.text)}
            </h5>
          )
        }
        if (block.kind === 'list') {
          return (
            <div key={i} className="md-lists">
              {lists(block.items)}
            </div>
          )
        }
        // Joined with a space: a hard-wrapped paragraph is one paragraph, and honouring its line
        // breaks would ragged-edge prose the model never meant to break.
        return (
          <p key={i} className="md-para">
            {inline(block.lines.join(' '))}
          </p>
        )
      })}
    </div>
  )
}
