/**
 * Which half of the graph a node belongs to, read from the ontology.
 *
 * The graph holds facts read out of documents and schema declared by a system of record, on
 * purpose — a metric over `matters` should reconcile with a fact from a page. But a reader
 * auditing a conflict check does not want a firm's Glue columns on screen.
 *
 * The layer comes from `entity_types[].layer` in the ontology response, so a domain pack adding
 * an entity kind needs no UI change and the healthcare pack gets the same grouping for free. No
 * entity prefix is named for that grouping, or anywhere else in the UI, with one exception that
 * `CATALOG_DETAIL_KINDS` below states its reasons for. Ids are built, never parsed.
 */

import type { Ontology } from './api'

/** A node whose kind the pack does not declare. Its own group, never hidden by default:
 *  an undeclared kind in the graph is drift worth seeing, and a filter that quietly files it
 *  under facts hides exactly what the closed vocabulary exists to surface. */
export const UNDECLARED = '__undeclared__'

const DEFAULT_LAYER = 'domain'

/** Labels for layer ids that need wording a lawyer would use. Unlisted layers are humanised,
 *  so a pack introducing one reads sensibly before anyone touches this. */
const LAYER_LABELS: Record<string, string> = {
  catalog: 'Catalog schema',
  metrics: 'Metrics',
}

/** The layer every shipped pack declares schema under. Named here so the one module that knows
 *  about layers stays the only one that does. */
export const CATALOG_LAYER = 'catalog'

export interface DetailKind {
  key: string
  /** Node `type` values this reveals. Every edge touching one of them is hidden until asked for. */
  types: readonly string[]
  label: string
  help: string
}

/** Catalog detail the explorer holds back, so the schema view opens at database and table level.
 *
 *  Not merely tidier: a column is a detail of a table rather than a peer of it, and a synonym or a
 *  topic is a model's proposal rather than something the catalogue declared.
 *
 *  The one place the UI names an id prefix, and it is unavoidable: of these kinds only `Column` is
 *  declared by any pack, while a description, a metric, a synonym and a topic node are minted by
 *  this system rather than by a vocabulary, so there is nothing to read the grouping out of.
 */
export const CATALOG_DETAIL_KINDS: readonly DetailKind[] = [
  {
    key: 'columns',
    types: ['column'],
    label: 'Columns',
    help: 'Every column of each table, as the catalogue scan declared it. Off by default because one wide table draws fifty of them, and a column belongs under its table rather than beside it.',
  },
  {
    key: 'metrics',
    types: ['metric'],
    label: 'Metrics',
    help: 'Governed metric definitions, joined to each table they read. Useful for the reverse question: which approved answers would a change to this table affect.',
  },
  {
    key: 'synonyms',
    types: ['synonym'],
    label: 'Synonyms',
    help: 'Other names somebody might say instead of the table name. A model proposed these, so they are pending until a reviewer approves them and are drawn faintly while they are.',
  },
  {
    key: 'topics',
    types: ['topic'],
    label: 'Topics',
    help: 'Subject-matter tags on a table. A model proposed these, so they are pending until a reviewer approves them and are drawn faintly while they are.',
  },
  {
    key: 'descriptions',
    types: ['description'],
    label: 'Descriptions',
    help: 'The description text attached to a table, which is what the SQL generator is given. Its node is identified by a digest of the text, so it reads better on the Tables page than on a canvas.',
  },
]

export interface LayerIndex {
  /** False when the pack did not load. Nothing may be asserted about a layer in that case:
   *  labelling every kind "not in the vocabulary" because a fetch failed is a lie. */
  loaded: boolean
  /** Layer for a node `type` — the lowercase id prefix the API sends. */
  layerOf: (nodeType: string) => string
  labelOf: (layer: string) => string
  /** The pack's own wording for an entity kind, so the legend does not show a raw id prefix. */
  entityLabelOf: (nodeType: string) => string
  /** Declared layers in pack order, then UNDECLARED. */
  order: string[]
}

function humanise(id: string): string {
  const s = id.replace(/[_-]+/g, ' ').trim()
  return s.charAt(0).toUpperCase() + s.slice(1)
}

export function buildLayerIndex(onto: Ontology | null): LayerIndex {
  const byType = new Map<string, string>()
  const labelByType = new Map<string, string>()
  const order: string[] = []

  for (const e of onto?.entity_types ?? []) {
    const layer = e.layer ?? DEFAULT_LAYER
    // `slug` is the prefix form the API sends as `type`; fall back to the id for an older
    // response, which is capitalised, hence the normalisation on both sides.
    const key = (e.slug ?? e.id).toLowerCase()
    byType.set(key, layer)
    labelByType.set(key, e.label)
    if (!order.includes(layer)) order.push(layer)
  }

  const factsLabel = onto ? `${humanise(onto.domain)} facts` : 'Facts'

  return {
    loaded: byType.size > 0,
    layerOf: (nodeType) => byType.get(nodeType.toLowerCase()) ?? UNDECLARED,
    entityLabelOf: (nodeType) => labelByType.get(nodeType.toLowerCase()) ?? humanise(nodeType),
    labelOf: (layer) => {
      if (layer === UNDECLARED) return 'Not in the vocabulary'
      if (layer === DEFAULT_LAYER) return factsLabel
      return LAYER_LABELS[layer] ?? humanise(layer)
    },
    order: [...order, UNDECLARED],
  }
}
