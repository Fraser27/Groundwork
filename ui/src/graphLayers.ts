/**
 * Which half of the graph a node belongs to, read from the ontology.
 *
 * The graph holds facts read out of documents and schema declared by a system of record, on
 * purpose — a metric over `matters` should reconcile with a fact from a page. But a reader
 * auditing a conflict check does not want a firm's Glue columns on screen.
 *
 * No entity prefix is named in this file, or anywhere in the UI. The layer comes from
 * `entity_types[].layer` in the ontology response, so a domain pack adding an entity kind
 * needs no UI change and the healthcare pack gets the same grouping for free.
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
