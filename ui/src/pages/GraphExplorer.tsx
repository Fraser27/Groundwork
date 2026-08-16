/**
 * GraphExplorer — force-directed canvas over the assertion graph.
 *
 * The edges are the point. Every edge is an assertion, drawn in the colour of its
 * epistemic class, so the shape of what the system believes — and how firmly — is
 * visible before anyone clicks anything. Pending edges are drawn faintly and
 * predicted ones dashed, because a guess must not look like a finding.
 *
 * Clicking an edge opens its provenance.
 *
 * `?highlight=<assertion_id>,...` emphasises named assertions and focuses on them — how an
 * answer from Ask is audited as a subgraph. The rest of the graph stays drawn: a traversal is
 * only checkable against the facts it did not use.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { api, type EpistemicClass, type GraphEdge, type GraphNode, type Matter } from '../api'
import { getTenantId } from '../auth'
import { EPISTEMIC, EPISTEMIC_ORDER, HELP } from '../epistemic'
import { useProvenance } from '../useProvenance'
import ConfidenceBar from '../components/ConfidenceBar'
import EpistemicBadge from '../components/EpistemicBadge'
import FieldHelp from '../components/FieldHelp'
import ProvenancePanel from '../components/ProvenancePanel'
import { EmptyState, ErrorState, Spinner } from '../components/Shared'

const NODE_COLOURS: Record<string, string> = {
  Matter: '#4361ee',
  Party: '#0d9488',
  Counsel: '#7c3aed',
  Document: '#d97706',
  Authority: '#dc2626',
  Court: '#0891b2',
  Deadline: '#db2777',
  Clause: '#65a30d',
  Topic: '#8b90a5',
}

const NODE_RADIUS: Record<string, number> = {
  Matter: 15,
  Party: 12,
  Counsel: 12,
  Document: 10,
  Authority: 10,
  Court: 9,
  Deadline: 9,
  Clause: 8,
  Topic: 7,
}

interface SimNode extends GraphNode {
  x: number
  y: number
  vx: number
  vy: number
}

/** Resolve a CSS custom property to a concrete colour for canvas drawing. */
function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim()
}

export default function GraphExplorer() {
  const tenant = getTenantId()
  const [searchParams, setSearchParams] = useSearchParams()
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const nodesRef = useRef<SimNode[]>([])
  const animRef = useRef(0)
  const tickRef = useRef<(() => void) | null>(null)
  const alphaRef = useRef(1)
  const dragRef = useRef<string | null>(null)
  const isPanning = useRef(false)
  const panStart = useRef({ x: 0, y: 0, panX: 0, panY: 0 })
  const downPos = useRef({ x: 0, y: 0, nodeId: null as string | null })

  const [nodes, setNodes] = useState<SimNode[]>([])
  const [edges, setEdges] = useState<GraphEdge[]>([])
  const [matters, setMatters] = useState<Matter[]>([])
  const [floor, setFloor] = useState(0.8)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [reloadKey, setReloadKey] = useState(0)
  const [fullscreen, setFullscreen] = useState(false)

  const [zoom, setZoom] = useState(1)
  const [panX, setPanX] = useState(0)
  const [panY, setPanY] = useState(0)
  const zoomRef = useRef(1)
  const panXRef = useRef(0)
  const panYRef = useRef(0)
  useEffect(() => {
    zoomRef.current = zoom
  }, [zoom])
  useEffect(() => {
    panXRef.current = panX
  }, [panX])
  useEffect(() => {
    panYRef.current = panY
  }, [panY])

  const [hovered, setHovered] = useState<{ kind: 'node' | 'edge'; id: string } | null>(null)
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [selectedEdge, setSelectedEdge] = useState<GraphEdge | null>(null)
  const { provenance, error: provError } = useProvenance(
    tenant,
    selectedEdge?.assertion_id ?? null,
  )

  const highlighted = useMemo(() => {
    const raw = searchParams.get('highlight')
    return new Set(
      (raw ?? '')
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean),
    )
  }, [searchParams])

  const [matterFilter, setMatterFilter] = useState('__all__')
  const [visibleClasses, setVisibleClasses] = useState<Set<EpistemicClass>>(
    // PREDICTED is off by default: it is a research hint, and showing it alongside
    // findings by default would misrepresent the graph.
    new Set(EPISTEMIC_ORDER.filter((c) => c !== 'PREDICTED')),
  )
  const [minConf, setMinConf] = useState(0)
  const [includePending, setIncludePending] = useState(true)
  const [governingOnly, setGoverningOnly] = useState(false)

  useEffect(() => {
    Promise.all([
      api.neighbourhood(tenant, { depth: 2 }),
      api.listMatters(tenant),
      api.getSettings(tenant),
    ])
      .then(([n, m, s]) => {
        const canvas = canvasRef.current
        const cw = canvas?.clientWidth || 900
        const ch = canvas?.clientHeight || 560
        const sim = n.nodes.map((node, i) => ({
          ...node,
          x: cw / 2 + Math.cos(i * 0.8) * (180 + Math.random() * 140),
          y: ch / 2 + Math.sin(i * 0.8) * (140 + Math.random() * 110),
          vx: 0,
          vy: 0,
        }))
        nodesRef.current = sim
        setNodes(sim)
        setEdges(n.edges)
        setMatters(m.matters)
        setFloor(s.min_confidence)
        setMinConf(0)
        setError('')
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false))
  }, [tenant, reloadKey])

  const nodeIndex = useMemo(() => new Map(nodes.map((n) => [n.id, n])), [nodes])

  const visibleEdges = useMemo(
    () =>
      edges.filter((e) => {
        // A highlighted edge ignores the filters. It was named in the link because an answer
        // rested on it, and a filter silently dropping it would read as the answer citing a
        // fact that is not in the graph.
        if (highlighted.has(e.assertion_id)) return true
        if (!visibleClasses.has(e.epistemic_class)) return false
        if (e.confidence < minConf) return false
        if (!includePending && e.review_state === 'PENDING') return false
        if (governingOnly && !e.governing) return false
        if (matterFilter !== '__all__') {
          // The assertion carries the matter, not the node — nodes are derived from entity
          // ids and have no matter at all. Endpoint ids are `kind:slug`, so a Matter entity
          // is `matter:<id>`; that catches an edge on the matter itself left unfiled.
          const entityId = `matter:${matterFilter}`
          const touches =
            e.matter_id === matterFilter || e.source === entityId || e.target === entityId
          if (!touches) return false
        }
        return true
      }),
    [edges, visibleClasses, minConf, includePending, governingOnly, matterFilter, highlighted],
  )

  const highlightEdges = useMemo(
    () => visibleEdges.filter((e) => highlighted.has(e.assertion_id)),
    [visibleEdges, highlighted],
  )

  /** Endpoints of the highlighted edges — the nodes the traversal actually passed through. */
  const highlightNodeIds = useMemo(() => {
    const ids = new Set<string>()
    for (const e of highlightEdges) {
      ids.add(e.source)
      ids.add(e.target)
    }
    return ids
  }, [highlightEdges])

  /** Named in the link but absent from the loaded graph — the overview is capped, so say so. */
  const missingHighlights = useMemo(
    () => [...highlighted].filter((id) => !edges.some((e) => e.assertion_id === id)),
    [highlighted, edges],
  )

  const visibleNodeIds = useMemo(() => {
    const ids = new Set<string>()
    for (const e of visibleEdges) {
      ids.add(e.source)
      ids.add(e.target)
    }
    return ids
  }, [visibleEdges])

  const nodeTypes = useMemo(() => [...new Set(nodes.map((n) => n.type))].sort(), [nodes])

  const wakeSim = useCallback((alpha = 0.5) => {
    alphaRef.current = Math.max(alphaRef.current, alpha)
    if (animRef.current === 0 && tickRef.current) {
      animRef.current = requestAnimationFrame(tickRef.current)
    }
  }, [])

  const screenToWorld = useCallback(
    (sx: number, sy: number) => ({
      x: (sx - panXRef.current) / zoomRef.current,
      y: (sy - panYRef.current) / zoomRef.current,
    }),
    [],
  )

  const fitToNodes = useCallback((targets: SimNode[]) => {
    if (targets.length === 0) return
    const canvas = canvasRef.current
    if (!canvas) return
    const cw = canvas.clientWidth
    const ch = canvas.clientHeight
    let minX = Infinity
    let maxX = -Infinity
    let minY = Infinity
    let maxY = -Infinity
    for (const n of targets) {
      minX = Math.min(minX, n.x)
      maxX = Math.max(maxX, n.x)
      minY = Math.min(minY, n.y)
      maxY = Math.max(maxY, n.y)
    }
    const w = maxX - minX || 100
    const h = maxY - minY || 100
    const pad = 90
    const z = Math.min((cw - pad * 2) / w, (ch - pad * 2) / h, 2.2)
    setZoom(z)
    setPanX(cw / 2 - ((minX + maxX) / 2) * z)
    setPanY(ch / 2 - ((minY + maxY) / 2) * z)
  }, [])

  const applyZoom = useCallback((next: number, cx: number, cy: number) => {
    const clamped = Math.min(Math.max(next, 0.15), 5)
    const old = zoomRef.current
    setPanX(cx - (cx - panXRef.current) * (clamped / old))
    setPanY(cy - (cy - panYRef.current) * (clamped / old))
    setZoom(clamped)
  }, [])

  /** Perpendicular distance from a point to a segment — for edge hit-testing. */
  const distToSegment = (
    px: number,
    py: number,
    ax: number,
    ay: number,
    bx: number,
    by: number,
  ) => {
    const dx = bx - ax
    const dy = by - ay
    const lenSq = dx * dx + dy * dy || 1
    let t = ((px - ax) * dx + (py - ay) * dy) / lenSq
    t = Math.max(0, Math.min(1, t))
    const cx = ax + t * dx
    const cy = ay + t * dy
    return Math.hypot(px - cx, py - cy)
  }

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const container = containerRef.current
    if (container) {
      canvas.width = container.clientWidth
      canvas.height = container.clientHeight
    }

    const visible = nodesRef.current.filter((n) => visibleNodeIds.has(n.id))
    const map = new Map(visible.map((n) => [n.id, n]))
    const z = zoomRef.current

    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.save()
    ctx.translate(panXRef.current, panYRef.current)
    ctx.scale(z, z)

    const labelColour = cssVar('--graph-label') || '#1a1d2e'
    const dimColour = cssVar('--text-dim') || '#6b7085'
    const haloColour = cssVar('--highlight-border') || 'rgba(217, 119, 6, 0.55)'
    const auditing = highlighted.size > 0

    // Edges, in epistemic-class colour. Line weight tracks confidence, opacity
    // tracks review state, and PREDICTED is dashed.
    for (const e of visibleEdges) {
      const a = map.get(e.source)
      const b = map.get(e.target)
      if (!a || !b) continue
      const colour = cssVar(EPISTEMIC[e.epistemic_class].colour.replace('var(', '').replace(')', ''))
      const isHovered = hovered?.kind === 'edge' && hovered.id === e.assertion_id
      const isSelected = selectedEdge?.assertion_id === e.assertion_id
      const isHighlit = highlighted.has(e.assertion_id)
      const touchesHoveredNode =
        hovered?.kind === 'node' && (e.source === hovered.id || e.target === hovered.id)

      ctx.save()
      // A halo under the stroke, not a recolour: the epistemic colour is the vocabulary a
      // reader has already learnt, and overwriting it to mean "used by this answer" would
      // hide how the fact was reached at the moment they are checking it.
      if (isHighlit) {
        ctx.strokeStyle = haloColour
        ctx.lineWidth = (7 + e.confidence * 2) / z
        ctx.lineCap = 'round'
        ctx.beginPath()
        ctx.moveTo(a.x, a.y)
        ctx.lineTo(b.x, b.y)
        ctx.stroke()
        ctx.lineCap = 'butt'
      }

      ctx.strokeStyle = colour || dimColour
      const baseAlpha =
        isSelected || isHovered || isHighlit
          ? 1
          : e.review_state === 'PENDING'
            ? 0.42
            : e.governing
              ? 0.85
              : 0.5
      // Context is kept, only pushed back. Filtering it away would remove the facts the
      // traversal declined to use, which is half of what makes it auditable.
      const pushedBack = auditing && !isHighlit && !isSelected && !isHovered
      ctx.globalAlpha = pushedBack ? baseAlpha * 0.3 : baseAlpha
      ctx.lineWidth =
        ((isSelected ? 3.4 : isHovered ? 3 : isHighlit ? 2.4 : 1) + e.confidence * 1.7) / z
      if (e.epistemic_class === 'PREDICTED') ctx.setLineDash([5 / z, 4 / z])
      ctx.beginPath()
      ctx.moveTo(a.x, a.y)
      ctx.lineTo(b.x, b.y)
      ctx.stroke()

      // Arrowhead, so direction is readable — REPRESENTS is not symmetric.
      const ang = Math.atan2(b.y - a.y, b.x - a.x)
      const r = (NODE_RADIUS[b.type] || 8) + 2
      const tipX = b.x - Math.cos(ang) * r
      const tipY = b.y - Math.sin(ang) * r
      const head = 7 / z
      ctx.setLineDash([])
      ctx.beginPath()
      ctx.moveTo(tipX, tipY)
      ctx.lineTo(tipX - head * Math.cos(ang - 0.4), tipY - head * Math.sin(ang - 0.4))
      ctx.lineTo(tipX - head * Math.cos(ang + 0.4), tipY - head * Math.sin(ang + 0.4))
      ctx.closePath()
      ctx.fillStyle = colour || dimColour
      ctx.fill()
      ctx.restore()

      if (z >= 1.15 || isHovered || isSelected || isHighlit || touchesHoveredNode) {
        const emphasis = isSelected || isHovered || isHighlit
        const mx = (a.x + b.x) / 2
        const my = (a.y + b.y) / 2
        ctx.fillStyle = emphasis ? labelColour : dimColour
        ctx.globalAlpha = auditing && !emphasis ? 0.4 : 1
        ctx.font = `${(emphasis ? 10.5 : 9) / Math.max(z, 0.6)}px Inter, system-ui, sans-serif`
        ctx.textAlign = 'center'
        ctx.fillText(e.predicate, mx, my - 5 / z)
        ctx.globalAlpha = 1
      }
    }

    // Nodes
    for (const n of visible) {
      const colour = NODE_COLOURS[n.type] || '#6c8cff'
      const radius = NODE_RADIUS[n.type] || 8
      const isHovered = hovered?.kind === 'node' && hovered.id === n.id
      const isSelected = selectedNode === n.id
      const onPath = highlightNodeIds.has(n.id)

      ctx.globalAlpha = auditing && !onPath && !isHovered && !isSelected ? 0.4 : 1
      if (onPath) {
        ctx.beginPath()
        ctx.arc(n.x, n.y, radius + 4.5 / Math.max(z, 0.6), 0, Math.PI * 2)
        ctx.strokeStyle = haloColour
        ctx.lineWidth = 2.5 / z
        ctx.stroke()
      }
      if (isHovered || isSelected) {
        ctx.shadowColor = colour
        ctx.shadowBlur = 13
      }
      ctx.beginPath()
      ctx.arc(n.x, n.y, radius, 0, Math.PI * 2)
      ctx.fillStyle = colour
      ctx.fill()
      ctx.shadowBlur = 0
      if (isHovered || isSelected) {
        ctx.strokeStyle = labelColour
        ctx.lineWidth = 2 / z
        ctx.stroke()
      }

      ctx.fillStyle = labelColour
      ctx.font = `${(n.type === 'Topic' ? 9.5 : 11) / Math.max(z * 0.85, 0.75)}px Inter, system-ui, sans-serif`
      ctx.textAlign = 'center'
      ctx.fillText(n.label, n.x, n.y + radius + 13 / Math.max(z * 0.85, 0.75))
      ctx.globalAlpha = 1
    }

    ctx.restore()
  }, [
    visibleEdges,
    visibleNodeIds,
    hovered,
    selectedNode,
    selectedEdge,
    highlighted,
    highlightNodeIds,
  ])

  // Force simulation. Sleeps when settled; wakeSim() restarts it.
  useEffect(() => {
    if (nodes.length === 0) return
    const byId = new Map(nodesRef.current.map((n) => [n.id, n]))

    const tick = () => {
      const alpha = alphaRef.current
      if (alpha > 0.001) alphaRef.current *= 0.991

      const visible = nodesRef.current.filter((n) => visibleNodeIds.has(n.id))
      if (alpha <= 0.001 && !dragRef.current) {
        draw()
        animRef.current = 0
        return
      }
      if (visible.length === 0) {
        draw()
        animRef.current = requestAnimationFrame(tick)
        return
      }

      const canvas = canvasRef.current
      const targetCx = (canvas?.clientWidth || 900) / 2 / zoomRef.current
      const targetCy = (canvas?.clientHeight || 560) / 2 / zoomRef.current
      let cx = 0
      let cy = 0
      for (const n of visible) {
        cx += n.x
        cy += n.y
      }
      cx /= visible.length
      cy /= visible.length
      for (const n of visible) {
        n.vx += (targetCx - cx) * 0.0002 * alpha
        n.vy += (targetCy - cy) * 0.0002 * alpha
      }

      for (let i = 0; i < visible.length; i++) {
        for (let j = i + 1; j < visible.length; j++) {
          const a = visible[i]
          const b = visible[j]
          const dx = b.x - a.x
          const dy = b.y - a.y
          const distSq = dx * dx + dy * dy || 1
          const dist = Math.sqrt(distSq)
          if (dist < 420) {
            const f = (950 * alpha) / distSq
            a.vx -= dx * f
            a.vy -= dy * f
            b.vx += dx * f
            b.vy += dy * f
          }
        }
      }

      for (const e of visibleEdges) {
        const a = byId.get(e.source)
        const b = byId.get(e.target)
        if (!a || !b) continue
        const dx = b.x - a.x
        const dy = b.y - a.y
        const dist = Math.hypot(dx, dy) || 1
        const f = ((dist - 140) * 0.005 * alpha) / dist
        a.vx += dx * f
        a.vy += dy * f
        b.vx -= dx * f
        b.vy -= dy * f
      }

      for (const n of visible) {
        if (n.id === dragRef.current) continue
        n.vx *= 0.88
        n.vy *= 0.88
        n.x += n.vx
        n.y += n.vy
      }

      draw()
      animRef.current = requestAnimationFrame(tick)
    }

    tickRef.current = tick
    animRef.current = requestAnimationFrame(tick)
    return () => {
      cancelAnimationFrame(animRef.current)
      animRef.current = 0
    }
  }, [nodes, visibleEdges, visibleNodeIds, draw])

  /** What the view fits to: the highlighted subgraph plus one hop, so the facts the answer
   *  did not use are on screen next to the ones it did. */
  const focusNodeIds = useMemo(() => {
    if (highlightNodeIds.size === 0) return visibleNodeIds
    const ids = new Set(highlightNodeIds)
    for (const e of visibleEdges) {
      if (highlightNodeIds.has(e.source)) ids.add(e.target)
      if (highlightNodeIds.has(e.target)) ids.add(e.source)
    }
    return ids
  }, [highlightNodeIds, visibleNodeIds, visibleEdges])

  // Re-heat and auto-fit when the filters change the visible set.
  useEffect(() => {
    if (visibleNodeIds.size === 0) return
    alphaRef.current = 0.7
    wakeSim(0.7)
    const t = setTimeout(
      () => fitToNodes(nodesRef.current.filter((n) => focusNodeIds.has(n.id))),
      650,
    )
    return () => clearTimeout(t)
  }, [visibleNodeIds, focusNodeIds, fitToNodes, wakeSim])

  useEffect(() => {
    draw()
  }, [panX, panY, zoom, draw])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (selectedEdge) setSelectedEdge(null)
        else if (fullscreen) setFullscreen(false)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [fullscreen, selectedEdge])

  const hitTest = useCallback(
    (sx: number, sy: number): { kind: 'node' | 'edge'; id: string } | null => {
      const { x, y } = screenToWorld(sx, sy)
      const visible = nodesRef.current.filter((n) => visibleNodeIds.has(n.id))
      for (let i = visible.length - 1; i >= 0; i--) {
        const n = visible[i]
        const r = (NODE_RADIUS[n.type] || 8) + 6
        if ((x - n.x) ** 2 + (y - n.y) ** 2 < r * r) return { kind: 'node', id: n.id }
      }
      // Nodes win over edges; edge tolerance scales with zoom so it stays clickable.
      const map = new Map(visible.map((n) => [n.id, n]))
      const tol = 6 / zoomRef.current
      for (const e of visibleEdges) {
        const a = map.get(e.source)
        const b = map.get(e.target)
        if (!a || !b) continue
        if (distToSegment(x, y, a.x, a.y, b.x, b.y) < tol) return { kind: 'edge', id: e.assertion_id }
      }
      return null
    },
    [screenToWorld, visibleNodeIds, visibleEdges],
  )

  const onMouseDown = (ev: React.MouseEvent) => {
    const rect = canvasRef.current?.getBoundingClientRect()
    if (!rect) return
    const hit = hitTest(ev.clientX - rect.left, ev.clientY - rect.top)
    downPos.current = { x: ev.clientX, y: ev.clientY, nodeId: null }
    if (hit?.kind === 'node') {
      dragRef.current = hit.id
      downPos.current.nodeId = hit.id
      wakeSim(0.3)
    } else if (hit?.kind === 'edge') {
      downPos.current.nodeId = `edge:${hit.id}`
    } else {
      isPanning.current = true
      panStart.current = {
        x: ev.clientX,
        y: ev.clientY,
        panX: panXRef.current,
        panY: panYRef.current,
      }
    }
  }

  const onMouseMove = (ev: React.MouseEvent) => {
    const rect = canvasRef.current?.getBoundingClientRect()
    if (!rect) return
    const sx = ev.clientX - rect.left
    const sy = ev.clientY - rect.top

    if (isPanning.current) {
      setPanX(panStart.current.panX + (ev.clientX - panStart.current.x))
      setPanY(panStart.current.panY + (ev.clientY - panStart.current.y))
      return
    }
    if (dragRef.current) {
      const { x, y } = screenToWorld(sx, sy)
      const n = nodesRef.current.find((m) => m.id === dragRef.current)
      if (n) {
        n.x = x
        n.y = y
        n.vx = 0
        n.vy = 0
      }
      return
    }
    setHovered(hitTest(sx, sy))
  }

  const onMouseUp = (ev: React.MouseEvent) => {
    const moved =
      Math.abs(ev.clientX - downPos.current.x) + Math.abs(ev.clientY - downPos.current.y)
    if (downPos.current.nodeId && moved < 5) {
      const id = downPos.current.nodeId
      if (id.startsWith('edge:')) {
        const e = edges.find((x) => x.assertion_id === id.slice(5))
        if (e) {
          setSelectedEdge(e)
          setSelectedNode(null)
        }
      } else {
        setSelectedNode(id)
        setSelectedEdge(null)
      }
    }
    dragRef.current = null
    isPanning.current = false
    downPos.current.nodeId = null
  }

  const toggleClass = (c: EpistemicClass) =>
    setVisibleClasses((prev) => {
      const next = new Set(prev)
      if (next.has(c)) next.delete(c)
      else next.add(c)
      return next
    })

  const selectedNodeObj = selectedNode ? nodeIndex.get(selectedNode) : null
  const selectedNodeEdges = selectedNode
    ? visibleEdges.filter((e) => e.source === selectedNode || e.target === selectedNode)
    : []
  const hoveredEdge =
    hovered?.kind === 'edge' ? edges.find((e) => e.assertion_id === hovered.id) : null

  const retry = () => {
    setLoading(true)
    setReloadKey((k) => k + 1)
  }

  const clearHighlight = () => {
    const next = new URLSearchParams(searchParams)
    next.delete('highlight')
    setSearchParams(next, { replace: true })
  }

  if (loading) return <Spinner />
  if (error)
    return (
      <ErrorState
        title="Could not load the graph"
        detail={error}
        onRetry={retry}
      />
    )
  if (nodes.length === 0)
    return (
      <EmptyState title="The graph is empty">
        No facts have been recorded for this tenant yet. Ingest a document from Documents, and the
        assertions drawn from it appear here.
      </EmptyState>
    )

  return (
    <div className={`graph-page${fullscreen ? ' graph-fullscreen' : ''}`}>
      {!fullscreen && (
        <div className="page-header">
          <h2>Graph</h2>
          <p>
            Every edge is an assertion, coloured by how it was reached and weighted by how confident
            the system is. Click an edge to see why it is believed.
          </p>
        </div>
      )}

      {highlighted.size > 0 && !fullscreen && (
        <div className="banner banner-info graph-audit-banner">
          <span>
            <strong>Auditing an answer.</strong> {highlightEdges.length} of {highlighted.size}{' '}
            {highlighted.size === 1 ? 'assertion' : 'assertions'} the answer used{' '}
            {highlightEdges.length === 1 ? 'is' : 'are'} ringed and drawn over the rest of the
            graph, which stays visible so you can see what the traversal did not use. Click one for
            its provenance.
            {missingHighlights.length > 0 && (
              <>
                {' '}
                {missingHighlights.length} did not come back in this view; the overview is
                capped, so open {missingHighlights.length === 1 ? 'it' : 'them'} from Audit.
              </>
            )}
          </span>
          <button
            className="btn btn-ghost btn-sm"
            style={{ marginLeft: 'auto' }}
            onClick={clearHighlight}
          >
            Clear
          </button>
        </div>
      )}

      <div className="graph-toolbar">
        <div className="graph-toolbar-left">
          <div className="toolbar-field">
            <label>
              Matter
              <FieldHelp text={HELP.matterWall} />
            </label>
            <select value={matterFilter} onChange={(e) => setMatterFilter(e.target.value)}>
              <option value="__all__">All matters</option>
              {matters
                .filter((m) => !m.walled)
                .map((m) => (
                  <option key={m.matter_id} value={m.matter_id}>
                    {m.matter_id} - {m.name}
                  </option>
                ))}
            </select>
          </div>
          <div className="toolbar-field">
            <label>
              Minimum confidence
              <FieldHelp text={HELP.confidenceFloor} />
            </label>
            <input
              type="range"
              min={0}
              max={0.99}
              step={0.01}
              value={minConf}
              onChange={(e) => setMinConf(Number(e.target.value))}
              style={{ minWidth: 130 }}
            />
          </div>
          <div className="toolbar-field">
            <label>Show</label>
            <div className="chip-toggles">
              <button
                className={`chip-toggle${includePending ? ' active' : ''}`}
                onClick={() => setIncludePending((v) => !v)}
                title="Include facts nobody has signed off yet. They are drawn faintly, and they never shape an answer while pending."
              >
                Unreviewed
              </button>
              <button
                className={`chip-toggle${governingOnly ? ' active' : ''}`}
                onClick={() => setGoverningOnly((v) => !v)}
                title={HELP.governingPredicate}
              >
                Governing only
              </button>
            </div>
          </div>
        </div>
        <div className="graph-toolbar-right">
          <span className="graph-stats">
            {visibleNodeIds.size} entities, {visibleEdges.length} assertions
            {minConf > 0 && ` · above ${minConf.toFixed(2)}`}
          </span>
        </div>
      </div>

      <div className="graph-container" ref={containerRef}>
        <canvas
          ref={canvasRef}
          onMouseDown={onMouseDown}
          onMouseMove={onMouseMove}
          onMouseUp={onMouseUp}
          onMouseLeave={onMouseUp}
          onWheel={(e) => {
            e.preventDefault()
            const rect = canvasRef.current?.getBoundingClientRect()
            if (!rect) return
            applyZoom(
              zoomRef.current * (e.deltaY < 0 ? 1.1 : 0.9),
              e.clientX - rect.left,
              e.clientY - rect.top,
            )
          }}
          // Drag/pan feedback is CSS :active on the canvas rather than a ref read,
          // so nothing about the pointer state is looked at during render.
          style={{ cursor: hovered ? 'pointer' : 'grab' }}
        />

        <div className="graph-legend">
          <div className="graph-legend-title">
            How each edge was reached
            <FieldHelp text={HELP.epistemicClass} align="right" />
          </div>
          {EPISTEMIC_ORDER.map((c) => {
            const on = visibleClasses.has(c)
            const n = edges.filter((e) => e.epistemic_class === c).length
            return (
              <button
                key={c}
                className={`graph-legend-row${on ? '' : ' off'}`}
                onClick={() => toggleClass(c)}
                title={`${EPISTEMIC[c].meaning} ${EPISTEMIC[c].trust}`}
              >
                <span
                  className={`graph-legend-line${c === 'PREDICTED' ? ' dashed' : ''}`}
                  style={{ '--epi-colour': EPISTEMIC[c].colour } as React.CSSProperties}
                />
                <span style={{ flex: 1 }}>{EPISTEMIC[c].label}</span>
                <span>{n}</span>
              </button>
            )
          })}
          {highlightEdges.length > 0 && (
            <>
              <div className="graph-legend-title" style={{ marginTop: 11 }}>
                This answer
              </div>
              <button
                className="graph-legend-row"
                onClick={() => fitToNodes(nodesRef.current.filter((n) => focusNodeIds.has(n.id)))}
                title="Recentre on the assertions this answer used"
              >
                <span className="graph-legend-line highlit" />
                <span style={{ flex: 1 }}>Used by the answer</span>
                <span>{highlightEdges.length}</span>
              </button>
            </>
          )}
          <div className="graph-legend-title" style={{ marginTop: 11 }}>
            Entities
          </div>
          {nodeTypes.map((t) => (
            <div className="graph-legend-row" key={t} style={{ cursor: 'default' }}>
              <span className="graph-legend-dot" style={{ background: NODE_COLOURS[t] || '#6c8cff' }} />
              <span>{t}</span>
            </div>
          ))}
        </div>

        <div className="graph-zoom-controls">
          <button
            onClick={() => {
              const c = canvasRef.current
              if (c) applyZoom(zoomRef.current * 1.3, c.clientWidth / 2, c.clientHeight / 2)
            }}
            title="Zoom in"
          >
            +
          </button>
          <button
            onClick={() => {
              const c = canvasRef.current
              if (c) applyZoom(zoomRef.current / 1.3, c.clientWidth / 2, c.clientHeight / 2)
            }}
            title="Zoom out"
          >
            −
          </button>
          <button
            onClick={() => fitToNodes(nodesRef.current.filter((n) => focusNodeIds.has(n.id)))}
            title="Fit to view"
          >
            ⤢
          </button>
          <button onClick={() => setFullscreen((f) => !f)} title={fullscreen ? 'Exit fullscreen (Esc)' : 'Fullscreen'}>
            {fullscreen ? '✕' : '⛶'}
          </button>
        </div>

        <div className="graph-zoom-indicator">
          {Math.round(zoom * 100)}%
          {hoveredEdge && (
            <>
              {' · '}
              {hoveredEdge.predicate} {hoveredEdge.confidence.toFixed(2)}
            </>
          )}
        </div>

        {/* Edge inspection, the provenance panel, in place. */}
        {selectedEdge && (
          <div className="graph-inspect">
            {provError ? (
              <ErrorState title="Could not load this provenance" detail={provError} />
            ) : provenance ? (
              <ProvenancePanel
                provenance={provenance}
                confidenceFloor={floor}
                onClose={() => setSelectedEdge(null)}
                compact
              />
            ) : (
              <Spinner />
            )}
          </div>
        )}

        {selectedNodeObj && !selectedEdge && (
          <div className="graph-inspect">
            <div className="graph-inspect-head">
              <span
                className="graph-legend-dot"
                style={{ background: NODE_COLOURS[selectedNodeObj.type] || '#6c8cff' }}
              />
              <strong>{selectedNodeObj.label}</strong>
              <button
                className="btn btn-ghost btn-sm"
                style={{ marginLeft: 'auto' }}
                onClick={() => setSelectedNode(null)}
                aria-label="Close"
              >
                ✕
              </button>
            </div>
            <div className="graph-inspect-body">
              <dl className="graph-inspect-meta">
                <div>
                  <dt>Type</dt>
                  <dd>{selectedNodeObj.type}</dd>
                </div>
                {selectedNodeObj.matter_id && (
                  <div>
                    <dt>Matter</dt>
                    <dd>
                      <code>{selectedNodeObj.matter_id}</code>
                    </dd>
                  </div>
                )}
                <div>
                  <dt>Assertions touching it</dt>
                  <dd>{selectedNodeEdges.length}</dd>
                </div>
              </dl>

              <div className="prov-section-title">
                Relationships
                <FieldHelp text="Click one to see the document page and words behind it, or the reasoning that produced it." />
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {selectedNodeEdges.map((e) => {
                  const other = nodeIndex.get(e.source === selectedNodeObj.id ? e.target : e.source)
                  const outbound = e.source === selectedNodeObj.id
                  return (
                    <button
                      key={e.assertion_id}
                      className="path-hop"
                      style={{
                        '--epi-colour': EPISTEMIC[e.epistemic_class].colour,
                        cursor: 'pointer',
                        textAlign: 'left',
                        font: 'inherit',
                        fontSize: 12.5,
                      } as React.CSSProperties}
                      onClick={() => setSelectedEdge(e)}
                    >
                      <EpistemicBadge
                        epistemicClass={e.epistemic_class}
                        size="sm"
                        showLabel={false}
                        tipPlacement="above"
                      />
                      <span style={{ flex: 1, minWidth: 0 }}>
                        {outbound ? '' : '← '}
                        <span className="prov-pred">{e.predicate}</span>{' '}
                        {other?.label ?? '-'}
                      </span>
                      <ConfidenceBar value={e.confidence} floor={floor} width={44} />
                    </button>
                  )
                })}
                {selectedNodeEdges.length === 0 && (
                  <p className="card-note">
                    No assertions on this entity pass the current filters.
                  </p>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
