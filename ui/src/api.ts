/**
 * Typed client for the LexGraph API.
 *
 * Every tenant-scoped path takes the tenant id explicitly. The server derives the
 * real tenant from the verified JWT and will reject a mismatch — the value here is
 * only for routing, never for authorisation.
 */

import { getAccessToken, isAuthEnabled } from './auth'

const BASE = '/api'

async function request<T>(path: string, opts?: RequestInit): Promise<T> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }

  if (isAuthEnabled()) {
    const token = getAccessToken()
    if (token) headers['Authorization'] = `Bearer ${token}`
  }

  const res = await fetch(`${BASE}${path}`, { headers, ...opts })

  if (res.status === 401) {
    if (isAuthEnabled()) {
      localStorage.clear()
      window.location.href = '/'
    }
    throw new Error('Unauthorized')
  }

  if (!res.ok) {
    throw new Error(`${res.status}: ${await res.text()}`)
  }
  return res.json()
}

// ── Epistemic contract ───────────────────────────────────────────────────────

/** Mirrors src/graph/assertions.py :: EpistemicClass. */
export type EpistemicClass =
  | 'DECLARED'
  | 'EXTRACTED_DET'
  | 'EXTRACTED_MODEL'
  | 'INFERRED'
  | 'PREDICTED'

export type ReviewState = 'AUTO_ASSERTED' | 'PENDING' | 'APPROVED' | 'REJECTED'

/**
 * Where a fact came from, in terms a person can check by hand.
 *
 * For a document that is file + page + verbatim quote: open the PDF at the page and
 * search for the sentence. Character offsets index the extracted text buffer rather
 * than the PDF, so they are debug metadata and never the citation.
 */
export interface SourceLocator {
  // Unstructured
  document_id?: string | null
  filename?: string | null
  page?: number | null
  chunk_id?: string | null
  quote?: string | null
  span_sha256?: string | null
  /** Debug only. Offsets into the extracted text, not into the PDF. */
  char_start?: number | null
  char_end?: number | null
  // Structured
  source_id?: string | null
  table?: string | null
  column?: string | null
  query_sha256?: string | null
}

export interface Assertion {
  assertion_id: string
  tenant_id: string
  matter_id?: string | null
  subject_id: string
  subject_label?: string
  subject_type?: string
  predicate: string
  object_id: string
  object_label?: string
  object_type?: string
  epistemic_class: EpistemicClass
  method: string
  confidence: number
  source_locator: SourceLocator
  premises: string[]
  rule_id?: string | null
  rule_version?: string | null
  valid_from?: string | null
  valid_until?: string | null
  recorded_at: string
  superseded_at?: string | null
  review_state: ReviewState
  reviewed_by?: string | null
  reviewed_at?: string | null
  /** Wider surrounding text, so a reviewer can judge the quote in context. */
  source_context?: string | null
}

/** One node of a proof tree: an assertion plus the assertions it rests on. */
export interface ProofNode {
  assertion: Assertion
  premises: ProofNode[]
}

/**
 * The citation for a document-sourced fact: which file, which page, which words.
 *
 * `download_url` is minted per request and expires, so it is never cached — a stale
 * one is refetched rather than reused.
 */
export interface PageCitation {
  document_id: string
  filename: string
  page: number
  quote: string
  chunk_id?: string | null
  page_count?: number | null
  context_before?: string | null
  context_after?: string | null
  span_sha256?: string | null
  download_url?: string | null
  expires_at?: string | null
  /** Debug only. Offsets into the extracted text, not into the PDF. */
  char_start?: number | null
  char_end?: number | null
}

export interface Provenance {
  assertion: Assertion
  /** Populated for document-sourced assertions. */
  citation?: PageCitation | null
  /** Populated for INFERRED assertions. */
  proof?: ProofNode | null
  /** Retraction / supersession trail, newest first. */
  history?: ProvenanceEvent[]
}

/** A short-lived link to the original file, minted per request. */
export interface DocumentDownload {
  document_id: string
  filename: string
  content_type?: string | null
  page_count?: number | null
  download_url: string
  expires_at?: string | null
}

export interface ProvenanceEvent {
  event_id: string
  timestamp: string
  action: 'ASSERTED' | 'APPROVED' | 'REJECTED' | 'SUPERSEDED' | 'RETRACTED'
  actor: string
  note?: string | null
}

// ── Tenancy, matters, sources ────────────────────────────────────────────────

export interface Tenant {
  tenant_id: string
  name: string
  ontology_domain: string
  created_at?: string | null
}

export interface Matter {
  matter_id: string
  name: string
  client?: string | null
  status: string
  opened_at?: string | null
  /** Ethical wall: this matter is walled off from the calling user. */
  walled?: boolean
  counts?: {
    documents: number
    assertions: number
    pending_review: number
    conflicts: number
  }
}

/**
 * A matter the caller is screened from.
 *
 * Named rather than counted. Within one firm a screen is disclosed, because a
 * conflict check that comes back clean only because the matching matters were
 * invisible is the harm the wall exists to prevent.
 */
export interface WithheldMatter {
  matter_id: string
  reason: string
  contact?: string | null
}

export interface MattersResponse {
  matters: Matter[]
  withheld: WithheldMatter[]
}

export interface Source {
  source_id: string
  name: string
  kind: 'GLUE' | 'REDSHIFT' | 'S3'
  database?: string | null
  region?: string | null
  table_count: number
  status: string
  last_scanned_at?: string | null
}

export interface TableSummary {
  full_name: string
  name: string
  database: string
  source_id: string
  description?: string
  row_count?: number | null
  /** Structured metadata enters the graph as DECLARED. */
  epistemic_class: EpistemicClass
}

export interface Column {
  name: string
  data_type: string
  description?: string
  is_partition: boolean
  is_primary_key: boolean
}

export interface TableDetail extends TableSummary {
  columns: Column[]
  /** The catalog scan that declared this table. */
  method: string
  scanned_at?: string | null
}

// ── Documents and the ingest state machine ───────────────────────────────────

export const INGEST_STATES = [
  'REGISTERED',
  'FETCHING',
  'PARSING',
  'CHUNKING',
  'EXTRACTING',
  'EMBEDDING',
  'GRAPH_STAGED',
  'PENDING_REVIEW',
  'LIVE',
] as const

export type IngestState =
  | (typeof INGEST_STATES)[number]
  | 'FETCH_FAILED'
  | 'PARSE_FAILED'
  | 'CHUNK_FAILED'
  | 'EXTRACT_FAILED'
  | 'EMBED_FAILED'
  | 'GRAPH_FAILED'

export interface DocumentSummary {
  document_id: string
  filename: string
  matter_id?: string | null
  state: IngestState
  uploaded_at: string
  page_count?: number | null
  size_bytes?: number | null
  assertion_count: number
  pending_review_count: number
  error?: string | null
}

export interface DocumentDetail extends DocumentSummary {
  s3_uri: string
  content_sha256?: string | null
  /** Ordered ingest transitions, so a stuck document is diagnosable. */
  timeline: { state: IngestState; at: string; detail?: string | null }[]
  assertions: Assertion[]
}

/**
 * A presigned POST. `fields` must be appended to the form *before* the file, and the
 * key is server-chosen — it carries the tenant prefix the IAM boundary relies on, so
 * nothing here is safe for the client to rewrite.
 */
export interface UploadTicket {
  upload_url: string
  fields: Record<string, string>
  key: string
  upload_id: string
  expires_in: number
  max_bytes: number
}

/** One pass of the pipeline over one document. Mirrors src/documents/models.py. */
export interface IngestJob {
  job_id: string
  document_id: string
  matter_id?: string | null
  state: IngestState
  reason?: string | null
  is_failed: boolean
  is_terminal: boolean
  /** Which phase a retry restarts from. Null when the state is not a recoverable failure. */
  retry_target?: IngestState | null
  chunk_count: number
  created_at: string
  updated_at: string
  history: { state: IngestState; at: string; reason?: string | null }[]
}

export interface IngestEvent {
  job_id: string
  document_id: string
  state: IngestState
  reason?: string | null
}

/** A user as an admin sees them. `status` is Cognito's, e.g. FORCE_CHANGE_PASSWORD. */
export interface TenantUser {
  user_id: string
  email: string
  display_name: string
  status: string
  enabled: boolean
  created_at: string
}

export interface CreatedUser {
  user_id: string
  email: string
  display_name: string
  status: string
  tenant_id: string
  note: string
}

// ── Governed metrics ─────────────────────────────────────────────────────────

export interface MetricParameter {
  column: string
  operator: string
  required: boolean
  description?: string
}

export interface Metric {
  metric_id: string
  name: string
  definition: string
  expression: string
  source_table: string
  source_id?: string
  grain: string[]
  time_grain_column?: string | null
  time_grains: string[]
  aggregation: 'additive' | 'semi_additive' | 'non_additive'
  parameters: MetricParameter[]
  filters: string[]
  synonyms: string[]
  status: 'draft' | 'approved' | 'deprecated'
  version: number
  owner?: string | null
  updated_by?: string | null
  updated_at?: string | null
}

// ── Query resolution ─────────────────────────────────────────────────────────

/** Which tier of the read path answered. Tier 1 is the only LLM-free one. */
export type ResolutionTier = 1 | 2 | 3 | 4

export interface QueryCitation {
  assertion_id: string
  label: string
  epistemic_class: EpistemicClass
  confidence: number
  document_id?: string | null
  filename?: string | null
  page?: number | null
  quote?: string | null
}

export interface QueryResult {
  question: string
  tier: ResolutionTier
  tier_reason: string
  answer: string
  /** Present for tiers 1, 3 and 4. Tier 1 SQL is compiled, not generated. */
  sql?: string | null
  metric_id?: string | null
  /** Present for tiers 2 and 3 — the assertions the traversal walked. */
  path?: {
    subject_label: string
    predicate: string
    object_label: string
    assertion_id: string
    epistemic_class: EpistemicClass
    confidence: number
  }[]
  citations: QueryCitation[]
  rows?: { columns: string[]; rows: (string | number | null)[][] } | null
  blocked?: boolean
  blocked_reason?: string | null
}

// ── Graph ────────────────────────────────────────────────────────────────────

export interface GraphNode {
  id: string
  label: string
  type: string
  matter_id?: string | null
  properties?: Record<string, unknown>
}

export interface GraphEdge {
  /** Edges are assertions — this is the id you look up provenance with. */
  assertion_id: string
  source: string
  target: string
  predicate: string
  epistemic_class: EpistemicClass
  confidence: number
  review_state: ReviewState
  governing: boolean
}

export interface Neighbourhood {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

// ── Ontology ─────────────────────────────────────────────────────────────────

export interface EntityDef {
  id: string
  label: string
  description: string
  help?: string | null
}

export interface PredicateDef {
  id: string
  label: string
  description: string
  governing: boolean
  domain: string[]
  range: string[]
  help?: string | null
  symmetric?: boolean
}

export interface RuleDef {
  id: string
  version: string
  description: string
  when: string[]
  then: string
  min_premise_class: EpistemicClass
  help?: string | null
}

export interface Ontology {
  domain: string
  version: number
  entity_types: EntityDef[]
  governing_predicates: PredicateDef[]
  descriptive_predicates: PredicateDef[]
  rules: RuleDef[]
}

// ── Dashboard / admin ────────────────────────────────────────────────────────

export interface DashboardStats {
  assertions_by_class: Record<EpistemicClass, number>
  pending_review: number
  documents_by_state: Partial<Record<IngestState, number>>
  matters: number
  metrics: { total: number; approved: number }
  recent_activity: ActivityEvent[]
}

export interface ActivityEvent {
  event_id: string
  timestamp: string
  actor: string
  action: string
  detail: string
  epistemic_class?: EpistemicClass | null
}

// ── Access: assignments, ethical screens, audit ──────────────────────────────

/**
 * Why a matter is or is not readable. Mirrors src/access.py :: AccessDecision.
 *
 * A bool would enforce access without explaining it. `SCREENED` names a contact;
 * `NOT_ASSIGNED` is an ordinary staffing gap and nobody decided anything.
 */
export type AccessDecision = 'ALLOWED' | 'SCREENED' | 'NOT_ASSIGNED' | 'PLATFORM_ADMIN'

export interface MatterAssignment {
  tenant_id: string
  user_id: string
  matter_id: string
  role: string
  granted_by: string
  granted_at: string
  /** Set rather than deleted. An assignment that once existed stays visible. */
  revoked_at?: string | null
  revoked_by?: string | null
}

export interface MatterScreen {
  tenant_id: string
  user_id: string
  matter_id: string
  /** Required by the API. Shown verbatim to the person screened. */
  reason: string
  screened_by: string
  screened_at: string
  contact?: string | null
  lifted_at?: string | null
  lifted_by?: string | null
}

export interface AccessEvent {
  event_id: string
  tenant_id: string
  actor: string
  action: 'ASSIGN' | 'UNASSIGN' | 'SCREEN' | 'LIFT_SCREEN'
  subject_user: string
  matter_id: string
  at: string
  reason?: string | null
  detail?: Record<string, unknown>
}

/** One matter a user can or cannot reach, with the resolved reason. */
export interface ResolvedAccess {
  matter_id: string
  matter_name?: string | null
  decision: AccessDecision
  /** The sentence the user themselves would be shown. */
  explanation: string
  role?: string | null
  reason?: string | null
  contact?: string | null
}

export interface UserAccess {
  user_id: string
  display_name?: string | null
  is_platform_admin: boolean
  assignments: MatterAssignment[]
  screens: MatterScreen[]
  decisions: ResolvedAccess[]
}

/** A member of a matter's team, or someone screened off it. */
export interface MatterTeamMember {
  user_id: string
  display_name?: string | null
  role: string
  granted_by: string
  granted_at: string
}

export interface MatterScreenedMember {
  user_id: string
  display_name?: string | null
  reason: string
  contact?: string | null
  screened_by: string
  screened_at: string
  /** True where the person also holds an assignment the screen overrides. */
  overrides_assignment: boolean
}

export interface MatterAccessDetail {
  matter_id: string
  matter_name?: string | null
  team: MatterTeamMember[]
  screened: MatterScreenedMember[]
}

/** A person who can be staffed onto a matter. Directory, not authorisation. */
export interface DirectoryUser {
  user_id: string
  display_name?: string | null
  is_platform_admin?: boolean
}

/** What a reset removes. Mirrors src/admin_ops.py :: ResetScope. */
export interface ResetScope {
  graph: boolean
  vectors: boolean
  jobs: boolean
  catalog: boolean
  metrics: boolean
}

export interface ResetReport {
  assertions_dropped: number
  documents_forgotten: number
  vectors_dropped: number
  jobs_dropped: number
  tables_forgotten: number
  metrics_dropped: number
  metrics_preserved: number
  errors: string[]
  s3_preserved?: boolean
  note: string
}

export interface ReplayReport {
  documents_found: number
  documents_ingested: number
  documents_failed: number
  tables_rescanned?: number
  errors: string[]
  note: string
}

export interface SampleDataReport {
  documents_loaded: number
  documents_skipped?: number
  chunks: number
  errors: string[]
  note: string
}

export interface ScanReport {
  source_id: string
  tables_found: number
  assertions_live: number
  scan_errors: string[]
  graph_error?: string | null
  note: string
}

export interface TenantSettings {
  tenant_id: string
  name: string
  ontology_domain: string
  /** Retrieval trust floor — assertions below this never shape an answer. */
  min_confidence: number
  block_ungoverned_queries: boolean
  extraction_model: string
  synthesis_model: string
  embedding_model: string
  available_models: { id: string; label: string }[]
  available_domains: string[]
}

// ── Calls ────────────────────────────────────────────────────────────────────

const q = (params: Record<string, string | number | boolean | undefined>) => {
  const sp = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== '') sp.set(k, String(v))
  }
  const s = sp.toString()
  return s ? `?${s}` : ''
}

export const api = {
  health: () => request<{ status: string; graph: string; version?: string }>('/health'),

  listTenants: () => request<Tenant[]>('/tenants'),
  createTenant: (t: { tenant_id: string; name: string; ontology_domain: string }) =>
    request<Tenant>('/tenants', { method: 'POST', body: JSON.stringify(t) }),

  // Withheld matters arrive alongside the readable ones rather than as a count, so the
  // page can name them. Normalised because the endpoint previously returned a bare
  // array and a mid-deploy client should not blank the page.
  listMatters: async (tenant: string): Promise<MattersResponse> => {
    const r = await request<MattersResponse | Matter[]>(`/tenants/${tenant}/matters`)
    if (Array.isArray(r)) return { matters: r, withheld: [] }
    return { matters: r.matters ?? [], withheld: r.withheld ?? [] }
  },

  listSources: (tenant: string) => request<Source[]>(`/tenants/${tenant}/sources`),

  /**
   * Invite a user. Cognito emails the temporary password, so no credential is ever
   * returned here. The tenant is taken from the caller's token, not sent.
   */
  createUser: (tenant: string, email: string, isAdmin = false) =>
    request<CreatedUser>(`/tenants/${tenant}/users`, {
      method: 'POST',
      body: JSON.stringify({ email, is_admin: isAdmin }),
    }),

  /** `mine` is users this admin created; `tenant` is everyone in the firm. */
  listUsers: (tenant: string, scope: 'mine' | 'tenant' = 'mine') =>
    request<{ scope: string; users: TenantUser[] }>(`/tenants/${tenant}/users?scope=${scope}`),
  createSource: (tenant: string, s: Partial<Source>) =>
    request<Source>(`/tenants/${tenant}/sources`, { method: 'POST', body: JSON.stringify(s) }),
  listTables: (tenant: string) => request<TableSummary[]>(`/tenants/${tenant}/tables`),
  getTable: (tenant: string, fullName: string) =>
    request<TableDetail>(`/tenants/${tenant}/tables/${encodeURIComponent(fullName)}`),

  listDocuments: (tenant: string, matterId?: string) =>
    request<DocumentSummary[]>(`/tenants/${tenant}/documents${q({ matter_id: matterId })}`),
  getDocument: (tenant: string, id: string) =>
    request<DocumentDetail>(`/tenants/${tenant}/documents/${id}`),
  // Presigned and short-lived, and issued only after matter access is checked. Never
  // cached: a held URL either expires or outlives the permission that granted it.
  documentDownload: (tenant: string, documentId: string) =>
    request<DocumentDownload>(`/tenants/${tenant}/documents/${documentId}/download`),
  uploadDocument: async (
    tenant: string,
    file: File,
    matterId?: string,
  ): Promise<DocumentSummary> => {
    const form = new FormData()
    form.append('file', file)
    if (matterId) form.append('matter_id', matterId)
    const headers: Record<string, string> = {}
    if (isAuthEnabled()) {
      const token = getAccessToken()
      if (token) headers['Authorization'] = `Bearer ${token}`
    }
    // Multipart: no Content-Type header — the browser supplies the boundary.
    const res = await fetch(`${BASE}/tenants/${tenant}/documents`, {
      method: 'POST',
      body: form,
      headers,
    })
    if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`)
    return res.json()
  },

  /**
   * Upload straight to S3 with a presigned POST, then let the notification start
   * ingestion.
   *
   * This is the path for anything non-trivial. `uploadDocument` above transcribes inside
   * the request, so a large document exceeds CloudFront's 60s origin timeout and returns
   * 504 after the write has already happened. Here the bytes never touch the API, so
   * document size and ingest duration stop being an HTTP concern.
   *
   * Resolves once S3 has the object. Ingestion is still running at that point — poll
   * `documentJobs`, or subscribe with `subscribeIngestEvents`.
   */
  presignUpload: (tenant: string, filename: string, mediaType?: string, matterId?: string) =>
    request<UploadTicket>(`/tenants/${tenant}/documents/presign`, {
      method: 'POST',
      body: JSON.stringify({
        filename,
        media_type: mediaType || undefined,
        matter_id: matterId || undefined,
      }),
    }),

  uploadViaPresignedPost: async (
    tenant: string,
    file: File,
    matterId?: string,
    onProgress?: (fraction: number) => void,
  ): Promise<UploadTicket> => {
    const ticket = await api.presignUpload(tenant, file.name, file.type, matterId)

    const form = new FormData()
    // Order matters to S3: every policy field must precede the file itself.
    for (const [k, v] of Object.entries(ticket.fields)) form.append(k, v)
    form.append('file', file)

    // XHR rather than fetch: fetch cannot report upload progress, and a large bundle
    // with no progress bar is indistinguishable from a hung page.
    await new Promise<void>((resolve, reject) => {
      const xhr = new XMLHttpRequest()
      xhr.open('POST', ticket.upload_url)
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total)
      }
      xhr.onload = () =>
        // S3 answers 204 to a presigned POST with no success_action_redirect.
        xhr.status >= 200 && xhr.status < 300
          ? resolve()
          : reject(new Error(`S3 rejected the upload (${xhr.status}): ${xhr.responseText}`))
      xhr.onerror = () => reject(new Error('network error uploading to S3'))
      xhr.send(form)
    })

    return ticket
  },

  documentJobs: (tenant: string, documentId: string) =>
    request<{ document_id: string; jobs: IngestJob[] }>(
      `/tenants/${tenant}/documents/${documentId}/jobs`,
    ),

  jobStatus: (tenant: string, jobId: string) =>
    request<IngestJob>(`/tenants/${tenant}/jobs/${jobId}`),

  /**
   * Live ingest progress. Returns a close function.
   *
   * Best-effort by design: the socket is served by one task, so with more than one task
   * a client may be connected to a task that is not running its ingest. The 30s poll is
   * the correctness guarantee and this only removes the wait, so nothing in the UI may
   * depend on an event arriving.
   */
  subscribeIngestEvents: (
    tenant: string,
    onEvent: (event: IngestEvent) => void,
    onError?: () => void,
  ): (() => void) => {
    const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const token = isAuthEnabled() ? getAccessToken() : ''
    const url =
      `${scheme}://${window.location.host}${BASE}/tenants/${tenant}/ingest/events` +
      `?token=${encodeURIComponent(token || '')}`

    let socket: WebSocket | null = null
    let closed = false
    try {
      socket = new WebSocket(url)
      socket.onmessage = (e) => {
        try {
          onEvent(JSON.parse(e.data))
        } catch {
          // A malformed frame is not worth tearing the connection down for.
        }
      }
      socket.onerror = () => onError?.()
      socket.onclose = () => {
        if (!closed) onError?.()
      }
    } catch {
      onError?.()
    }

    return () => {
      closed = true
      socket?.close()
    }
  },

  listAssertions: (
    tenant: string,
    opts: {
      review_state?: ReviewState
      epistemic_class?: EpistemicClass
      matter_id?: string
      document_id?: string
      predicate?: string
      search?: string
      min_confidence?: number
      as_of?: string
      limit?: number
    } = {},
  ) => request<Assertion[]>(`/tenants/${tenant}/assertions${q(opts)}`),

  approveAssertion: (tenant: string, id: string, note?: string) =>
    request<Assertion>(`/tenants/${tenant}/assertions/${id}/approve`, {
      method: 'POST',
      body: JSON.stringify({ note }),
    }),
  rejectAssertion: (tenant: string, id: string, note?: string) =>
    request<Assertion>(`/tenants/${tenant}/assertions/${id}/reject`, {
      method: 'POST',
      body: JSON.stringify({ note }),
    }),
  getProvenance: (tenant: string, id: string) =>
    request<Provenance>(`/tenants/${tenant}/assertions/${id}/provenance`),

  listMetrics: (tenant: string) => request<Metric[]>(`/tenants/${tenant}/metrics`),
  createMetric: (tenant: string, m: Partial<Metric>) =>
    request<Metric>(`/tenants/${tenant}/metrics`, { method: 'POST', body: JSON.stringify(m) }),
  updateMetric: (tenant: string, id: string, m: Partial<Metric>) =>
    request<Metric>(`/tenants/${tenant}/metrics/${id}`, {
      method: 'PUT',
      body: JSON.stringify(m),
    }),
  setMetricStatus: (tenant: string, id: string, status: Metric['status']) =>
    request<Metric>(`/tenants/${tenant}/metrics/${id}/status`, {
      method: 'POST',
      body: JSON.stringify({ status }),
    }),
  compileMetric: (tenant: string, id: string) =>
    request<{ metric_id: string; sql: string }>(`/tenants/${tenant}/metrics/${id}/compile`, {
      method: 'POST',
    }),

  query: (
    tenant: string,
    body: { question: string; matter_id?: string; as_of?: string; min_confidence?: number },
  ) => request<QueryResult>(`/tenants/${tenant}/query`, { method: 'POST', body: JSON.stringify(body) }),

  neighbourhood: (
    tenant: string,
    opts: {
      node_id?: string
      matter_id?: string
      depth?: number
      min_confidence?: number
      as_of?: string
      include_pending?: boolean
      include_suggestions?: boolean
    },
  ) => request<Neighbourhood>(`/tenants/${tenant}/graph/neighbourhood${q(opts)}`),

  ontology: (domain: string) => request<Ontology>(`/ontology/${domain}`),

  dashboard: (tenant: string) => request<DashboardStats>(`/tenants/${tenant}/dashboard`),

  getSettings: (tenant: string) => request<TenantSettings>(`/tenants/${tenant}/settings`),
  updateSettings: (tenant: string, s: Partial<TenantSettings>) =>
    request<TenantSettings>(`/tenants/${tenant}/settings`, {
      method: 'PUT',
      body: JSON.stringify(s),
    }),

  /**
   * Drop derived data. S3 and Glue are untouched, so everything except metrics is
   * rebuildable by `replay` and `scanSources`. `confirm_metric_loss` is required when
   * `metrics` is set, and the server answers 400 without it.
   */
  resetDerived: (tenant: string, scope: ResetScope, confirmMetricLoss = false) =>
    request<ResetReport>(`/tenants/${tenant}/admin/reset`, {
      method: 'POST',
      body: JSON.stringify({ ...scope, confirm_metric_loss: confirmMetricLoss }),
    }),

  replay: (tenant: string, runModelExtraction = false) =>
    request<ReplayReport>(
      `/tenants/${tenant}/admin/replay${q({ run_model_extraction: runModelExtraction })}`,
      { method: 'POST' },
    ),

  loadSampleData: (tenant: string, runModelExtraction = false) =>
    request<SampleDataReport>(
      `/tenants/${tenant}/admin/sample-data${q({ run_model_extraction: runModelExtraction })}`,
      { method: 'POST' },
    ),

  scanSources: (tenant: string) =>
    request<ScanReport>(`/tenants/${tenant}/sources/scan`, {
      method: 'POST',
      body: JSON.stringify({}),
    }),

  listAccessUsers: (tenant: string) => request<DirectoryUser[]>(`/tenants/${tenant}/access/users`),
  getUserAccess: (tenant: string, userId: string) =>
    request<UserAccess>(`/tenants/${tenant}/access/users/${encodeURIComponent(userId)}`),
  getMatterAccess: (tenant: string, matterId: string) =>
    request<MatterAccessDetail>(`/tenants/${tenant}/access/matters/${encodeURIComponent(matterId)}`),

  assign: (tenant: string, body: { user_id: string; matter_id: string; role: string }) =>
    request<MatterAssignment>(`/tenants/${tenant}/access/assignments`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  // DELETE with a body: unassigning is recorded, not erased, and the reason travels
  // with the request so the audit row is written in one hop.
  unassign: (tenant: string, body: { user_id: string; matter_id: string; reason: string }) =>
    request<{ ok: boolean }>(`/tenants/${tenant}/access/assignments`, {
      method: 'DELETE',
      body: JSON.stringify(body),
    }),

  screen: (
    tenant: string,
    body: { user_id: string; matter_id: string; reason: string; contact?: string },
  ) =>
    request<MatterScreen>(`/tenants/${tenant}/access/screens`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  liftScreen: (tenant: string, body: { user_id: string; matter_id: string; reason: string }) =>
    request<{ ok: boolean }>(`/tenants/${tenant}/access/screens`, {
      method: 'DELETE',
      body: JSON.stringify(body),
    }),

  accessAudit: (tenant: string, opts: { matter_id?: string; user_id?: string } = {}) =>
    request<AccessEvent[]>(`/tenants/${tenant}/access/audit${q(opts)}`),
}
