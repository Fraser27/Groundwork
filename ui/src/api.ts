/**
 * Typed client for the Groundwork API.
 *
 * Every tenant-scoped path takes the tenant id explicitly. The server derives the
 * real tenant from the verified JWT and will reject a mismatch — the value here is
 * only for routing, never for authorisation.
 */

import { ensureAccessToken, hasPendingAuthCode, isAuthEnabled } from './auth'

const BASE = '/api'

/** What an expired session looks like on a socket, where there is no status code to carry it. */
const SESSION_EXPIRED = 'your session has expired, reload the page to sign in again'

async function request<T>(path: string, opts?: RequestInit): Promise<T> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }

  if (isAuthEnabled()) {
    const token = await ensureAccessToken()
    if (token) headers['Authorization'] = `Bearer ${token}`
  }

  const res = await fetch(`${BASE}${path}`, { headers, ...opts })

  if (res.status === 401) {
    // A 401 while a code is being exchanged is not an expired session, it is a request that ran
    // too early. Clearing storage and redirecting to `/` would drop the `?code=` this page load
    // exists to redeem and the PKCE verifier that redeems it, so the sign-in could never finish
    // and the next attempt would land here again. Report it and let the caller degrade.
    if (isAuthEnabled() && !hasPendingAuthCode()) {
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
  /**
   * What the extractor claimed, before the cap or an approval rescaled `confidence`.
   *
   * Null for a fact that never self-reported: a catalog scan and a reviewer's correction
   * assert their confidence rather than estimating it. Populated by `_to_out` in
   * `routes_review.py`, so it is on every assertion the queue and provenance panel read.
   */
  raw_confidence?: number | null
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
  /**
   * Existing entity ids this claim's endpoints may be another spelling of.
   *
   * Advisory, computed when the claim was staged. Shown while it is still PENDING because that is
   * the one moment fixing it is free: once a conclusion rests on the fact, changing the id means
   * a cascade. Empty means either no collision or that nothing was wired to look.
   */
  near_duplicates?: string[]
  /**
   * Conclusions the pack's rules drew because this approval made its premises usable.
   *
   * Nearly always 0 — most facts are not the last premise of anything. Non-zero means approving
   * this one completed a derivation, and the conclusion is in the queue awaiting review.
   */
  inferred?: number
}

/**
 * One premise of an inference, as the endpoint returns it: flat, not nested.
 *
 * `visible: false` means the premise exists but this reader may not see it, so only its id
 * is sent. A gap in a proof tree has to be shown as a gap; silently dropping the row would
 * make a partially visible derivation look fully accounted for.
 */
export interface Premise extends Assertion {
  visible: boolean
}

/**
 * The citation for a document-sourced fact: which file, which page, which words.
 *
 * `download_url` is minted per request and expires, so it is never cached — a stale
 * one is refetched rather than reused.
 */
export interface PageCitation {
  document_id: string
  filename?: string | null
  page?: number | null
  quote?: string | null
  chunk_id?: string | null
  download_url?: string | null
  expires_at?: string | null
  /** Why there is no link. Provenance without one is degraded, not broken. */
  link_unavailable?: string | null
}

export interface Provenance {
  assertion: Assertion
  /**
   * The premises an INFERRED fact rests on, in the order the rule bound them.
   *
   * One level deep. A premise that is itself inferred carries its own `premises` ids, and
   * following those means another fetch; the panel links rather than recursing.
   */
  premises?: Premise[]
  /** Populated for document-sourced assertions. Named `document` by the endpoint. */
  document?: PageCitation | null
  rule_id?: string | null
  rule_version?: string | null
  is_current?: boolean
  /** Plain-language account of where the fact came from. */
  explanation?: string
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

/** A reviewer's correction: their version, and the model's now closed. */
export interface CorrectionResult {
  corrected: Assertion
  superseded: Assertion
  note: string
}

/** What a wipe withdrew. Soft: these facts are closed, not deleted. */
export interface WipeReport {
  scope: 'document' | 'matter'
  target: string
  assertions_superseded: number
  vectors_deleted: number
  jobs_dropped: number
  documents: string[]
  errors: string[]
  at: string
  note: string
}

/** One row of the graph audit log: who changed what the system believes. */
export interface GraphAuditEvent {
  at: string
  actor: string
  /** Resolved from the directory at read time, so null when the sub is not a current user. */
  actor_email?: string | null
  action: 'SUPERSEDE' | 'WIPE_DOCUMENT' | 'WIPE_MATTER'
  document_id?: string | null
  matter_id?: string | null
  assertion_ids: string[]
  affected: number
  reason?: string | null
  detail: Record<string, unknown>
}

/** One row of the question log: what was asked, which tier answered, on what basis. */
export interface QueryAuditEvent {
  at: string
  actor: string
  actor_email?: string | null
  question: string
  tier: number
  tier_name: string
  governed: boolean
  /** A tier can be reached, find nothing, and the empty answer is still recorded. */
  answered: boolean
  sql?: string | null
  assertion_ids: string[]
  document_ids: string[]
  facts_used: number
  ids_truncated: boolean
  /** Which way in produced this row. Absent on a row written before the field existed, which was
   *  always the Ask page. */
  surface?: string
  /**
   * What the answer may be called, in the words the trace shows: `governed`, `verbatim + inferred`,
   * `... + written by agent`.
   *
   * Read this in preference to `governed`, which cannot express a composed answer: a run mixing a
   * compiled metric with a model's reading is neither, and the boolean has to pick one.
   */
  governance?: string
  /** `governance` when there is one, else the boolean rendered as a word. Always safe to show. */
  basis?: string
  /** Retrieval runs only. Ties the row to the transcript that produced it. */
  run_id?: string | null
  /** Retrieval runs only: the tool ladder in call order, repeats included. */
  tools_called?: string[]
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
  created_by?: string | null
  /** Set when the tenant was deleted. The record survives so a reused id reads as reused. */
  deleted_at?: string | null
  deleted_by?: string | null
  is_live?: boolean
}

/** What a tenant delete removed, and what it could not. `complete` false means run it again. */
export interface TenantDeleteReport {
  tenant_id: string
  complete: boolean
  users_deleted: number
  groups_deleted: number
  assertions_dropped: number
  vectors_dropped: number
  jobs_dropped: number
  grants_dropped: number
  graph_audit_dropped: number
  query_audit_dropped: number
  documents_erased: number
  settings_deleted: boolean
  tombstoned: boolean
  errors: string[]
  note: string
}

/**
 * A matter record, plus the count the route derives.
 *
 * This is the whole of it. `client`, `status`, `opened_at` and a `counts` object were all declared
 * here and none has ever been sent: the Matters table read `m.counts?.assertions`, so a matter with
 * a document and ten approved facts displayed 0 facts and a dash in every other column, and
 * `status` rendered as an empty tag. Counts are derived in the page from the documents and
 * assertions it already loads.
 *
 * `name` is optional because a matter can also be *inferred* from assertions referring to a
 * reference no record names — after a reset, say. Those rows carry the reference as the name.
 */
export interface Matter {
  matter_id: string
  name?: string
  assertion_count?: number
  created_at?: string | null
  created_by?: string | null
  updated_at?: string | null
  updated_by?: string | null
  /** Ethical wall: this matter is walled off from the calling user. */
  walled?: boolean
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

/** What a bulk link moved. Assertion ids are unchanged: a matter is not part of a fact's identity. */
export interface LinkReport {
  matter_id: string
  documents: string[]
  assertions_relinked: number
  previous_matters: Record<string, string | null>
  errors: string[]
  at: string
  note: string
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

/** A Glue database as offered for selection, before anything is read into the graph. */
export interface GlueDatabase {
  name: string
  /** Null when it could not be counted, which is not the same as empty. */
  table_count: number | null
  /** Already represented in the graph, so a scan would replace rather than add. */
  scanned?: boolean
  error?: string
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

/**
 * Where a description came from, which is not decoration.
 *
 * A model's guess and a colleague's correction are different claims about what a column means, and
 * the first is the one worth double-checking before trusting a generated query.
 */
export type DescriptionSource = '' | 'glue' | 'human' | 'model'

/** A description a model proposed and nobody has reviewed. Not in use until approved. */
export interface PendingDescription {
  assertion_id: string
  object_id: string
  method: string
  confidence: number
}

export interface Column {
  name: string
  data_type: string
  description?: string
  description_source?: DescriptionSource
  pending_description?: PendingDescription | null
  is_partition: boolean
  is_primary_key: boolean
}

/** A governed metric whose SQL reads this table. */
export interface TableMetric {
  metric_id: string
  name: string
  definition: string
}

export interface TableDetail extends TableSummary {
  columns: Column[]
  /** The catalog scan that declared this table. */
  method: string
  scanned_at?: string | null
  description_source?: DescriptionSource
  pending_description?: PendingDescription | null
  /** How many proposals across this table and its columns are waiting for review. */
  pending_enrichment?: number
  /** Metrics compiled against this table. Always sent, so absent means unanswered, not none. */
  metrics?: TableMetric[]
  /** Other wording that resolves to this table. */
  synonyms?: string[]
  topics?: string[]
}

/** Progress of a catalog enrichment run. In-memory server side, so it resets on a deploy. */
export interface EnrichmentStatus {
  state: 'none' | 'running' | 'done' | 'failed'
  source_id?: string
  tables_total?: number
  tables_done?: number
  staged?: number
  nodes_written?: number
  errors?: string[]
  started_at?: string
  finished_at?: string | null
  note?: string
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
  // Between review and live, and it was missing: a document settled by an approval sits here,
  // so the pipeline showed every step incomplete for a document that had finished.
  'APPROVED',
  'LIVE',
] as const

/**
 * Mirrors `JobState` in src/documents/models.py, and it had drifted in both directions:
 * `GRAPH_FAILED` was declared and does not exist in Python, while `APPROVED`, `STAGE_FAILED`,
 * `PROMOTE_FAILED`, `CHUNK_FAILED` and `EXTRACT_FAILED` are all sent and were absent. A state
 * the API sends and this union omits falls through to a raw string with no label and no help
 * text, which is degradation rather than a crash and therefore easy to miss.
 */
export type IngestState =
  | (typeof INGEST_STATES)[number]
  | 'FETCH_FAILED'
  | 'PARSE_FAILED'
  | 'CHUNK_FAILED'
  | 'EXTRACT_FAILED'
  | 'EMBED_FAILED'
  | 'STAGE_FAILED'
  | 'PROMOTE_FAILED'

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
  /**
   * The rule pass that ran once this document's facts went live, or null for a document ingested
   * before this was recorded.
   *
   * Worth as much as the facts above it. Those say what was read out of the page; this says what
   * the pack could and could not conclude once they joined what was already in the graph, and a
   * rule that drew nothing because a premise matched nothing is the case that otherwise looks
   * identical to a clean check.
   */
  reasoning?: ReasonerReport | null
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

/** A table the compiled query joins to the source table, and the equality it joins on. */
export interface MetricJoin {
  table: string
  source_column: string
  target_column: string
  join_type: string
}

/** simple aggregates one table; derived composes other simple metrics as CTEs. */
export type MetricType = 'simple' | 'derived'

export interface Metric {
  metric_id: string
  name: string
  definition: string
  expression: string
  source_table: string
  source_id?: string
  /** Optional only because an older API may not send it. A write always states it. */
  type?: MetricType
  joins?: MetricJoin[]
  /** For a derived metric, the simple metrics it composes, by name or id. */
  base_metrics?: string[]
  /** Result column to graph node label, declared rather than inferred from values. */
  entity_columns?: Record<string, string>
  grain: string[]
  time_grain_column?: string | null
  time_grains: string[]
  aggregation: 'additive' | 'semi_additive' | 'non_additive'
  parameters: MetricParameter[]
  filters: string[]
  synonyms: string[]
  status: 'draft' | 'approved' | 'deprecated'
  version: number
  /** What kind of quantity this is, e.g. currency or percent. Presentation only. */
  value_type?: string
  /**
   * The unit the figure is in, e.g. GBP or hours.
   *
   * Not decoration. The compiler warns when a derived metric composes base metrics with
   * different units, and it reads that off this field, so an unset unit means the check on
   * combining a currency with a count silently passes.
   */
  unit?: string
  /** A display pattern, e.g. £#,##0. Never applied to the stored figure. */
  format?: string
  owner?: string | null
  updated_by?: string | null
  updated_at?: string | null
}

/**
 * Every field a write carries, none of them optional.
 *
 * The edit form fetches a definition and sends it back, so a field it has no editor for still
 * has to be returned: leaving one out rewrites a governed definition on save. Required here so
 * that omitting one is a type error rather than a silent loss.
 */
export type MetricWrite = Required<
  Pick<
    Metric,
    | 'metric_id'
    | 'name'
    | 'definition'
    | 'expression'
    | 'source_table'
    | 'type'
    | 'joins'
    | 'base_metrics'
    | 'entity_columns'
    | 'grain'
    | 'time_grain_column'
    | 'time_grains'
    | 'aggregation'
    | 'parameters'
    | 'filters'
    | 'synonyms'
    | 'value_type'
    | 'unit'
    | 'format'
    | 'owner'
  >
>

/**
 * A compiled metric definition, saved or not.
 *
 * `warnings` is the part that is easy to drop and expensive to drop: fan-out inflation over a
 * join, a semi-additive refusal and a unit mismatch are all reported here and nowhere else.
 */
export interface CompiledMetric {
  metric_id: string
  sql: string
  source_table?: string
  warnings: string[]
  note?: string
}

/** A saved definition with the SQL it compiled to, and whatever the compiler flagged on the way. */
export type SavedMetric = Metric & { sql: string; warnings: string[]; note: string }

// ── Query resolution ─────────────────────────────────────────────────────────

/**
 * Which tier of the read path answered. Tier 1 is the only one with no model anywhere in it.
 *
 * 4 was a retired tier and is deliberately not in the union: nothing can answer that way now.
 * The question log still holds rows naming it, so anything reading a recorded tier takes a
 * `number` and goes through `tierMeta`.
 */
export type ResolutionTier = 1 | 2 | 3

/**
 * A document span the answer rests on. Only tier 3 sends any.
 *
 * Deliberately this narrow: it is built from the retrieved passage, so there is no assertion
 * behind it and nothing to label it with. It previously also declared `assertion_id`, `label`,
 * `epistemic_class` and `confidence`, none of which the resolver has ever sent — reading
 * `epistemic_class` off one of these throws before it can render.
 */
export interface QueryCitation {
  document_id: string
  page?: number | null
  /** Debug only. Offsets into the extracted text, not into the file. */
  char_start?: number | null
  char_end?: number | null
}

/** One assertion an answer rests on, with the terms that matched it. */
export interface QueryHit {
  assertion_id: string
  subject_id: string
  predicate: string
  object_id: string
  epistemic_class: EpistemicClass
  confidence: number
  matter_id?: string | null
  matched_on: string[]
  /** Hops from a cited passage, 1-based. Null on a tier 2 match, explained by `matched_on`. */
  hops?: number | null
  source: SourceLocator
}

/**
 * One hop of a chain, with the direction the chain reads it in.
 *
 * `subject_id` and `object_id` stay as the assertion was written, so `reversed` is what says the
 * chain traversed it the other way. Render the direction from that flag: printing
 * `A REPRESENTS B` for a chain that arrived at A from B states the relationship backwards.
 */
export interface QueryPathStep {
  assertion_id: string
  subject_id: string
  predicate: string
  object_id: string
  epistemic_class: EpistemicClass
  confidence: number
  reversed: boolean
}

/**
 * A multi-hop connection the graph assembled, two hops or more.
 *
 * Every step is an edge already in the lane's flat fact list — see `src/query/paths.py`. The value
 * is that the join is the graph's rather than a reader's: the alternative is a person or a model
 * spotting that three rows out of two hundred share an entity, which is the one step in the chain
 * nobody can audit.
 *
 * `confidence` is the weakest hop, because a chain is an argument and an argument is as good as its
 * worst step.
 */
export interface QueryPath {
  nodes: string[]
  steps: QueryPathStep[]
  confidence: number
}

/** A passage the vector search returned, before the graph expanded around it. */
export interface QueryPassage {
  document_id: string
  filename?: string | null
  page?: number | null
  text?: string | null
  score?: number | null
  char_start?: number | null
  char_end?: number | null
}

export interface QueryRows {
  columns: string[]
  rows: (string | number | null)[][]
}

/**
 * Shaped by whichever tier answered, so a union rather than a string.
 *
 * It was declared `string` and rendered straight as a React child, which blanked the whole page
 * the moment tier 2 answered — the graph tier puts a list of assertions here, not prose. `tsc` was
 * clean throughout, because a type is a claim about the response and nothing checks it against one.
 */
export type QueryAnswer =
  | QueryRows
  | QueryHit[]
  | {
      passages: QueryPassage[]
      related: QueryHit[]
      paths?: QueryPath[]
      generated?: GeneratedSQLResult
    }
  | GraphFirstAnswer
  | null

/**
 * Tier 2's answer: what the graph matched, and what querying its landing zone returned.
 *
 * A variant rather than a reuse of the tier 3 shape. It is `facts`, not `related`, and the
 * difference is the direction: tier 3 relates the graph to passages it already had, tier 2 matched
 * the graph first and the passages came from where those facts did. Declared because it was not,
 * and `asHits` silently returned nothing for a whole tier.
 */
export interface GraphFirstAnswer {
  facts: QueryHit[]
  passages: QueryPassage[]
  tables: CatalogSchemaRef[]
  /**
   * How far the traversal reached: any of `facts`, `documents`, `tables`.
   *
   * Load-bearing, not diagnostic. An empty passage list means either that no document was reached
   * or that the documents held nothing matching, and only this distinguishes them — so the trace
   * reads it rather than reporting both as a lane that found nothing.
   */
  landed: string[]
  /** Multi-hop connections between the facts, assembled by `src/query/paths.py`. */
  paths?: QueryPath[]
  generated?: GeneratedSQLResult
}

/**
 * A query an AI wrote, from tier 3's SQL lane. Only present when one was written and run.
 *
 * `rows` and `error` are both optional and exactly one is set: the firewall validates tables, not
 * columns, so a hallucinated column errors at Athena. Show the error beside the SQL — rendering it
 * as no rows would read as "no data", which is the opposite of what happened.
 */
export interface GeneratedSQLResult {
  sql: string
  /** What the prompt was offered, which is also what the firewall allowed. Anything else was
   *  unexecutable, not merely discouraged. */
  tables_offered: string[]
  rows?: QueryRows | null
  error?: string | null
  /** `blocked` when the firewall refused it, otherwise Athena's own code. */
  error_code?: string | null
}

/**
 * Something the deterministic veto refused to let through, exactly as `Block.to_dict` sends it.
 *
 * `rule` is `ethical_screen` for a recorded wall and the ontology rule's name for an inference
 * that fired. `contact` is only ever set for a screen: a rule block is not somebody's decision
 * to explain.
 */
export interface QueryBlock {
  subject: string
  reason: string
  rule: string
  matter_id?: string | null
  contact?: string | null
  /** Signed-off facts combined to reach this refusal. 0 for an ethical screen, which is an
   *  instruction rather than a derivation. Higher means less likely anyone found it unaided. */
  premise_count?: number
  /**
   * Whether this finding suppressed the evidence or only reported itself.
   *
   * Absent reads as `withhold`, which is the pack default. A conflict notifies: whether it is a
   * real conflict is a lawyer's judgement, and they cannot make it from evidence they were never
   * shown. An ethical screen always withholds — it is the one true prohibition.
   */
  effect?: 'withhold' | 'notify'
}

/** Which index a routing hit came from. `passages` is the document-chunk index. */
export type RouterLayerKind = 'metric' | 'entity' | 'table' | 'passages'

/**
 * One thing the router matched, with the cosine that matched it.
 *
 * `detail` is free-form per kind — a metric sends its source table, a table its columns — so it
 * is read defensively rather than declared per kind. Nothing here is a claim about relevance:
 * cosine similarity is not calibrated across questions.
 */
export interface RouterItem {
  kind: RouterLayerKind
  item_id: string
  label: string
  similarity: number
  detail?: Record<string, unknown> | null
}

export interface RouterLayer {
  kind: RouterLayerKind
  /** The tier a reader thinks of this layer as. `tiers` is authoritative. */
  tier: number | null
  /** Every tier this layer justifies running. A layer can justify more than one. */
  tiers?: number[]
  score: number
  /** Before the boost, so the trace can show what the boost did rather than only its effect. */
  raw_score: number
  boost: number
  hit_count: number
  /** Score as a fraction of the best-scoring layer. This is what the margin is measured against. */
  relative: number
  selected: boolean
  reason: string
  items?: RouterItem[]
}

/**
 * How the tiers were chosen, or the record that they were not.
 *
 * Absent when the router is disabled or the deployment has no vector store, so every read of
 * this is guarded. `degraded` means the router ran every tier rather than choosing, which is a
 * materially different story from a router that chose, and the UI says so rather than drawing
 * an empty diagram.
 */
export interface RouterTrace {
  enabled: boolean
  degraded: boolean
  /**
   * Whether the caller acted on this decision or only recorded it.
   *
   * False on `/query/compose`, which runs every permitted lane whatever the scores say — so the
   * step must not label a tier "not selected" while the step below it shows what that tier
   * returned. Optional because an older response omits it; absent reads as true, which is what
   * `/query` has always done.
   */
  applied?: boolean
  reason?: string | null
  margin: number
  min_similarity: number
  metric_boost: number
  best_score: number
  layers?: RouterLayer[]
  tiers_selected?: number[]
  /** Tier number as a string, because these are JSON object keys. */
  tiers_dropped?: Record<string, string>
  /**
   * Tiers the tenant does not permit, sent as data rather than left to be read out of the
   * wording of `tiers_dropped`. A tier refused by policy was never even searched, which is a
   * different fact from one that was measured and out-scored.
   */
  tiers_forbidden?: string[]
}

/**
 * What the ethical wall did, refusals and clearances alike.
 *
 * `subjects_cleared` is here because the wall approving something is as much a fact as it
 * refusing something: a step that is only ever visible when it blocks reads as an exception
 * rather than as a gate everything passed through.
 */
export interface GateTrace {
  seeds_considered: number
  subjects_cleared: number
  /**
   * Subjects a finding named, whether or not it withheld anything.
   *
   * Distinct from both other counters. A notify finding suppresses nothing, so "4 of 5 cleared, 0
   * items withheld" beside a live conflict would be true and read as reassurance.
   */
  subjects_flagged?: number
  items_withheld: number
  /**
   * Why the rule-block half did not run, or null when it did.
   *
   * Non-null means only the ethical screens were applied: the graph was not checked for
   * conflicts, so "nothing refused" here is not a clearance. The wall fails open rather than
   * refusing every answer on a transient graph error, which is only defensible if it says so.
   */
  degraded?: string | null
  blocks?: QueryBlock[]
  /**
   * Entities named by a blocking fact nobody has reviewed yet.
   *
   * Not refusals — an unreviewed derivation may not withhold evidence. But "nothing refused"
   * over a graph holding a conflict about this party in the review queue is a true sentence that
   * reads as a false one, so the reader is shown both.
   */
  awaiting_review?: string[]
}

export interface QueryResult {
  tier: ResolutionTier
  tier_name: string
  /** True for all three tiers today. Read rather than assumed: it is the server's claim, not ours. */
  governed: boolean
  /**
   * Whether the same question would reach the same answer with no model involved.
   *
   * A narrower claim than `governed`, and it can be false while that is true: a tier-1 metric no
   * question word matched was chosen by similarity, so the SQL is still compiled from an approved
   * definition while the choice of definition is not reproducible. Optional because an older
   * response omits it; absent reads as true, which is what tier 1 always was.
   */
  deterministic_selection?: boolean
  /** Tier 1 only. Which metric, and how it was reached. */
  metric_selection?: MetricSelection | null
  explanation: string
  answer: QueryAnswer
  /** Tier 1 only, and compiled from a metric definition rather than generated. */
  sql?: string | null
  citations: QueryCitation[]
  assertions_used: string[]
  tiers_attempted: ResolutionTier[]
  warnings: string[]
  /** Always an array from the API. Optional here only so a cached older response cannot crash
   *  the page — the read path guards on `?? []` rather than trusting this. */
  blocks?: QueryBlock[]
  /** Genuinely absent where the router is off or there is no vector store. Never asserted. */
  router?: RouterTrace | null
  gate?: GateTrace | null
  /**
   * The floor the server actually applied, which is not always the one asked for.
   *
   * A request may raise the tenant's floor and never lower it, so a slider set below it is
   * ignored. Read this rather than the local value: the page used to report "nothing cleared the
   * trust floor of 0.85" from its own state while the field was being dropped in transit, naming
   * a number no read had ever used.
   */
  min_confidence?: number
}

// ── Composed answers ─────────────────────────────────────────────────────────

/** Where one part of a composed answer came from. Mirrors src/query/planner.py :: Lane. */
export type Lane = 'metric' | 'graph' | 'passages' | 'catalog' | 'sql'

/** How much of a part a model wrote. Separate from the lane: retrieval can be fuzzy and the
 *  text it returned still exact. Mirrors planner.py :: Provenance.
 *
 *  `model_selected` is a compiled metric whose *definition* a model chose — no question word
 *  matched, so similarity picked between approved metrics. The figure is exact; which figure was
 *  computed is not reproducible, which is why it is neither 'deterministic' nor 'inferred'.
 *
 *  `model_written` is a model choosing the arithmetic itself, not choosing between approved ones.
 *  Nothing approved the query, so a part carrying it is never governed. */
export type PartProvenance =
  | 'deterministic'
  | 'model_selected'
  | 'verbatim'
  | 'inferred'
  | 'model_written'

/** How tier 1 reached its metric. Mirrors metric_matcher.py :: selection_of. */
export interface MetricSelection {
  metric_id: string
  selected_by: 'keyword' | 'router'
  /** False means an embedding narrowed the choice, so the same question worded differently could
   *  reach a different metric. The SQL is compiled from the approved definition either way. */
  deterministic: boolean
  similarity?: number | null
  matched_on?: string[]
  note?: string
}

/** Schema the catalog lane offered. Columns only — rows never leave the warehouse. */
export interface CatalogSchemaRef {
  full_name: string
  description?: string | null
  columns?: string[]
}

/**
 * One lane's contribution, never merged with another's.
 *
 * `content` is shaped by the lane — rows for a metric, hits for the graph, passages for
 * documents, schema for the catalog — so it is narrowed at the point of use rather than
 * declared as any one of them. It is legitimately null for a metric part when no executor
 * is wired: the SQL is the reviewable artefact and tier 1 stops there.
 */
export interface AnswerPart {
  lane: Lane
  provenance: PartProvenance
  tier: number
  content: unknown
  sql?: string | null
  citations?: QueryCitation[]
  assertion_ids?: string[]
  /** Graph lane only: multi-hop connections between the facts in `content`. Absent on a response
   *  from before the field existed, and empty when nothing in the lane chained up. */
  paths?: QueryPath[]
  confidence?: number | null
  /** Metric lane only. */
  metric_selection?: MetricSelection | null
  /**
   * Why this part has no content, when the reason is a failure rather than absence.
   *
   * SQL lane mainly: the firewall validates tables, not columns, so a hallucinated column reaches
   * Athena and errors. Render it — a part with an error and `content: null` is not an empty result,
   * and showing it as one would read as "no data".
   */
  error?: string | null
}

/** Graph first or vector first: the same traversal in opposite directions, and a tenant runs one. */
export type RetrievalDirection = 'graph_first' | 'vector_first' | 'metrics_only'

export interface ComposedResult {
  parts?: AnswerPart[]
  blocks?: QueryBlock[]
  lanes_run?: Lane[]
  lanes_skipped?: Record<string, string>
  /**
   * The direction these lanes were run in. Absent on a response from before the field existed.
   *
   * Needed because a skipped lane has no part and therefore no tier of its own, and labelling it
   * from a hardcoded map put tier 3 on the lanes a graph-first tenant had declined.
   */
  retrieval_direction?: RetrievalDirection
  synthesis?: string | null
  /** Never plain "governed" when a model contributed. Composed server-side. */
  governance: string
  fully_deterministic: boolean
  warnings?: string[]
  note?: string
  router?: RouterTrace | null
  gate?: GateTrace | null
  /** The floor the server applied. See `QueryResult.min_confidence`; the same one-way clamp. */
  min_confidence?: number
}

// ── Retrieval agent ──────────────────────────────────────────────────────────

/** How to render one tool result. From the tool's name server-side, never from the payload. */
export type ResultKind =
  | 'composed'
  | 'resolution'
  | 'assertions'
  | 'provenance'
  | 'graph'
  | 'metrics'
  | 'ontology'
  | 'json'

/**
 * One event from a retrieval run.
 *
 * `seq` is monotonic across the whole run. Two events in the same millisecond cannot be
 * ordered by `at`, and a transcript read from a POST body has no delivery order to rely on.
 */
export interface RetrievalEvent {
  kind: 'run_started' | 'tool_call' | 'tool_result' | 'text' | 'run_finished' | 'run_failed'
  run_id: string
  tenant_id: string
  turn: number
  seq: number
  at: string

  question?: string
  model_id?: string
  max_turns?: number

  tool?: string
  arguments?: Record<string, unknown>
  /** Which cap stopped this call, when one did. The attempt stays in the transcript. */
  cancelled?: string

  result_kind?: ResultKind
  result?: unknown
  is_error?: boolean
  error?: string

  text?: string
  answer?: string
  stop_reason?: string
  turns?: number
  was_capped?: boolean
  /** On `run_finished`: whether ask or compose ran at all. False means the prose has nothing
   *  under it, which is not a capped run and not an error, so nothing else would reveal it. */
  searched?: boolean
  warnings?: string[]
}

/** Which search tool a run may use. `auto` is the agent's own choice between the two. */
export type RetrievalTool = 'auto' | 'ask' | 'compose'

/** A whole run, as the POST route returns it. */
export interface RetrievalRun {
  run_id: string
  events: RetrievalEvent[]
  answer: string
  stop_reason: string
  turns: number
  was_capped: boolean
  max_turns: number
  note: string
}

// ── Graph ────────────────────────────────────────────────────────────────────

/** Exactly what `_node()` sends. Nodes are derived from entity ids, so there is nothing
 *  else on them — no matter, no properties. The matter lives on the assertion. */
export interface GraphNode {
  id: string
  label: string
  /** The id prefix: `party:acme` -> `party`. Lowercase; `EntityDef.slug` is what it matches. */
  type: string
  /** Catalog nodes only: the database the table or column belongs to. Sent rather than parsed out
   *  of the id, because the scanner owns that format and this side owns none of it. */
  database?: string
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
  matter_id?: string | null
}

export interface Neighbourhood {
  nodes: GraphNode[]
  edges: GraphEdge[]
  /** Set when the overview hit its cap. A hairball is not a diagram, so the graph is capped
   *  rather than drawn in full, and a truncated view should say so. */
  truncated?: boolean
  total_edges?: number
  /** Echoed back when a layer was requested. Both counts above then describe that layer alone,
   *  which is the only honest total to compare against once the cap is applied inside one. */
  layer?: string | null
  confidence_floor?: number
}

// ── Ontology ─────────────────────────────────────────────────────────────────

export interface EntityDef {
  id: string
  label: string
  description: string
  help?: string | null
  /** The id prefix form of `id`, lowercase. Compare against `GraphNode.type`. */
  slug?: string
  /** `domain` for facts read out of documents, `catalog` for schema declared by a source. */
  layer?: string
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
  transitive?: boolean
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
  /**
   * The ceiling on any unreviewed model claim. The floor must stay strictly above it, so this is
   * the floor control's lower bound rather than a number anyone edits here.
   */
  model_confidence_cap: number
  block_ungoverned_queries: boolean
  /**
   * Vector router. Sent by the projection now — they were absent, so the toggle read `undefined`,
   * showed off whatever the tenant was configured to do, and the re-read that follows every save
   * reverted it. Still optional, so a deployment running an older API leaves the rest of this
   * page working rather than blanking it.
   */
  router_enabled?: boolean
  router_min_similarity?: number
  router_margin?: number
  router_metric_boost?: number
  /** The tenant's tier cap, so the page can show which routes are permitted at all. */
  allowed_tiers?: number[]
  /**
   * Which way this tenant traverses. Derived from `allowed_tiers` by the API rather than stored,
   * because a second field would be a second default and the two could disagree about which lane
   * is live.
   */
  retrieval_direction?: RetrievalDirection
  extraction_model: string
  synthesis_model: string
  /** Writes tier 3's SQL, and separates a question that asks two things into both. */
  query_model?: string
  /** Drives the Retrieval agent's loop. Separate from the query model, deliberately. */
  retrieval_agent_model?: string
  /** Writes catalog descriptions. A cheap model is the right choice, so it is settable alone. */
  enrichment_model?: string
  embedding_model: string
  /** `note` says what choosing this model trades away, since cheaper is usually the reason. */
  available_models: { id: string; label: string; note?: string }[]
  available_domains: string[]
  /**
   * What this tenant's pack calls the unit work is organised by: Matter for law, Encounter for
   * care, Facility for lending. The scoping key is still `matter_id` everywhere, so this is
   * wording only.
   */
  unit_label?: { singular: string; plural: string }
  /**
   * Questions worth asking of this pack's data, in the order the pack author chose.
   *
   * Was a hardcoded array in `Retrieval.tsx` and `QueryBuilder.tsx` asking about a conflict for a
   * client called Calder, so under any pack but legal the one affordance that shows a new reader
   * what the system can be asked returned nothing. Absent or empty means the page offers none — a pack that
   * declares no question is better served by silence than by a question that answers nothing.
   */
  example_questions?: string[]
}

/**
 * The retrieval knobs the `/settings` projection does not carry.
 *
 * Read and written through `/governance` directly. Routing them through `updateSettings` would
 * patch correctly and then blank the control, because that call re-reads `/settings` and a field
 * absent from the projection comes back `undefined`.
 */
export interface RetrievalGovernance {
  vector_top_k: number
  graph_expand_depth: number
  graph_expand_limit: number
  max_compose_calls: number
}

// ── Entity merge ─────────────────────────────────────────────────────────────

/** Ids sharing a blocking key. Grouping is by name shape, so `party:acme-ltd` and
 *  `party:acme-limited` land together and may still be two companies. */
export interface DuplicateGroup {
  key: string
  entity_ids: string[]
}

/** What a merge did, or would do when `dry_run`. Mirrors src/documents/merge.py :: Merge. */
export interface MergeResult {
  losing_id: string
  winning_id: string
  /** Current assertions naming the losing id at either end. Each is restated about the winner. */
  affected: string[]
  /** Conclusions that fall because an affected premise closes, without naming the merged id
   *  themselves. Often empty even when conclusions do fall: one naming the losing id directly is
   *  restated as part of `affected` instead. */
  cascaded: string[]
  /** Ids of the replacement assertions. Empty for a dry run. */
  rewritten: string[]
  dry_run: boolean
  note: string
}

// ── Reasoning ────────────────────────────────────────────────────────────────

/** One conclusion a rule drew. Mirrors ReasonerReport.to_dict() in src/reasoning/engine.py. */
export interface ReasonerInference {
  assertion_id: string
  rule_id: string
  subject_id: string
  predicate: string
  object_id: string
  confidence: number
  premises: string[]
  matter_id?: string | null
}

/**
 * What a reasoning pass did. Mirrors src/reasoning/engine.py :: ReasonerReport.
 *
 * The three failure maps are separate because they mean different things, and a caller that
 * merges them recreates the ambiguity the report exists to remove: `rules_skipped` could never
 * fire, `rules_starved` ran and its join came up empty, `conclusions_refused` fired and an
 * invariant rejected the conclusion. Zero inferences over 40 facts is not zero rules run.
 */
export interface ReasonerReport {
  inferences: ReasonerInference[]
  count: number
  rules_evaluated: number
  /** Rule id → why it could never fire, e.g. a conclusion outside the vocabulary. */
  rules_skipped: Record<string, string>
  /** Rule id → which premise emptied the join, e.g. "no ADVERSE_TO fact matches
   *  (m:Matter)->(p:Party)". Shown verbatim: the premise is the actionable part. */
  rules_starved: Record<string, string>
  conclusions_refused: string[]
  facts_considered: number
  /** Added by the endpoint. Equal to `count` — conclusions are staged, never written live. */
  staged?: number
  /**
   * False when the pass itself failed, on the ingest report only.
   *
   * Kept separate from a zero count for the reason this codebase keeps restating: "ran and found
   * nothing" is about the facts and "never ran" is about the system, and a reader shown one number
   * cannot tell which they have.
   */
  ran?: boolean
  error?: string
  note?: string
}

// ── Calls ────────────────────────────────────────────────────────────────────

/**
 * Send a parameter with a deliberately empty value.
 *
 * `q()` drops empty strings, which is right for a blank search box and wrong for a filter where
 * empty *means* something. The assertions endpoint filters on `review_state` only when it is
 * truthy, so `review_state=` is how a caller asks for every state -- and omitting it instead lets
 * the server apply its PENDING default, which is what made a matter's approved facts count zero.
 */
export const EMPTY = '\u0000'

const q = (params: Record<string, string | number | boolean | undefined>) => {
  const sp = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) {
    if (v === EMPTY) sp.set(k, '')
    else if (v !== undefined && v !== '') sp.set(k, String(v))
  }
  const s = sp.toString()
  return s ? `?${s}` : ''
}

export const api = {
  health: () =>
    request<{
      status: string
      graph: string
      version?: string
      /** The operator tenant, so the UI knows whether to offer the platform screen. */
      home_tenant?: string
    }>('/health'),

  /**
   * Platform routes. `/platform`, not `/tenants/{tenant}`, because the tenant is an argument
   * here rather than the caller's own — see `routes_tenants.py`. Only an admin of the operator
   * tenant may call them; everyone else gets a 403.
   */
  listTenants: async (): Promise<{ tenants: Tenant[]; home_tenant: string }> =>
    request<{ tenants: Tenant[]; home_tenant: string }>('/platform/tenants'),
  createTenant: (t: {
    tenant_id: string
    admin_email: string
    name?: string
    ontology_domain?: string
  }) => request<Record<string, unknown>>('/platform/tenants', {
    method: 'POST',
    body: JSON.stringify(t),
  }),
  deleteTenant: (tenantId: string) =>
    request<TenantDeleteReport>(`/platform/tenants/${tenantId}/delete`, {
      method: 'POST',
      body: JSON.stringify({
        confirm_tenant_id: tenantId,
        confirm_document_loss: true,
        confirm_audit_loss: true,
      }),
    }),

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

  /** Removes the Cognito account and the cached tenant binding. Not reversible. */
  deleteUser: (tenant: string, email: string) =>
    request<{ email: string; deleted: boolean; note: string }>(
      `/tenants/${tenant}/users/${encodeURIComponent(email)}`,
      { method: 'DELETE' },
    ),
  /** Create a matter, or rename an existing one. It then exists before any document is filed. */
  createMatter: (tenant: string, matterId: string, name: string) =>
    request<Matter>(`/tenants/${tenant}/matters`, {
      method: 'POST',
      body: JSON.stringify({ matter_id: matterId, name }),
    }),

  /** File several documents under a matter, or move them there. Audited. */
  linkDocumentsToMatter: (
    tenant: string,
    matterId: string,
    documentIds: string[],
    reason?: string,
  ) =>
    request<LinkReport>(`/tenants/${tenant}/matters/${encodeURIComponent(matterId)}/documents`, {
      method: 'POST',
      body: JSON.stringify({ document_ids: documentIds, reason: reason || undefined }),
    }),

  createSource: (tenant: string, s: Partial<Source>) =>
    request<Source>(`/tenants/${tenant}/sources`, { method: 'POST', body: JSON.stringify(s) }),
  listTables: (tenant: string) => request<TableSummary[]>(`/tenants/${tenant}/tables`),
  getTable: (tenant: string, fullName: string) =>
    request<TableDetail>(`/tenants/${tenant}/tables/${encodeURIComponent(fullName)}`),

  listDocuments: (tenant: string, matterId?: string) =>
    request<{ documents: DocumentSummary[] } | DocumentSummary[]>(
      `/tenants/${tenant}/documents${q({ matter_id: matterId })}`,
    ).then((r) => (Array.isArray(r) ? r : r.documents ?? [])),
  getDocument: (tenant: string, id: string) =>
    request<DocumentDetail>(`/tenants/${tenant}/documents/${id}`),
  // Presigned and short-lived, and issued only after matter access is checked. Never
  // cached: a held URL either expires or outlives the permission that granted it.
  documentDownload: (tenant: string, documentId: string) =>
    request<DocumentDownload>(`/tenants/${tenant}/documents/${documentId}/download`),
  uploadDocument: async (
    tenant: string,
    file: File,
    matterId: string,
  ): Promise<DocumentSummary> => {
    const form = new FormData()
    form.append('file', file)
    // Always sent, never conditionally. The guard turned an empty matter into an absent field, so
    // the server saw an upload that named no matter rather than one naming a blank -- and it now
    // refuses both, because an unfiled chunk is readable by somebody screened from the matter.
    form.append('matter_id', matterId)
    const headers: Record<string, string> = {}
    if (isAuthEnabled()) {
      const token = await ensureAccessToken()
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
  /** `matterId` is required: the API refuses an upload that names no matter, because facts with
   *  no matter are unattributable and both the Matters and Access pages group by it. */
  presignUpload: (tenant: string, filename: string, mediaType: string | undefined, matterId: string) =>
    request<UploadTicket>(`/tenants/${tenant}/documents/presign`, {
      method: 'POST',
      body: JSON.stringify({
        filename,
        media_type: mediaType || undefined,
        matter_id: matterId,
      }),
    }),

  uploadViaPresignedPost: async (
    tenant: string,
    file: File,
    matterId: string,
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
    let socket: WebSocket | null = null
    let closed = false

    // Opened after the token is in hand, because a browser cannot set headers on the handshake and
    // an absent token would otherwise be sent as `token=` and refused. The poll behind this covers
    // the wait.
    void (async () => {
      const token = isAuthEnabled() ? await ensureAccessToken() : ''
      if (closed) return
      if (isAuthEnabled() && !token) {
        onError?.()
        return
      }

      const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws'
      const url =
        `${scheme}://${window.location.host}${BASE}/tenants/${tenant}/ingest/events` +
        `?token=${encodeURIComponent(token || '')}`
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
    })()

    return () => {
      closed = true
      socket?.close()
    }
  },

  /**
   * Run the agent, receiving each turn as it happens.
   *
   * Unlike `subscribeIngestEvents` this socket *drives* the work: the question goes as the
   * first frame and the run happens inside the handler, so closing the socket stops the run.
   * There is no poll behind it, so a failure is reported rather than silently degraded.
   */
  runRetrieval: (
    tenant: string,
    question: string,
    onEvent: (event: RetrievalEvent) => void,
    onError?: (detail: string) => void,
    /** Which search tool to allow. `auto` leaves it to the agent; the others withhold the one
     *  not chosen, so the choice binds rather than suggests. */
    tool: RetrievalTool = 'auto',
  ): (() => void) => {
    let socket: WebSocket | null = null
    let closed = false
    let sawEvent = false

    void (async () => {
      const token = isAuthEnabled() ? await ensureAccessToken() : ''
      if (closed) return
      if (isAuthEnabled() && !token) {
        // Said here rather than left to the server. With no token the socket used to be opened
        // anyway, as `token=`, and closed with a policy violation carrying no reason -- so an hour
        // after signing in the page reported that the agent had refused the connection, which reads
        // as an outage and sent the reader looking at CloudFront.
        onError?.(SESSION_EXPIRED)
        return
      }

      const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws'
      const url =
        `${scheme}://${window.location.host}${BASE}/tenants/${tenant}/retrieval/events` +
        `?token=${encodeURIComponent(token || '')}`
      try {
        socket = new WebSocket(url)
        socket.onopen = () => socket?.send(JSON.stringify({ question, tool }))
        socket.onmessage = (e) => {
          try {
            sawEvent = true
            onEvent(JSON.parse(e.data))
          } catch {
            // A malformed frame is not worth tearing the connection down for.
          }
        }
        socket.onerror = () => onError?.('the connection to the agent failed')
        socket.onclose = (e) => {
          // A close before any event means the server refused: a bad token, or no MCP endpoint.
          // Its reason is the only explanation the user will get, so it is passed through.
          if (!closed && !sawEvent) onError?.(e.reason || 'the agent refused the connection')
        }
      } catch {
        onError?.('could not open a connection to the agent')
      }
    })()

    return () => {
      closed = true
      socket?.close()
    }
  },

  listAssertions: (
    tenant: string,
    opts: {
      /** Defaults to PENDING *on the server*, so omitting it loads only unreviewed facts --
       *  which is why a matter with ten approved facts counted zero. Pass 'ALL' for every state.
       *
       *  Not '': `q()` strips empty values, so an empty string never reaches the server and the
       *  default applies again. 'ALL' is translated below into the empty value the API wants. */
      review_state?: ReviewState | 'ALL'
      epistemic_class?: EpistemicClass
      matter_id?: string
      document_id?: string
      predicate?: string
      search?: string
      min_confidence?: number
      as_of?: string
      limit?: number
      /** Catalog descriptions are excluded by default; they are reviewed on their own table. */
      include_catalog?: boolean
    } = {},
  ) =>
    // The endpoint returns {assertions, total, confidence_floor}. Typing it as a bare array
    // made `.filter` a runtime crash that the compiler could not see.
    //
    // 'ALL' becomes an explicitly empty `review_state=`, which is how the API says "every state":
    // it filters only when the parameter is truthy. It has to be sent rather than omitted, because
    // omitting it lets the server apply its PENDING default.
    request<{ assertions: Assertion[]; catalog_pending?: number }>(
      `/tenants/${tenant}/assertions${q({
        ...opts,
        review_state: opts.review_state === 'ALL' ? EMPTY : opts.review_state,
      })}`,
    ).then((r) => r.assertions ?? []),

  /**
   * The same read, keeping the envelope.
   *
   * `listAssertions` unwraps to the array because almost every caller wants the rows. The review
   * queue also needs `catalog_pending`, which says how many descriptions are waiting on the Tables
   * page: filtering them out of the queue and then not saying so would look like nothing is there.
   */
  listAssertionsWithCounts: (
    tenant: string,
    opts: { review_state?: ReviewState | 'ALL'; limit?: number; include_catalog?: boolean } = {},
  ) =>
    request<{ assertions: Assertion[]; total: number; catalog_pending?: number }>(
      `/tenants/${tenant}/assertions${q({
        ...opts,
        review_state: opts.review_state === 'ALL' ? EMPTY : opts.review_state,
      })}`,
    ),

  approveAssertion: (tenant: string, id: string, note?: string) =>
    request<Assertion>(`/tenants/${tenant}/assertions/${id}/approve`, {
      method: 'POST',
      body: JSON.stringify({ note }),
    }),
  /**
   * Approve several claims in one request.
   *
   * Not a loop over `approveAssertion`: each approval runs a reasoning pass over the tenant's live
   * facts, so a loop of twenty is twenty passes — and a conflict whose premises were approved
   * together should be drawn from all of them rather than from however many were live partway
   * through. Reports partial success, so nineteen decisions are not lost to a twentieth that a
   * cascade had already rejected.
   */
  approveAssertions: (tenant: string, ids: string[], note?: string) =>
    request<{
      approved: Assertion[]
      failed: { assertion_id: string; reason: string }[]
      inferred: number
    }>(`/tenants/${tenant}/assertions/approve`, {
      method: 'POST',
      body: JSON.stringify({ assertion_ids: ids, note }),
    }),
  rejectAssertion: (tenant: string, id: string, note?: string) =>
    request<Assertion>(`/tenants/${tenant}/assertions/${id}/reject`, {
      method: 'POST',
      body: JSON.stringify({ note }),
    }),
  /**
   * Record what a reviewer says instead. Never an edit: the response carries the reviewer's new
   * DECLARED assertion and the original, now closed, so the caller can show both.
   */
  correctAssertion: (
    tenant: string,
    id: string,
    body: { predicate?: string; subject_id?: string; object_id?: string; reason: string },
  ) =>
    request<CorrectionResult>(`/tenants/${tenant}/assertions/${id}/correct`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  /**
   * Fire the pack's rules over this tenant's live facts and report what happened.
   *
   * Conclusions are staged into the review queue, not written live. The report is worth as much
   * as the conclusions: a conflict check whose premises matched nothing returns zero inferences
   * and so does a clean one, and only `rules_starved` tells the two apart. Admin only.
   */
  runReasoner: (tenant: string) =>
    request<ReasonerReport>(`/tenants/${tenant}/reason`, { method: 'POST' }),

  /** Withdraw everything derived from one document. Soft, audited, and needs a reason. */
  wipeDocument: (tenant: string, documentId: string, reason: string) =>
    request<WipeReport>(`/tenants/${tenant}/documents/${documentId}/wipe`, {
      method: 'POST',
      body: JSON.stringify({ reason }),
    }),

  /** The same for every document on a matter. */
  wipeMatter: (tenant: string, matterId: string, reason: string) =>
    request<WipeReport>(`/tenants/${tenant}/matters/${encodeURIComponent(matterId)}/wipe`, {
      method: 'POST',
      body: JSON.stringify({ reason }),
    }),

  /** Who changed what the system believes, newest first. */
  graphAudit: (tenant: string, limit = 100) =>
    request<{ events: GraphAuditEvent[]; count: number; note: string }>(
      `/tenants/${tenant}/audit/graph?limit=${limit}`,
    ).then((r) => r.events ?? []),

  /**
   * What was asked and on what basis, newest first.
   *
   * `assertionId` inverts it: which questions rested on one fact. `scanned` is how many rows the
   * server read, and it is returned rather than dropped because the answer is exact only within
   * that window -- there is no index on a citation.
   */
  questionAudit: (tenant: string, opts?: { limit?: number; assertionId?: string }) =>
    request<{
      questions: QueryAuditEvent[]
      count: number
      scanned: number
      assertion_id: string | null
      note: string
    }>(
      `/tenants/${tenant}/audit/questions${q({
        limit: opts?.limit ?? 100,
        assertion_id: opts?.assertionId,
      })}`,
    ),

  getProvenance: (tenant: string, id: string) =>
    request<Provenance>(`/tenants/${tenant}/assertions/${id}/provenance`),

  listMetrics: (tenant: string) => request<Metric[]>(`/tenants/${tenant}/metrics`),
  createMetric: (tenant: string, m: MetricWrite) =>
    request<SavedMetric>(`/tenants/${tenant}/metrics`, { method: 'POST', body: JSON.stringify(m) }),
  updateMetric: (tenant: string, id: string, m: MetricWrite) =>
    request<SavedMetric>(`/tenants/${tenant}/metrics/${id}`, {
      method: 'PUT',
      body: JSON.stringify(m),
    }),
  setMetricStatus: (tenant: string, id: string, status: Metric['status']) =>
    request<Metric>(`/tenants/${tenant}/metrics/${id}/status`, {
      method: 'POST',
      body: JSON.stringify({ status }),
    }),
  compileMetric: (tenant: string, id: string) =>
    request<CompiledMetric>(`/tenants/${tenant}/metrics/${id}/compile`, {
      method: 'POST',
    }),
  /**
   * Compile a definition that has not been saved.
   *
   * The affordance that makes a deterministic compiler worth having: an author reads the SQL
   * their definition produces before anyone can be answered with it. No model is involved, so
   * the SQL shown here is the SQL that will run.
   */
  previewMetric: (tenant: string, m: MetricWrite) =>
    request<CompiledMetric>(`/tenants/${tenant}/metrics/preview`, {
      method: 'POST',
      body: JSON.stringify(m),
    }),
  /**
   * Load the example pack shipped for this tenant's ontology, as drafts.
   *
   * Always drafts from here. The endpoint takes `approve`, and offering that to a button would
   * make one click put definitions naming a fictional company's tables into service.
   */
  seedMetrics: (tenant: string) =>
    request<{ created: number; skipped: number; note: string }>(
      `/tenants/${tenant}/metrics/seed`,
      { method: 'POST' },
    ),

  /**
   * @deprecated Use `compose`, or drive the agent through `runRetrieval`.
   *
   * Answers from the first tier that can, and returns one tier's confidence for it. That is the
   * limitation `compose` exists to fix: a compiled metric, a quoted passage and a model's reading
   * are not the same kind of claim, and reporting one number over them invents a statistic.
   *
   * Still called by the Ask page, which is deprecated with it and no longer in the menu.
   */
  query: (
    tenant: string,
    body: { question: string; matter_id?: string; as_of?: string; min_confidence?: number },
  ) =>
    request<QueryResult>(`/tenants/${tenant}/query`, {
      method: 'POST',
      // The API field is `query`. Sending `question` produced a 422 naming a field the UI
      // never showed the user.
      body: JSON.stringify({
        query: body.question,
        matter_id: body.matter_id,
        as_of: body.as_of,
        min_confidence: body.min_confidence,
      }),
    }),

  /**
   * Answer from every lane at once rather than from the first tier that can.
   *
   * `synthesise: false` is the reviewable form: every part is returned with its own provenance
   * and no model writes over them. A governed metric that matches still short-circuits, so a
   * composed answer can legitimately come back as one lane.
   */
  compose: (
    tenant: string,
    body: {
      question: string
      execute?: boolean
      synthesise?: boolean
      min_confidence?: number
    },
  ) =>
    request<ComposedResult>(`/tenants/${tenant}/query/compose`, {
      method: 'POST',
      // The API field is `query`, as on /query. Sending `question` is a 422 naming a field the
      // user never saw.
      body: JSON.stringify({
        query: body.question,
        execute: body.execute ?? true,
        synthesise: body.synthesise ?? true,
        min_confidence: body.min_confidence,
      }),
    }),

  neighbourhood: (
    tenant: string,
    opts: {
      node_id?: string
      matter_id?: string
      depth?: number
      limit?: number
      min_confidence?: number
      as_of?: string
      include_pending?: boolean
      include_suggestions?: boolean
      /** Caps the overview within one half of the graph instead of across the whole of it. */
      layer?: string
    },
  ) => request<Neighbourhood>(`/tenants/${tenant}/graph/neighbourhood${q(opts)}`),

  ontology: (domain: string) => request<Ontology>(`/ontology/${domain}`),

  dashboard: (tenant: string) => request<DashboardStats>(`/tenants/${tenant}/dashboard`),

  getSettings: (tenant: string) => request<TenantSettings>(`/tenants/${tenant}/settings`),

  /**
   * Writes go to PATCH /governance, which is where the settings actually live and where
   * `GovernanceSettings.apply` validates them. There is no PUT /settings; GET /settings is
   * a read-only projection that joins governance onto process config.
   *
   * `min_confidence` is renamed on the way out: the projection flattens
   * `min_confidence_floor`, and sending the flattened name silently changes nothing,
   * because `apply` ignores keys it does not recognise.
   */
  updateSettings: async (tenant: string, s: Partial<TenantSettings>): Promise<TenantSettings> => {
    const { min_confidence, ...rest } = s
    const patch: Record<string, unknown> = { ...rest }
    if (min_confidence !== undefined) patch.min_confidence_floor = min_confidence
    await request<{ settings: unknown; warnings: string[] }>(`/tenants/${tenant}/governance`, {
      method: 'PATCH',
      body: JSON.stringify(patch),
    })
    // Re-read rather than trusting the patch response: the two shapes differ, and the
    // projection is what the page renders.
    return request<TenantSettings>(`/tenants/${tenant}/settings`)
  },

  /** `help` is FIELD_HELP, shipped with the settings so a tooltip cannot drift from its control. */
  getGovernance: (tenant: string) =>
    request<{ settings: RetrievalGovernance; help: Record<string, string> }>(
      `/tenants/${tenant}/governance`,
    ),

  updateGovernance: (tenant: string, patch: Partial<RetrievalGovernance>) =>
    request<{ settings: RetrievalGovernance; warnings: string[] }>(
      `/tenants/${tenant}/governance`,
      { method: 'PATCH', body: JSON.stringify(patch) },
    ),

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

  /** What is in the Glue catalog. Reads nothing into the graph. */
  glueDatabases: (tenant: string) =>
    request<{ databases: GlueDatabase[]; errors: string[]; note: string }>(
      `/tenants/${tenant}/glue/databases`,
    ),

  /** Scan the chosen databases. An empty list means every one the role can see, which is
   *  rarely what a firm wants: a shared catalog holds other teams' data. */
  scanSources: (tenant: string, databases: string[] = []) =>
    request<ScanReport>(`/tenants/${tenant}/sources/scan`, {
      method: 'POST',
      body: JSON.stringify({ databases }),
    }),

  /**
   * Ask a model to describe these tables. Returns as soon as the work is queued.
   *
   * 202, not a result: a Bedrock call per table is far too slow to hold a request open. Poll
   * `enrichmentStatus` for progress. Nothing it proposes is in use until somebody approves it.
   */
  enrichCatalog: (tenant: string, tables: string[] = [], sourceId = 'glue-main') =>
    request<{ status: string; tables_queued: number; max_tables_per_run: number; model: string }>(
      `/tenants/${tenant}/sources/enrich`,
      { method: 'POST', body: JSON.stringify({ tables, source_id: sourceId }) },
    ),

  enrichmentStatus: (tenant: string) =>
    request<EnrichmentStatus>(`/tenants/${tenant}/sources/enrich/status`),

  /** Record a description a person wrote. Live at once, and it outranks a model's proposal. */
  setDescription: (tenant: string, fullName: string, text: string, column?: string) =>
    request<{ assertion_id: string; live: boolean; source: string }>(
      `/tenants/${tenant}/tables/${encodeURIComponent(fullName)}/description`,
      { method: 'PATCH', body: JSON.stringify({ column: column ?? null, text }) },
    ),

  /** Approve every pending description, synonym and topic for one table. */
  approveTableEnrichment: (tenant: string, fullName: string) =>
    request<{ pending: number; approved: number; live: number; failed: Record<string, string> }>(
      `/tenants/${tenant}/tables/${encodeURIComponent(fullName)}/enrichment/approve`,
      { method: 'POST' },
    ),

  listAccessUsers: (tenant: string) =>
    // /access/users/{id} is one user's grants; the directory is /users, which returns
    // {scope, users}.
    request<{ users: DirectoryUser[] }>(`/tenants/${tenant}/users?scope=tenant`).then(
      (r) => r.users ?? [],
    ),
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

  // Unwraps {matter_id, user_id, events}. Declaring the response as a bare array instead
  // typechecked fine and then threw "is not iterable" the moment the caller spread it.
  accessAudit: (tenant: string, opts: { matter_id?: string; user_id?: string } = {}) =>
    request<{ events: AccessEvent[] } | AccessEvent[]>(
      `/tenants/${tenant}/access/audit${q(opts)}`,
    ).then((r) => (Array.isArray(r) ? r : r.events ?? [])),

  /** Groups of entity ids that may name one thing. A group is a question, not a finding. */
  entityDuplicates: (tenant: string) =>
    request<{ groups: DuplicateGroup[]; count: number }>(`/tenants/${tenant}/entities/duplicates`),

  /**
   * Restate every claim about `losing_id` as a claim about `winning_id`.
   *
   * `dryRun` mirrors the server default of true. Committing is a separate, explicit call because a
   * merge cascades: conclusions resting on the restated premises are withdrawn with them.
   */
  mergeEntities: (
    tenant: string,
    body: { losing_id: string; winning_id: string; reason: string; dry_run?: boolean },
  ) =>
    request<MergeResult>(`/tenants/${tenant}/entities/merge`, {
      method: 'POST',
      body: JSON.stringify({ dry_run: true, ...body }),
    }),
}
