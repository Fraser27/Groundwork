/**
 * Plain-language descriptions of the assertion contract, for a reader who is a
 * lawyer rather than a data engineer.
 *
 * The wording here is the product's explanation of itself, so it is kept in one
 * place: the badge tooltip, the dashboard, the review queue and the graph legend
 * all read from this. Mirrors src/graph/assertions.py.
 */

import type {
  AccessDecision,
  EpistemicClass,
  IngestState,
  Lane,
  ResolutionTier,
  ReviewState,
  RouterLayerKind,
} from './api'

export interface EpistemicMeta {
  label: string
  /** CSS variable holding this class's colour. */
  colour: string
  /** What it means — one sentence. */
  meaning: string
  /** How much to trust it, and what the system does with it. */
  trust: string
  /** Does it reach the live graph without a human signing it off? */
  autoAsserted: boolean
}

export const EPISTEMIC: Record<EpistemicClass, EpistemicMeta> = {
  DECLARED: {
    label: 'Declared',
    colour: 'var(--epi-declared)',
    meaning:
      'A system of record said so, a case management export, or a scan of the data catalogue.',
    trust:
      'Treat as given. It is true by definition of the source; if it is wrong, the source is wrong. Enters the graph without review.',
    autoAsserted: true,
  },
  EXTRACTED_DET: {
    label: 'Extracted (verified)',
    colour: 'var(--epi-extracted-det)',
    meaning:
      'A quoted sentence was found on a named page of the document, and the system checked the words are there.',
    trust:
      'The check is repeatable and says only that the words appear, nothing about what they imply. Anything read into them is a separate claim that waits for review. Enters the graph without review.',
    autoAsserted: true,
  },
  EXTRACTED_MODEL: {
    label: 'Extracted (model)',
    colour: 'var(--epi-extracted-model)',
    meaning:
      'Something read into the document rather than merely found in it, that one holding undercuts another, that a clause replaces an earlier one.',
    trust:
      'A judgement, not a check, so it is not yet trusted. This is the only class that waits for a person to approve or reject it, which is what the review queue is for.',
    autoAsserted: false,
  },
  INFERRED: {
    label: 'Inferred',
    colour: 'var(--epi-inferred)',
    meaning: 'A rule derived this from other assertions rather than reading it anywhere.',
    trust:
      'As trustworthy as its weakest premise, and never more, the system refuses to record an inference more confident than what it rests on. Open the proof tree to see exactly what it rests on.',
    autoAsserted: false,
  },
  PREDICTED: {
    label: 'Predicted',
    colour: 'var(--epi-predicted)',
    meaning:
      'A statistical guess from the shape of the graph. Nobody asserted it and no document says it.',
    trust:
      'A research hint, never a finding. Excluded from answers entirely unless you explicitly ask to see suggestions.',
    autoAsserted: false,
  },
}

export const EPISTEMIC_ORDER: EpistemicClass[] = [
  'DECLARED',
  'EXTRACTED_DET',
  'EXTRACTED_MODEL',
  'INFERRED',
  'PREDICTED',
]

export const REVIEW_STATE_LABEL: Record<ReviewState, string> = {
  AUTO_ASSERTED: 'Auto-asserted',
  PENDING: 'Pending review',
  APPROVED: 'Approved',
  REJECTED: 'Rejected',
}

// ── Glossary. Single source for every term that earns a (?) tooltip. ────────

export const HELP = {
  epistemicClass:
    'How the system came to believe a fact: declared by a system of record, quoted from a document and checked, read into a document by a language model, inferred by a rule, or guessed. Every fact in LexGraph carries one, which is what lets you ask why it is believed.',
  confidence:
    'How sure the system is of this single fact, from 0 to 1. A model’s interpretation is held below the trust floor until someone approves it, and approval raises it above the floor in the order the model reported. A quoted name found on the page sits on the floor exactly: the check is certain, but knowing a name appears somewhere is worth less than knowing who is adverse to whom. An inference can never exceed the weakest fact it rests on.',
  confidenceFloor:
    'The trust floor for retrieval. Facts scoring below it stay visible in the review queue and the audit trail, but they never shape an answer. Raising it makes the system more cautious and more likely to say it does not know.',
  rawConfidence:
    'What the model claimed for itself before anyone checked it. Approving a fact raises its confidence above the trust floor, keeping the order the model reported, so it can actually be used in an answer. This is the original number, kept so the one above it can always be explained.',
  reviewState:
    'Whether a person has signed this off. Derived from how the fact was reached, not chosen: a fact declared by a system of record, or a quote the system confirmed is on the page, goes live directly. Anything a model read into a document, and anything inferred, waits for a person. No ingest path can opt out of review.',
  premise:
    'A fact that an inference rests on. Retract a premise and the conclusion drawn from it is retracted too, so a proof tree is never left hanging.',
  proofTree:
    'The full chain of reasoning behind an inferred fact, unwound to the documents and records at its base. If any step in it is wrong, the conclusion is wrong, and you can see which step.',
  sourceLocator:
    'Exactly where a fact came from, precise enough to check by hand: for a document, the file, the page and the sentence quoted from it; for a database, the source, table and column.',
  pageCitation:
    'The citation for a fact read from a document: the file, the page, and the words themselves. It is deliberately what you would use to check it by hand, open the file at that page and search for the sentence. Any viewer can do that, and it still works after the document is re-processed.',
  quote:
    'The words from the document, copied exactly. Not a summary: a paraphrase could not be searched for in the original, so a claim whose quote cannot be found on the stated page is refused rather than recorded.',
  textOffsets:
    'An internal position within the extracted text, kept only for diagnosis. It does not correspond to a position in the file you would open, so it is not the citation, the file, page and quote are. Safe to ignore.',
  spanHash:
    'A fingerprint of the quoted words taken when the fact was recorded. If the underlying document is ever replaced, the fingerprint stops matching and the fact is flagged rather than silently drifting.',
  method:
    'The specific, versioned thing that produced this fact, for example a named model, a catalogue scan, or a rule with its version. Versioned so that improving it supersedes its old output instead of quietly mixing generations.',
  governingPredicate:
    'A relationship that drives a decision with legal consequence: conflicts, privilege, deadlines, citation authority. The list is closed, and a proposed fact using anything outside it is rejected when it is written, because a conflict check that misses a synonym looks exactly like a clean conflict check.',
  descriptivePredicate:
    'A subject-matter tag. The list is open. Sprawl here costs search precision, not a negligence claim.',
  matterWall:
    'An ethical wall. Matters you are walled off from are invisible to you: they do not appear in results, counts, or the graph. A denial always beats a permission, so a broad role cannot defeat a wall.',
  matterAssignment:
    'Being on a matter’s team. You can read a matter only if someone put you on it, so the starting position for everyone is no access at all. Taking someone off does not erase the record that they were on it, the file has to show who could read what, and when.',
  ethicalScreen:
    'A recorded instruction that one person must not see one matter, whatever else they hold. It overrides being on the team, and it overrides holding the administrator role, because a wall a senior person can read through is not a wall. Lifting one is also recorded rather than tidied away.',
  accessDecision:
    'The reason a matter is or is not open to someone, in four possibilities: they are on the team; they are screened from it; nobody has put them on it; or they hold the administrator role and can reach it without being on the team. Naming the reason matters, “screened” calls for a conversation with the risk team, while “not on the team” is usually just someone forgetting to staff them.',
  accessAudit:
    'The record of every change to who may read what: who made the change, who it was about, which matter, the reason given, and when. Entries are only ever added, never edited or removed, which is what makes it usable as evidence.',
  platformAdminAccess:
    'Access held because of a role rather than because anyone staffed the person onto the matter. It is shown separately for exactly that reason: it is the entry a reviewer should ask about. A screen still overrides it.',
  asOf:
    'Reconstructs what the file showed on a chosen date, including facts since retracted. This is the question that matters when someone asks what you knew at the time you advised.',
  bitemporal:
    'Two clocks. World time is when a fact was true; transaction time is when the system learned it. Keeping both is what makes "what did we know then" answerable.',
  tenant:
    'One firm. Each tenant has its own graph and every read is filtered to it. Matters are subgraphs inside a tenant, not separate graphs, because conflict checking is by definition cross-matter.',
  timeGrain:
    'The coarsest time buckets this metric may be reported in, monthly, quarterly, and so on. Fixed per metric so the same number cannot be quietly re-cut into a period it is not valid for.',
  additivity:
    'Whether a measure may be summed across periods. Fees billed are additive; matter headcount is not, adding it across months produces a number that means nothing.',
  governedMetric:
    'A metric definition that compiles to SQL deterministically. The same question always produces the same SQL, and no language model is involved in generating it.',
  resolutionTier:
    'Which part of the read path answered the question. The lower the tier, the less of the answer was generated: tier 1 is SQL compiled from an approved definition with no model at all, tier 2 walks facts already in the graph, tier 3 retrieves passages and the table definitions around them. None of the three lets a model write a query.',
  vectorRouter:
    'How the system decides which parts of itself to search. Your question is compared against short descriptions of this firm’s metrics, entities, tables and documents, and only the parts that came back close to it are searched. It chooses where to look; it never decides what is true, and it cannot let a part answer that your tenant is not permitted to use.',
  routerLayer:
    'One place the question could be answered from: the governed metrics, the entities in the graph, the catalogue tables, or the text of your documents. Each is scored separately, because the four hold different kinds of thing and one score across all of them would hide which.',
  similarity:
    'How close the wording of your question came to a description of this metric, entity, table or passage. It is a measure of resemblance between two pieces of text, not a probability and not a relevance percentage: 0.61 does not mean 61% likely to be right, and there is no fixed value above which something is relevant. Use it to compare candidates within one question, not across questions.',
  routerMargin:
    'How much worse than the best-scoring layer a layer may be and still be searched. At 0.35, a layer scoring within 35% of the best is searched too; anything further behind is left out. Not a relevance threshold, a comparison between layers, because resemblance scores are not comparable between one question and the next. Widen it to search more broadly and pay more latency; narrow it to search only the clearest match.',
  routerMinSimilarity:
    'The floor below which a match is not treated as a match at all. It answers one question only, did anything resemble the question, and when the answer is no the system searches everything rather than picking the least bad option. Keep it low: it is a sanity check, not a quality bar.',
  routerMetricBoost:
    'A fixed amount added to the governed-metric layer’s score, so a near-tie is resolved towards the deterministic answer. Governed metrics compile to SQL with no model involved, which makes them the better tie-break on principle rather than on score. It cannot promote a metric that matched nothing.',
  routerDegraded:
    'The router could not choose, so every permitted tier was run. It happens when nothing resembled the question, or when the search index is unreachable. The answer is not worse for it, but it is a different fact about the system than a router that chose, so it is stated rather than left to look like a choice.',
  routerRecorded:
    'The router scored the question but did not narrow anything. This view exists so you can see everything the firm’s data holds on a question, so it searches every permitted source and reports each separately; narrowing it would hide a source on the strength of a score you could then no longer check. The scores are still worth reading: they say where your wording resembles the system, which is often why one source came back richer than another.',
  tierForbidden:
    'This way of answering is switched off for your firm, so it was never tried. It says nothing about whether it would have been relevant, which is why it reads differently from a tier that was tried and scored too low.',
  ethicalGate:
    'The last step before you are shown anything: the graph, not a model, decides what may be included. It runs after retrieval and before any summary is written, so nothing the wall refused ever reaches the model. Both halves are recorded, what it refused and what it let through, because a gate that is only visible when it blocks reads as an exception rather than as something everything passed through.',
  ungovernedKillSwitch:
    'Governs whether a question no approved metric can answer may be answered with SQL a model wrote. On, it is refused instead. Governed metrics keep working: they compile from a definition and never depended on a model. Refused questions are logged and make a good backlog of metrics worth defining.',
  ontologyDomain:
    'Which vocabulary of entities, relationships and rules this tenant uses. The platform is domain-agnostic; the legal pack is simply the default.',
  graphLayer:
    'Which half of the graph an entity belongs to: facts read out of your documents, or schema declared by a system of record such as a Glue catalogue. They are one graph on purpose, so a metric over a table can be reconciled against a fact from a page, but you can read one at a time. Click a heading to isolate it. Edges that join the two stay drawn, because that join is the point.',
  ingestState:
    'Where a document is in the pipeline. Its text becomes searchable as soon as it is parsed, but anything a model read into it waits for review before it can shape an answer.',
  supersede:
    'Facts are never edited or deleted. A correction records a new fact and marks the old one superseded from that moment, so the audit trail stays intact.',
  retraction:
    'Withdrawing a fact. Anything inferred from it is withdrawn at the same time, so no conclusion outlives the reason it was drawn.',

  // ── Maintenance ───────────────────────────────────────────────────────────
  //
  // These four actions are destructive or long-running, so each says what it touches and
  // what it leaves alone. The uploaded files are never among the things touched, which is
  // the fact that makes the rest of it safe.
  derivedData:
    'Everything the system worked out from your documents and schemas: the facts in the graph, the search index, and the record of past ingest runs. All of it can be rebuilt, because the documents in S3 and the schemas in Glue are the originals and they are never touched.',
  reset:
    'Deletes the derived data you tick, for this tenant only. Your uploaded documents are not touched, so anything removed here can be rebuilt by Replay or Scan catalog. The one exception is metric definitions, which were written in this app and have no original to rebuild from.',
  replay:
    'Re-runs the reading pipeline over every document still in S3: each page is transcribed, split into passages, indexed for search, and read for facts again. Use it after a reset, or after improving an extractor. It reads the same original files, so citations still point at the same page and the same sentence.',
  scanCatalog:
    'Reads table and column definitions from AWS Glue and records them as facts declared by a system of record. Schemas only, no rows are read, so it is cheap and safe to re-run. Rows stay in the warehouse and are queried in place when a question needs them.',
} as const

// ── Access decisions ─────────────────────────────────────────────────────────

export interface AccessDecisionMeta {
  label: string
  colour: string
  /** What it means for the person. One sentence. */
  meaning: string
  /** What, if anything, a reviewer should do about it. */
  action: string
}

/** Mirrors src/access.py :: AccessDecision. Four states, each read differently. */
export const ACCESS_DECISIONS: Record<AccessDecision, AccessDecisionMeta> = {
  ALLOWED: {
    label: 'On the team',
    colour: 'var(--green)',
    meaning: 'Someone put this person on the matter, so they can read it.',
    action: 'Nothing to do. Check the team list is still the right one.',
  },
  SCREENED: {
    label: 'Screened',
    colour: 'var(--red)',
    meaning:
      'A wall was raised against this person on this matter. It overrides being on the team and it overrides the administrator role.',
    action:
      'The person is told the matter by name, shown the reason, and pointed at the contact. Lift it only with a reason of your own.',
  },
  NOT_ASSIGNED: {
    label: 'Not on the team',
    colour: 'var(--text-dim)',
    meaning: 'Nobody has put this person on the matter, so it is closed to them.',
    action: 'Not a wall, nobody decided anything. Add them if they should be working on it.',
  },
  PLATFORM_ADMIN: {
    label: 'By role',
    colour: 'var(--orange)',
    meaning:
      'Readable because this person holds the administrator role, not because anyone staffed them onto the matter.',
    action: 'The entry worth questioning. Screen them from anything they should not reach.',
  },
}

export const ACCESS_ACTION_LABEL: Record<string, string> = {
  ASSIGN: 'Added to matter',
  UNASSIGN: 'Removed from matter',
  SCREEN: 'Screen raised',
  LIFT_SCREEN: 'Screen lifted',
}

// ── Ingest state machine ─────────────────────────────────────────────────────

export const INGEST_STEP_LABEL: Record<string, string> = {
  REGISTERED: 'Registered',
  FETCHING: 'Fetching',
  PARSING: 'Parsing',
  CHUNKING: 'Chunking',
  EXTRACTING: 'Extracting',
  EMBEDDING: 'Embedding',
  GRAPH_STAGED: 'Staged',
  PENDING_REVIEW: 'Review',
  APPROVED: 'Approved',
  LIVE: 'Live',
  // Failures. Absent before, so a document that failed at chunking, staging or promotion showed
  // the raw enum name with no help text -- degraded quietly rather than crashing, which is why
  // it survived.
  FETCH_FAILED: 'Fetch failed',
  PARSE_FAILED: 'Parse failed',
  CHUNK_FAILED: 'Chunking failed',
  EXTRACT_FAILED: 'Extraction failed',
  EMBED_FAILED: 'Embedding failed',
  STAGE_FAILED: 'Staging failed',
  PROMOTE_FAILED: 'Promotion failed',
}

export const INGEST_STEP_HELP: Record<string, string> = {
  REGISTERED: 'The file is in immutable storage. Nothing has been read from it yet.',
  FETCHING: 'Retrieving the file for processing.',
  PARSING: 'Converting the document to text, keeping the page each passage came from so every later claim can cite a page you can turn to.',
  CHUNKING: 'Splitting the text into passages, each still carrying its page number.',
  EXTRACTING: 'Reading the passages to propose facts. A quote the system can confirm is on the stated page goes live directly; anything read into the text goes to review.',
  EMBEDDING: 'Indexing the passages for search. From here the verbatim text is findable even though its extracted facts are not yet trusted.',
  GRAPH_STAGED: 'Proposed facts are written to the graph but held back from answers.',
  PENDING_REVIEW: 'Waiting for a human to approve or reject the model-extracted facts.',
  APPROVED: 'Signed off, and about to go live. A document sits here only briefly.',
  LIVE: 'Reviewed and in use. Approved facts can now shape answers.',
  STAGE_FAILED:
    'The facts were read but could not be written to the graph. Nothing is lost — the document is still in S3 and Replay re-runs it.',
  PROMOTE_FAILED:
    'Approved but not yet in service: the step that makes facts live did not finish. The approval stands and Replay retries it.',
}

export function ingestPhase(state: IngestState): 'pending' | 'running' | 'review' | 'live' | 'failed' {
  if (state.endsWith('_FAILED')) return 'failed'
  // APPROVED reads as live rather than running: a reviewer has signed it off and only the
  // promotion step remains. Falling through to 'running' showed a finished document as in flight.
  if (state === 'LIVE' || state === 'APPROVED') return 'live'
  if (state === 'PENDING_REVIEW' || state === 'GRAPH_STAGED') return 'review'
  if (state === 'REGISTERED') return 'pending'
  return 'running'
}

/** The failed state that corresponds to each step, for drawing the pipeline. */
const FAILURE_OF: Record<string, string> = {
  FETCHING: 'FETCH_FAILED',
  PARSING: 'PARSE_FAILED',
  CHUNKING: 'CHUNK_FAILED',
  EXTRACTING: 'EXTRACT_FAILED',
  EMBEDDING: 'EMBED_FAILED',
  GRAPH_STAGED: 'GRAPH_FAILED',
}

export function failureStep(state: IngestState): string | null {
  const entry = Object.entries(FAILURE_OF).find(([, failed]) => failed === state)
  return entry ? entry[0] : null
}

// ── Resolution tiers ─────────────────────────────────────────────────────────

export interface TierMeta {
  label: string
  colour: string
  detail: string
  llm: string
}

export const TIERS: Record<ResolutionTier, TierMeta> = {
  1: {
    label: 'Governed metric',
    colour: 'var(--green)',
    detail: 'A governed metric matched the question and compiled to SQL deterministically.',
    llm: 'No model wrote any part of this query.',
  },
  2: {
    label: 'Graph traversal',
    colour: 'var(--epi-declared)',
    detail:
      'The answer came from walking assertions in the graph, expanding only along edges that clear the trust floor.',
    llm: 'A model phrased the answer; every fact in it is a cited assertion.',
  },
  3: {
    label: 'Hybrid',
    colour: 'var(--purple)',
    detail:
      'Passages were retrieved, the graph was walked out from them, and the catalogue supplied the definitions of any tables the question named. No SQL is written here.',
    llm: 'A model combined the parts; each one is traceable to its own source.',
  },
}

/** Tiers that no longer exist.
 *
 * Kept because the question log is append-only: rows recorded before a tier was retired still
 * name it, and they have to read as history rather than as a rendering fault.
 */
export const RETIRED_TIERS: Record<number, TierMeta> = {
  4: {
    label: 'Generated SQL (retired)',
    colour: 'var(--text-dim)',
    detail:
      'A route that let a model write SQL against the schema. Retired: no new question can be answered this way, so this can only be an answer given before it was withdrawn.',
    llm: 'A model wrote this query, and it was never a governed answer.',
  },
}

export function tierMeta(tier: number): TierMeta | null {
  if (tier === 1 || tier === 2 || tier === 3) return TIERS[tier]
  return RETIRED_TIERS[tier] ?? null
}

// ── The router's layers, and the lanes they turn into ───────────────────────

export interface LayerMeta {
  label: string
  colour: string
  /** What was searched, and what a hit in it is. One sentence. */
  meaning: string
  /** What expanding this layer shows. */
  items: string
}

/** Mirrors src/query/router_index.py :: ROUTING_KINDS, plus the document-chunk index. */
export const ROUTER_LAYERS: Record<RouterLayerKind, LayerMeta> = {
  metric: {
    label: 'Governed metrics',
    colour: 'var(--green)',
    meaning: 'The approved metric definitions this firm has written.',
    items: 'the metrics that matched, and what each measures',
  },
  entity: {
    label: 'Graph entities',
    colour: 'var(--epi-declared)',
    meaning: 'The parties, matters and authorities recorded in the graph.',
    items: 'the entities that matched, and their type',
  },
  table: {
    label: 'Catalogue tables',
    colour: 'var(--teal)',
    meaning: 'Table and column definitions read from the data catalogue. Schemas only.',
    items: 'the tables that matched, and their columns',
  },
  passages: {
    label: 'Document text',
    colour: 'var(--purple)',
    meaning: 'The text of your documents, indexed passage by passage.',
    items: 'the passages that matched, with the page each came from',
  },
}

export interface LaneMeta {
  label: string
  colour: string
  /** What this lane returns, in the terms the diagram uses. */
  returns: string
}

/** Mirrors src/query/planner.py :: Lane. The tier each lane corresponds to is in `TIERS`. */
export const LANES: Record<Lane, LaneMeta> = {
  metric: { label: 'Governed metric', colour: 'var(--green)', returns: 'rows, and the SQL' },
  graph: { label: 'Graph', colour: 'var(--epi-declared)', returns: 'facts, each one an assertion' },
  passages: { label: 'Documents', colour: 'var(--purple)', returns: 'passages, each with its page' },
  catalog: { label: 'Catalogue', colour: 'var(--teal)', returns: 'table and column definitions' },
  // Orange, alone among the lanes. The others return something a human approved or something
  // quoted; this returns a query nothing approved, and the colour is the disclosure.
  sql: { label: 'AI-written query', colour: 'var(--orange)', returns: 'rows, and the SQL to check' },
}

export const PART_PROVENANCE_LABEL: Record<string, string> = {
  deterministic: 'deterministic',
  // Not 'deterministic'. The figure is an exact aggregate over an approved definition, but no
  // question word matched it, so similarity chose which definition — and a tag reading
  // deterministic would rest a keyword match's guarantee on a cosine.
  model_selected: 'exact figure, metric chosen by similarity',
  verbatim: 'quoted verbatim',
  inferred: 'a model’s reading',
  // Not 'exact figure' either. model_selected picks between sums a person approved; this one wrote
  // the sum. The arithmetic is the thing to check, so the label points at it.
  model_written: 'query written by AI — read the SQL',
}
