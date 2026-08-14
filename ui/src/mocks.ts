/* ═══════════════════════════════════════════════════════════════════════════
   MOCK DATA — DELETE THIS FILE ONCE THE API IS LIVE.
   ═══════════════════════════════════════════════════════════════════════════

   Every page fetches for real first and falls back to these fixtures only when
   the request fails, so pages light up against a running API with no code change.
   When a fixture is in use the page shows a "sample data" flag, so nobody mistakes
   a demo for a deployment.

   To remove: delete this file, then delete the `fallback(...)` wrapper at each
   call site. `grep -rn "mocks" src` finds all of them.
   ══════════════════════════════════════════════════════════════════════════ */

import type {
  AccessEvent,
  ActivityEvent,
  Assertion,
  DashboardStats,
  DirectoryUser,
  DocumentDetail,
  DocumentSummary,
  Matter,
  MatterAccessDetail,
  MatterAssignment,
  MatterScreen,
  MattersResponse,
  Metric,
  Neighbourhood,
  Ontology,
  PageCitation,
  Provenance,
  QueryResult,
  ResolvedAccess,
  Source,
  TableDetail,
  TableSummary,
  TenantSettings,
  UserAccess,
  WithheldMatter,
} from './api'

/** True once any page has fallen back to a fixture in this session. */
let mockActive = false
export const isMockActive = () => mockActive

/** Try the real API; use the fixture if it is not there yet. */
export async function fallback<T>(p: Promise<T>, mock: T): Promise<T> {
  try {
    return await p
  } catch {
    mockActive = true
    return mock
  }
}

const iso = (daysAgo: number, hour = 9) => {
  const d = new Date()
  d.setDate(d.getDate() - daysAgo)
  d.setHours(hour, 17, 0, 0)
  return d.toISOString()
}

const loc = (o: Partial<Assertion['source_locator']>) => ({ ...o })

/** The method id a verified-presence claim carries: a model proposed it, a search confirmed it. */
const VERIFIED = 'llm:claude-sonnet-5+quote_check@v1'

function assertion(o: Partial<Assertion> & Pick<Assertion, 'assertion_id'>): Assertion {
  return {
    tenant_id: 'demo-firm',
    subject_id: 'n1',
    predicate: 'MENTIONS',
    object_id: 'n2',
    epistemic_class: 'EXTRACTED_MODEL',
    method: 'llm:claude-sonnet-5',
    confidence: 0.8,
    source_locator: {},
    premises: [],
    recorded_at: iso(1),
    review_state: 'PENDING',
    ...o,
  }
}

export const MOCK_MATTERS: Matter[] = [
  {
    matter_id: 'MAT-2041',
    name: 'Halveston Group — Series C',
    client: 'Halveston Group Ltd',
    status: 'open',
    opened_at: iso(112),
    counts: { documents: 34, assertions: 412, pending_review: 9, conflicts: 1 },
  },
  {
    matter_id: 'MAT-2088',
    name: 'Rowe v. Castleton Freight',
    client: 'Rowe (Claimant)',
    status: 'open',
    opened_at: iso(64),
    counts: { documents: 61, assertions: 908, pending_review: 14, conflicts: 0 },
  },
  {
    matter_id: 'MAT-1977',
    name: 'Northmoor Estate — Trust restructure',
    client: 'Northmoor Trustees',
    status: 'open',
    opened_at: iso(203),
    counts: { documents: 18, assertions: 221, pending_review: 2, conflicts: 0 },
  },
  {
    matter_id: 'MAT-1590',
    name: 'Ashcombe Holdings — Disposal',
    client: 'Ashcombe Holdings plc',
    status: 'closed',
    opened_at: iso(521),
    counts: { documents: 96, assertions: 1344, pending_review: 0, conflicts: 0 },
  },
]

/**
 * Matters the signed-in user is screened from, named rather than counted.
 *
 * Deliberately not in MOCK_MATTERS. The two lists stay apart the whole way through, so
 * a screened matter can never be mistaken for a readable one.
 */
export const MOCK_WITHHELD: WithheldMatter[] = [
  {
    matter_id: 'MAT-2103',
    reason: 'Acted for the counterparty, Brannigan Aggregates Ltd, in 2024.',
    contact: 'r.okonjo@thornevaux.example (Risk)',
  },
]

export const MOCK_MATTERS_RESPONSE: MattersResponse = {
  matters: MOCK_MATTERS,
  withheld: MOCK_WITHHELD,
}

// ── Access: firm directory, assignments, screens, audit ──────────────────────

/** Names of matters the signed-in user cannot read but a risk officer administering
 *  access can. Kept here so the access page can label the screen it is managing. */
const WITHHELD_MATTER_NAMES: Record<string, string> = {
  'MAT-2103': 'Brannigan Aggregates — Supply dispute',
}

export const MOCK_ACCESS_USERS: DirectoryUser[] = [
  { user_id: 'a.mensah@thornevaux.example', display_name: 'Adaeze Mensah' },
  { user_id: 'j.trelawney@thornevaux.example', display_name: 'James Trelawney' },
  { user_id: 'k.iyer@thornevaux.example', display_name: 'Kavi Iyer' },
  { user_id: 'p.duval@thornevaux.example', display_name: 'Perrine Duval' },
  { user_id: 'r.okonjo@thornevaux.example', display_name: 'Rita Okonjo', is_platform_admin: true },
  { user_id: 's.aldridge@thornevaux.example', display_name: 'Sian Aldridge' },
]

const nameOf = (userId: string) =>
  MOCK_ACCESS_USERS.find((u) => u.user_id === userId)?.display_name ?? userId

const matterNameOf = (matterId: string) =>
  MOCK_MATTERS.find((m) => m.matter_id === matterId)?.name ??
  WITHHELD_MATTER_NAMES[matterId] ??
  matterId

/** Every matter access can be administered over, readable to the caller or not. */
const MOCK_ACCESS_MATTERS: { matter_id: string; name: string }[] = [
  ...MOCK_MATTERS.map((m) => ({ matter_id: m.matter_id, name: m.name })),
  ...Object.entries(WITHHELD_MATTER_NAMES).map(([matter_id, name]) => ({ matter_id, name })),
]

const assign = (
  user_id: string,
  matter_id: string,
  role: string,
  granted_by: string,
  days: number,
): MatterAssignment => ({
  tenant_id: 'demo-firm',
  user_id,
  matter_id,
  role,
  granted_by,
  granted_at: iso(days, 10),
  revoked_at: null,
  revoked_by: null,
})

const MOCK_ASSIGNMENTS: MatterAssignment[] = [
  assign('a.mensah@thornevaux.example', 'MAT-2041', 'supervising partner', 'r.okonjo@thornevaux.example', 112),
  assign('k.iyer@thornevaux.example', 'MAT-2041', 'associate', 'a.mensah@thornevaux.example', 110),
  assign('p.duval@thornevaux.example', 'MAT-2041', 'paralegal', 'a.mensah@thornevaux.example', 74),
  assign('j.trelawney@thornevaux.example', 'MAT-2088', 'supervising partner', 'r.okonjo@thornevaux.example', 64),
  assign('k.iyer@thornevaux.example', 'MAT-2088', 'associate', 'j.trelawney@thornevaux.example', 60),
  assign('s.aldridge@thornevaux.example', 'MAT-1977', 'supervising partner', 'r.okonjo@thornevaux.example', 203),
  assign('p.duval@thornevaux.example', 'MAT-1977', 'paralegal', 's.aldridge@thornevaux.example', 190),
  assign('a.mensah@thornevaux.example', 'MAT-1590', 'supervising partner', 'r.okonjo@thornevaux.example', 521),
  assign('j.trelawney@thornevaux.example', 'MAT-2103', 'supervising partner', 'r.okonjo@thornevaux.example', 30),
  // On the team on paper; the screen below is what actually decides it. This is the row
  // both views have to render honestly rather than showing as ordinary access.
  assign('k.iyer@thornevaux.example', 'MAT-2103', 'associate', 'j.trelawney@thornevaux.example', 30),
]

const MOCK_SCREENS: MatterScreen[] = [
  {
    tenant_id: 'demo-firm',
    user_id: 'k.iyer@thornevaux.example',
    matter_id: 'MAT-2103',
    reason: 'Acted for the counterparty, Brannigan Aggregates Ltd, in 2024.',
    screened_by: 'r.okonjo@thornevaux.example',
    screened_at: iso(28, 9),
    contact: 'r.okonjo@thornevaux.example (Risk)',
    lifted_at: null,
    lifted_by: null,
  },
]

/** Mirrors MatterAccess.decide() and .explain() in src/access.py. A screen wins first. */
function decideAccess(userId: string, matterId: string, isAdmin: boolean): ResolvedAccess {
  const screen = MOCK_SCREENS.find(
    (s) => s.user_id === userId && s.matter_id === matterId && !s.lifted_at,
  )
  const assignment = MOCK_ASSIGNMENTS.find(
    (a) => a.user_id === userId && a.matter_id === matterId && !a.revoked_at,
  )
  const base = { matter_id: matterId, matter_name: matterNameOf(matterId) }

  if (screen) {
    const route = screen.contact ? `Contact ${screen.contact} to discuss.` : 'Contact your risk team.'
    return {
      ...base,
      decision: 'SCREENED',
      explanation: `You are screened from ${matterId}. Reason recorded: ${screen.reason} ${route}`,
      role: assignment?.role ?? null,
      reason: screen.reason,
      contact: screen.contact,
    }
  }
  if (isAdmin) {
    return {
      ...base,
      decision: 'PLATFORM_ADMIN',
      explanation: 'Visible because you hold platform-admin, not because of an assignment.',
      role: assignment?.role ?? null,
    }
  }
  if (assignment) {
    return {
      ...base,
      decision: 'ALLOWED',
      explanation: `You are assigned to ${matterId}.`,
      role: assignment.role,
    }
  }
  return {
    ...base,
    decision: 'NOT_ASSIGNED',
    explanation: `You are not assigned to ${matterId}. Ask the matter owner to add you if you need access.`,
    role: null,
  }
}

export function mockUserAccess(userId: string): UserAccess {
  const user = MOCK_ACCESS_USERS.find((u) => u.user_id === userId) ?? MOCK_ACCESS_USERS[0]
  const isAdmin = !!user.is_platform_admin
  return {
    user_id: user.user_id,
    display_name: user.display_name,
    is_platform_admin: isAdmin,
    assignments: MOCK_ASSIGNMENTS.filter((a) => a.user_id === user.user_id),
    screens: MOCK_SCREENS.filter((s) => s.user_id === user.user_id),
    decisions: MOCK_ACCESS_MATTERS.map((m) => decideAccess(user.user_id, m.matter_id, isAdmin)),
  }
}

export function mockMatterAccess(matterId: string): MatterAccessDetail {
  return {
    matter_id: matterId,
    matter_name: matterNameOf(matterId),
    team: MOCK_ASSIGNMENTS.filter((a) => a.matter_id === matterId && !a.revoked_at).map((a) => ({
      user_id: a.user_id,
      display_name: nameOf(a.user_id),
      role: a.role,
      granted_by: a.granted_by,
      granted_at: a.granted_at,
    })),
    screened: MOCK_SCREENS.filter((s) => s.matter_id === matterId && !s.lifted_at).map((s) => ({
      user_id: s.user_id,
      display_name: nameOf(s.user_id),
      reason: s.reason,
      contact: s.contact,
      screened_by: s.screened_by,
      screened_at: s.screened_at,
      overrides_assignment: MOCK_ASSIGNMENTS.some(
        (a) => a.user_id === s.user_id && a.matter_id === s.matter_id && !a.revoked_at,
      ),
    })),
  }
}

const accessEvent = (
  event_id: string,
  action: AccessEvent['action'],
  actor: string,
  subject_user: string,
  matter_id: string,
  days: number,
  reason?: string,
  detail?: Record<string, unknown>,
): AccessEvent => ({
  event_id,
  tenant_id: 'demo-firm',
  actor,
  action,
  subject_user,
  matter_id,
  at: iso(days, 10),
  reason: reason ?? null,
  detail: detail ?? {},
})

/** Newest first, as the endpoint returns it. Includes a lifted screen and a removal,
 *  because both survive in the trail rather than being tidied away. */
export const MOCK_ACCESS_AUDIT: AccessEvent[] = [
  accessEvent('ae_014', 'SCREEN', 'r.okonjo@thornevaux.example', 'k.iyer@thornevaux.example', 'MAT-2103', 28,
    'Acted for the counterparty, Brannigan Aggregates Ltd, in 2024.',
    { contact: 'r.okonjo@thornevaux.example (Risk)' }),
  accessEvent('ae_013', 'ASSIGN', 'j.trelawney@thornevaux.example', 'k.iyer@thornevaux.example', 'MAT-2103', 30,
    undefined, { role: 'associate' }),
  accessEvent('ae_012', 'ASSIGN', 'r.okonjo@thornevaux.example', 'j.trelawney@thornevaux.example', 'MAT-2103', 30,
    undefined, { role: 'supervising partner' }),
  accessEvent('ae_011', 'LIFT_SCREEN', 'r.okonjo@thornevaux.example', 'p.duval@thornevaux.example', 'MAT-1977', 44,
    'File review completed: the earlier engagement was for an unrelated party. Screen no longer justified.'),
  accessEvent('ae_010', 'UNASSIGN', 'a.mensah@thornevaux.example', 's.aldridge@thornevaux.example', 'MAT-2041', 58,
    'Moved to the Northmoor restructure and no longer working on this matter.'),
  accessEvent('ae_009', 'SCREEN', 'r.okonjo@thornevaux.example', 'p.duval@thornevaux.example', 'MAT-1977', 61,
    'Possible former engagement on the trustee side. Screened pending a file review.',
    { contact: 'r.okonjo@thornevaux.example (Risk)' }),
  accessEvent('ae_008', 'ASSIGN', 'a.mensah@thornevaux.example', 'p.duval@thornevaux.example', 'MAT-2041', 74,
    undefined, { role: 'paralegal' }),
  accessEvent('ae_007', 'ASSIGN', 'j.trelawney@thornevaux.example', 'k.iyer@thornevaux.example', 'MAT-2088', 60,
    undefined, { role: 'associate' }),
  accessEvent('ae_006', 'ASSIGN', 'r.okonjo@thornevaux.example', 'j.trelawney@thornevaux.example', 'MAT-2088', 64,
    undefined, { role: 'supervising partner' }),
  accessEvent('ae_005', 'ASSIGN', 's.aldridge@thornevaux.example', 'p.duval@thornevaux.example', 'MAT-1977', 190,
    undefined, { role: 'paralegal' }),
  accessEvent('ae_004', 'ASSIGN', 'r.okonjo@thornevaux.example', 's.aldridge@thornevaux.example', 'MAT-1977', 203,
    undefined, { role: 'supervising partner' }),
  accessEvent('ae_003', 'ASSIGN', 'a.mensah@thornevaux.example', 'k.iyer@thornevaux.example', 'MAT-2041', 110,
    undefined, { role: 'associate' }),
  accessEvent('ae_002', 'ASSIGN', 'r.okonjo@thornevaux.example', 'a.mensah@thornevaux.example', 'MAT-2041', 112,
    undefined, { role: 'supervising partner' }),
  accessEvent('ae_001', 'ASSIGN', 'r.okonjo@thornevaux.example', 'a.mensah@thornevaux.example', 'MAT-1590', 521,
    undefined, { role: 'supervising partner' }),
]

export function mockAccessAudit(opts: { matter_id?: string; user_id?: string } = {}): AccessEvent[] {
  return MOCK_ACCESS_AUDIT.filter(
    (e) =>
      (!opts.matter_id || e.matter_id === opts.matter_id) &&
      (!opts.user_id || e.subject_user === opts.user_id),
  )
}

export const MOCK_SOURCES: Source[] = [
  {
    source_id: 'src_glue_billing',
    name: 'Practice billing (Glue)',
    kind: 'GLUE',
    database: 'firm_billing',
    region: 'eu-west-2',
    table_count: 11,
    status: 'connected',
    last_scanned_at: iso(1, 4),
  },
  {
    source_id: 'src_glue_matters',
    name: 'Matter management (Glue)',
    kind: 'GLUE',
    database: 'firm_matters',
    region: 'eu-west-2',
    table_count: 7,
    status: 'connected',
    last_scanned_at: iso(1, 4),
  },
  {
    source_id: 'src_rs_finance',
    name: 'Finance warehouse (Redshift)',
    kind: 'REDSHIFT',
    database: 'finance',
    region: 'eu-west-2',
    table_count: 4,
    status: 'degraded',
    last_scanned_at: iso(9, 4),
  },
]

export const MOCK_TABLES: TableSummary[] = [
  {
    full_name: 'firm_matters.matter',
    name: 'matter',
    database: 'firm_matters',
    source_id: 'src_glue_matters',
    description: 'One row per engagement. The spine of the matter subgraph.',
    row_count: 4182,
    epistemic_class: 'DECLARED',
  },
  {
    full_name: 'firm_matters.party',
    name: 'party',
    database: 'firm_matters',
    source_id: 'src_glue_matters',
    description: 'Clients, counterparties and third parties. Role lives on the relationship.',
    row_count: 21904,
    epistemic_class: 'DECLARED',
  },
  {
    full_name: 'firm_matters.matter_party',
    name: 'matter_party',
    database: 'firm_matters',
    source_id: 'src_glue_matters',
    description: 'Party-to-matter roles, including adverse positions.',
    row_count: 38771,
    epistemic_class: 'DECLARED',
  },
  {
    full_name: 'firm_billing.time_entry',
    name: 'time_entry',
    database: 'firm_billing',
    source_id: 'src_glue_billing',
    description: 'Recorded time at fee-earner grain.',
    row_count: 2841903,
    epistemic_class: 'DECLARED',
  },
  {
    full_name: 'firm_billing.invoice',
    name: 'invoice',
    database: 'firm_billing',
    source_id: 'src_glue_billing',
    description: 'Issued invoices with matter and period.',
    row_count: 92014,
    epistemic_class: 'DECLARED',
  },
  {
    full_name: 'finance.wip_snapshot',
    name: 'wip_snapshot',
    database: 'finance',
    source_id: 'src_rs_finance',
    description: 'Month-end work in progress. A balance, so not additive across periods.',
    row_count: 40122,
    epistemic_class: 'DECLARED',
  },
]

export const MOCK_TABLE_DETAIL: TableDetail = {
  ...MOCK_TABLES[0],
  method: 'glue:catalog_scan@v2',
  scanned_at: iso(1, 4),
  columns: [
    { name: 'matter_id', data_type: 'varchar', description: 'Primary key.', is_partition: false, is_primary_key: true },
    { name: 'matter_name', data_type: 'varchar', description: 'Engagement name as opened.', is_partition: false, is_primary_key: false },
    { name: 'client_party_id', data_type: 'varchar', description: 'Instructing party.', is_partition: false, is_primary_key: false },
    { name: 'practice_area', data_type: 'varchar', description: 'Owning department.', is_partition: false, is_primary_key: false },
    { name: 'opened_date', data_type: 'date', description: 'Date the engagement was opened.', is_partition: false, is_primary_key: false },
    { name: 'closed_date', data_type: 'date', description: 'Null while the matter is live.', is_partition: false, is_primary_key: false },
    { name: 'lead_fee_earner', data_type: 'varchar', description: 'Supervising partner.', is_partition: false, is_primary_key: false },
    { name: 'open_year', data_type: 'varchar', description: 'Partition key. A string, not a date.', is_partition: true, is_primary_key: false },
  ],
}

export const MOCK_DOCUMENTS: DocumentSummary[] = [
  {
    document_id: 'doc_8f21a0',
    filename: 'Halveston — Subscription Agreement (executed).pdf',
    matter_id: 'MAT-2041',
    state: 'PENDING_REVIEW',
    uploaded_at: iso(0, 8),
    page_count: 74,
    size_bytes: 3_812_004,
    assertion_count: 61,
    pending_review_count: 9,
  },
  {
    document_id: 'doc_44c9de',
    filename: 'Rowe — Particulars of Claim.pdf',
    matter_id: 'MAT-2088',
    state: 'LIVE',
    uploaded_at: iso(3, 11),
    page_count: 22,
    size_bytes: 901_233,
    assertion_count: 38,
    pending_review_count: 0,
  },
  {
    document_id: 'doc_1b7702',
    filename: 'Castleton — Expert report (Kearns).pdf',
    matter_id: 'MAT-2088',
    state: 'EXTRACTING',
    uploaded_at: iso(0, 12),
    page_count: 118,
    size_bytes: 7_004_881,
    assertion_count: 0,
    pending_review_count: 0,
  },
  {
    document_id: 'doc_9aa311',
    filename: 'Northmoor — Trust Deed 1994 (scanned).pdf',
    matter_id: 'MAT-1977',
    state: 'PARSE_FAILED',
    uploaded_at: iso(1, 15),
    page_count: null,
    size_bytes: 12_400_112,
    assertion_count: 0,
    pending_review_count: 0,
    error: 'Transcription failed on page 4: the vision model was throttled. Retry from parsing.',
  },
  {
    document_id: 'doc_5c0f8e',
    filename: 'Halveston — Board minutes 12 Mar.docx',
    matter_id: 'MAT-2041',
    state: 'EMBEDDING',
    uploaded_at: iso(0, 13),
    page_count: 6,
    size_bytes: 88_120,
    assertion_count: 7,
    pending_review_count: 0,
  },
  {
    document_id: 'doc_2d5b41',
    filename: 'Ashcombe — SPA (final).pdf',
    matter_id: 'MAT-1590',
    state: 'LIVE',
    uploaded_at: iso(41, 10),
    page_count: 210,
    size_bytes: 9_221_004,
    assertion_count: 204,
    pending_review_count: 0,
  },
]

export const MOCK_PENDING: Assertion[] = [
  assertion({
    assertion_id: 'a_7c41e9b2f0a34d18',
    matter_id: 'MAT-2041',
    subject_id: 'p_halveston',
    subject_label: 'Halveston Group Ltd',
    subject_type: 'Party',
    predicate: 'ADVERSE_TO',
    object_id: 'p_meridian',
    object_label: 'Meridian Capital Partners LLP',
    object_type: 'Party',
    confidence: 0.71,
    source_locator: loc({
      document_id: 'doc_8f21a0',
      filename: 'Halveston — Subscription Agreement (executed).pdf',
      page: 41,
      chunk_id: 'doc_8f21a0#p41#c3',
      quote:
        'the Company shall indemnify Meridian Capital Partners LLP in respect of any claim brought against it by the Investor',
    }),
    source_context:
      '12.4 Save in respect of fraud or wilful default, the Company shall indemnify Meridian Capital Partners LLP in respect of any claim brought against it by the Investor arising out of the transactions contemplated by this Agreement, subject always to the limitations in clause 13.',
    recorded_at: iso(0, 8),
  }),
  assertion({
    assertion_id: 'a_1f90d7ac55be4402',
    matter_id: 'MAT-2041',
    subject_id: 'd_8f21a0',
    subject_label: 'Subscription Agreement',
    subject_type: 'Document',
    predicate: 'SUBJECT_TO_PRIVILEGE',
    object_id: 'MAT-2041',
    object_label: 'Halveston Group — Series C',
    object_type: 'Matter',
    confidence: 0.64,
    source_locator: loc({
      document_id: 'doc_8f21a0',
      filename: 'Halveston — Subscription Agreement (executed).pdf',
      page: 1,
      chunk_id: 'doc_8f21a0#p1#c1',
      quote: 'PRIVILEGED AND CONFIDENTIAL — PREPARED IN CONTEMPLATION OF LITIGATION',
    }),
    source_context:
      'PRIVILEGED AND CONFIDENTIAL — PREPARED IN CONTEMPLATION OF LITIGATION. This draft is circulated to the board for comment only and should not be forwarded outside the group.',
    recorded_at: iso(0, 8),
  }),
  assertion({
    assertion_id: 'a_b3e8225c91df4770',
    matter_id: 'MAT-2088',
    subject_id: 'c_dalgleish',
    subject_label: 'Dalgleish & Rowe LLP',
    subject_type: 'Counsel',
    predicate: 'REPRESENTS',
    object_id: 'p_castleton',
    object_label: 'Castleton Freight Ltd',
    object_type: 'Party',
    confidence: 0.58,
    source_locator: loc({
      document_id: 'doc_44c9de',
      filename: 'Rowe — Particulars of Claim.pdf',
      page: 2,
      chunk_id: 'doc_44c9de#p2#c1',
      quote: 'Dalgleish & Rowe LLP, solicitors for the Second Defendant',
    }),
    source_context:
      'Served this 14th day of the month by Dalgleish & Rowe LLP, solicitors for the Second Defendant, whose address for service is set out below. The First Defendant is separately represented.',
    recorded_at: iso(2, 16),
  }),
  assertion({
    assertion_id: 'a_5da0417ee2cb49aa',
    matter_id: 'MAT-2088',
    subject_id: 'd_44c9de',
    subject_label: 'Particulars of Claim',
    subject_type: 'Document',
    predicate: 'DISTINGUISHES',
    object_id: 'auth_wilkes',
    object_label: 'Wilkes v Argonaut Shipping [2019] EWCA Civ 884',
    object_type: 'Authority',
    confidence: 0.79,
    source_locator: loc({
      document_id: 'doc_44c9de',
      filename: 'Rowe — Particulars of Claim.pdf',
      page: 14,
      chunk_id: 'doc_44c9de#p14#c2',
      quote:
        'Wilkes is not on point: there the carrier had accepted the goods without reservation',
    }),
    source_context:
      'The Defendant will no doubt rely on Wilkes v Argonaut Shipping. Wilkes is not on point: there the carrier had accepted the goods without reservation, whereas here the bill of lading was clausulated on receipt.',
    recorded_at: iso(2, 16),
  }),
  assertion({
    assertion_id: 'a_902ab7f14c0e4e63',
    matter_id: 'MAT-1977',
    subject_id: 'dl_northmoor_1',
    subject_label: 'Appointment deadline',
    subject_type: 'Deadline',
    predicate: 'DEADLINE_FOR',
    object_id: 'MAT-1977',
    object_label: 'Northmoor Estate — Trust restructure',
    object_type: 'Matter',
    confidence: 0.45,
    source_locator: loc({
      document_id: 'doc_9aa311',
      filename: 'Northmoor — Trust Deed 1994 (scanned).pdf',
      page: 7,
      chunk_id: 'doc_9aa311#p7#c4',
      quote: 'within two years of the death of the Settlor',
    }),
    source_context:
      'The Trustees may exercise the power of appointment conferred by clause 4 within two years of the death of the Settlor, and not thereafter.',
    recorded_at: iso(1, 15),
  }),
]

const INFERRED_CONFLICT = assertion({
  assertion_id: 'a_inf_conflict_0091',
  matter_id: 'MAT-2041',
  subject_id: 'MAT-2041',
  subject_label: 'Halveston Group — Series C',
  subject_type: 'Matter',
  predicate: 'POTENTIAL_CONFLICT',
  object_id: 'p_meridian',
  object_label: 'Meridian Capital Partners LLP',
  object_type: 'Party',
  epistemic_class: 'INFERRED',
  method: 'rule:conflict_check@v1',
  rule_id: 'conflict_check',
  rule_version: 'v1',
  confidence: 0.95,
  premises: ['a_prem_represents_01', 'a_prem_adverse_01'],
  source_locator: loc({ source_id: 'src_glue_matters', table: 'matter_party' }),
  review_state: 'PENDING',
  recorded_at: iso(1, 7),
})

export const MOCK_ASSERTIONS: Assertion[] = [
  ...MOCK_PENDING,
  INFERRED_CONFLICT,
  // Verified presence: a model said these words are on page 3, a string search agreed.
  // MENTIONS is the only predicate a presence check can support.
  assertion({
    assertion_id: 'a_c17f4e6b8a2d4915',
    matter_id: 'MAT-2041',
    subject_id: 'd_5c0f8e',
    subject_label: 'Board minutes 12 Mar',
    subject_type: 'Document',
    predicate: 'MENTIONS',
    object_id: 'p_ashcombe',
    object_label: 'Ashcombe Holdings plc',
    object_type: 'Party',
    epistemic_class: 'EXTRACTED_DET',
    method: VERIFIED,
    confidence: 0.99,
    review_state: 'AUTO_ASSERTED',
    source_locator: loc({
      document_id: 'doc_5c0f8e',
      filename: 'Halveston — Board minutes 12 Mar.docx',
      page: 3,
      chunk_id: 'doc_5c0f8e#p3#c2',
      quote: 'a possible approach from Ashcombe Holdings plc was noted',
    }),
    source_context:
      'Under item 6, a possible approach from Ashcombe Holdings plc was noted. No decision was taken and the matter was deferred.',
    recorded_at: iso(0, 13),
  }),
  assertion({
    assertion_id: 'a_prem_represents_01',
    matter_id: 'MAT-1590',
    subject_id: 'c_firm',
    subject_label: 'Thorne Vaux LLP',
    subject_type: 'Counsel',
    predicate: 'REPRESENTS',
    object_id: 'p_meridian',
    object_label: 'Meridian Capital Partners LLP',
    object_type: 'Party',
    epistemic_class: 'DECLARED',
    method: 'glue:catalog_scan@v2',
    confidence: 1.0,
    review_state: 'AUTO_ASSERTED',
    source_locator: loc({ source_id: 'src_glue_matters', table: 'matter_party', column: 'role' }),
    recorded_at: iso(30, 4),
  }),
  // Adverse positions come from the case management export rather than a document:
  // a quote proves a name appears, and asserting a position from that is judgement.
  assertion({
    assertion_id: 'a_prem_adverse_01',
    matter_id: 'MAT-2041',
    subject_id: 'MAT-2041',
    subject_label: 'Halveston Group — Series C',
    subject_type: 'Matter',
    predicate: 'ADVERSE_TO',
    object_id: 'p_meridian',
    object_label: 'Meridian Capital Partners LLP',
    object_type: 'Party',
    epistemic_class: 'DECLARED',
    method: 'glue:catalog_scan@v2',
    confidence: 1.0,
    review_state: 'AUTO_ASSERTED',
    source_locator: loc({ source_id: 'src_glue_matters', table: 'matter_party', column: 'role' }),
    recorded_at: iso(1, 7),
  }),
  // Verified presence of the same party in the agreement — the words are on page 3,
  // and that is all this fact claims.
  assertion({
    assertion_id: 'a_mentions_meridian_01',
    matter_id: 'MAT-2041',
    subject_id: 'd_8f21a0',
    subject_label: 'Subscription Agreement',
    subject_type: 'Document',
    predicate: 'MENTIONS',
    object_id: 'p_meridian',
    object_label: 'Meridian Capital Partners LLP',
    object_type: 'Party',
    epistemic_class: 'EXTRACTED_DET',
    method: VERIFIED,
    confidence: 0.99,
    review_state: 'AUTO_ASSERTED',
    source_locator: loc({
      document_id: 'doc_8f21a0',
      filename: 'Halveston — Subscription Agreement (executed).pdf',
      page: 3,
      chunk_id: 'doc_8f21a0#p3#c1',
      quote: 'Meridian Capital Partners LLP (the "Adverse Party")',
    }),
    recorded_at: iso(1, 7),
  }),
  // CITES implies reliance, which a quote match cannot establish — so it is reviewed
  // rather than auto-asserted. "The court declined to follow Wilkes" is the case that
  // makes this necessary.
  assertion({
    assertion_id: 'a_cite_wilkes_01',
    matter_id: 'MAT-2088',
    subject_id: 'd_44c9de',
    subject_label: 'Particulars of Claim',
    subject_type: 'Document',
    predicate: 'CITES',
    object_id: 'auth_wilkes',
    object_label: 'Wilkes v Argonaut Shipping [2019] EWCA Civ 884',
    object_type: 'Authority',
    epistemic_class: 'EXTRACTED_MODEL',
    method: 'llm:claude-sonnet-5',
    confidence: 0.79,
    review_state: 'APPROVED',
    reviewed_by: 'k.iyer@thornevaux.example',
    reviewed_at: iso(3, 13),
    source_locator: loc({
      document_id: 'doc_44c9de',
      filename: 'Rowe — Particulars of Claim.pdf',
      page: 14,
      chunk_id: 'doc_44c9de#p14#c1',
      quote: 'Wilkes v Argonaut Shipping [2019] EWCA Civ 884',
    }),
    recorded_at: iso(3, 11),
  }),
  assertion({
    assertion_id: 'a_mentions_wilkes_01',
    matter_id: 'MAT-2088',
    subject_id: 'd_44c9de',
    subject_label: 'Particulars of Claim',
    subject_type: 'Document',
    predicate: 'MENTIONS',
    object_id: 'auth_wilkes',
    object_label: 'Wilkes v Argonaut Shipping [2019] EWCA Civ 884',
    object_type: 'Authority',
    epistemic_class: 'EXTRACTED_DET',
    method: VERIFIED,
    confidence: 0.99,
    review_state: 'AUTO_ASSERTED',
    source_locator: loc({
      document_id: 'doc_44c9de',
      filename: 'Rowe — Particulars of Claim.pdf',
      page: 14,
      chunk_id: 'doc_44c9de#p14#c1',
      quote: 'Wilkes v Argonaut Shipping [2019] EWCA Civ 884',
    }),
    recorded_at: iso(3, 11),
  }),
  assertion({
    assertion_id: 'a_pred_link_01',
    matter_id: 'MAT-2088',
    subject_id: 'p_castleton',
    subject_label: 'Castleton Freight Ltd',
    subject_type: 'Party',
    predicate: 'CONCERNS_TOPIC',
    object_id: 't_carriage',
    object_label: 'Carriage of goods by sea',
    object_type: 'Topic',
    epistemic_class: 'PREDICTED',
    method: 'linkpred:node2vec@v1',
    confidence: 0.52,
    review_state: 'PENDING',
    source_locator: loc({ source_id: 'src_glue_matters', table: 'matter' }),
    recorded_at: iso(4, 5),
  }),
  assertion({
    assertion_id: 'a_rejected_01',
    matter_id: 'MAT-2088',
    subject_id: 'c_dalgleish',
    subject_label: 'Dalgleish & Rowe LLP',
    subject_type: 'Counsel',
    predicate: 'REPRESENTS',
    object_id: 'p_rowe',
    object_label: 'Rowe (Claimant)',
    object_type: 'Party',
    confidence: 0.61,
    review_state: 'REJECTED',
    reviewed_by: 'a.mensah@thornevaux.example',
    reviewed_at: iso(5, 14),
    source_locator: loc({
      document_id: 'doc_44c9de',
      filename: 'Rowe — Particulars of Claim.pdf',
      page: 2,
      chunk_id: 'doc_44c9de#p2#c1',
      quote: 'Dalgleish & Rowe LLP, solicitors for the Second Defendant',
    }),
    superseded_at: iso(5, 14),
    recorded_at: iso(6, 9),
  }),
]

/** File + page + quote for a document-sourced fact. No download_url: the fixture has
 *  no file behind it, and the viewer says so rather than showing an empty frame. */
function mockCitation(a: Assertion): PageCitation | null {
  const l = a.source_locator
  if (!l.document_id || !l.quote || l.page == null) return null
  const doc = MOCK_DOCUMENTS.find((d) => d.document_id === l.document_id)
  const context = a.source_context ?? ''
  return {
    document_id: l.document_id,
    filename: l.filename ?? doc?.filename ?? l.document_id,
    page: l.page,
    quote: l.quote,
    chunk_id: l.chunk_id,
    page_count: doc?.page_count ?? null,
    context_before: context ? context.split(l.quote)[0] || null : null,
    context_after: context ? context.split(l.quote)[1] || null : null,
    span_sha256: 'e3b0c44298fc1c149afbf4c8996fb924',
    download_url: null,
    expires_at: null,
  }
}

export function mockProvenance(id: string): Provenance {
  const a = MOCK_ASSERTIONS.find((x) => x.assertion_id === id) ?? MOCK_PENDING[0]

  const proof =
    a.epistemic_class === 'INFERRED'
      ? {
          assertion: a,
          premises: a.premises
            .map((pid) => MOCK_ASSERTIONS.find((x) => x.assertion_id === pid))
            .filter((x): x is Assertion => !!x)
            .map((prem) => ({ assertion: prem, premises: [] })),
        }
      : null

  return {
    assertion: a,
    citation: mockCitation(a),
    proof,
    history: [
      {
        event_id: `${a.assertion_id}-1`,
        timestamp: a.recorded_at,
        action: 'ASSERTED',
        actor: a.method,
        note: `Recorded with confidence ${a.confidence.toFixed(2)}`,
      },
      ...(a.review_state === 'REJECTED'
        ? ([
            {
              event_id: `${a.assertion_id}-2`,
              timestamp: a.reviewed_at ?? a.recorded_at,
              action: 'REJECTED' as const,
              actor: a.reviewed_by ?? 'reviewer',
              note: 'Reads as counsel for the Second Defendant, not the Claimant. Model inverted the parties.',
            },
            {
              event_id: `${a.assertion_id}-3`,
              timestamp: a.reviewed_at ?? a.recorded_at,
              action: 'SUPERSEDED' as const,
              actor: 'system',
              note: 'Withdrawn from the live graph.',
            },
          ])
        : []),
    ],
  }
}

export function mockDocumentDetail(id: string): DocumentDetail {
  const d = MOCK_DOCUMENTS.find((x) => x.document_id === id) ?? MOCK_DOCUMENTS[0]
  const states: DocumentDetail['timeline'] = [
    { state: 'REGISTERED', at: d.uploaded_at, detail: 'Stored immutably; content hash recorded.' },
    { state: 'FETCHING', at: d.uploaded_at, detail: null },
    {
      state: 'PARSING',
      at: d.uploaded_at,
      detail: 'vision:claude-haiku-4-5@v1, page by page at 150 DPI',
    },
  ]
  if (d.state === 'PARSE_FAILED') {
    states.push({ state: 'PARSE_FAILED', at: d.uploaded_at, detail: d.error ?? null })
  } else {
    states.push(
      { state: 'CHUNKING', at: d.uploaded_at, detail: '412 passages, page numbers retained' },
      { state: 'EXTRACTING', at: d.uploaded_at, detail: 'llm:claude-sonnet-5, quotes checked against the page' },
    )
    if (d.state === 'LIVE' || d.state === 'PENDING_REVIEW' || d.state === 'EMBEDDING') {
      states.push({ state: 'EMBEDDING', at: d.uploaded_at, detail: 'Verbatim text now searchable' })
    }
    if (d.state === 'LIVE' || d.state === 'PENDING_REVIEW') {
      states.push({ state: 'GRAPH_STAGED', at: d.uploaded_at, detail: null })
      states.push({ state: 'PENDING_REVIEW', at: d.uploaded_at, detail: `${d.pending_review_count} model claims awaiting sign-off` })
    }
    if (d.state === 'LIVE') {
      states.push({ state: 'LIVE', at: d.uploaded_at, detail: 'All model claims reviewed' })
    }
  }

  return {
    ...d,
    s3_uri: `s3://lexgraph-demo-firm-raw/${d.matter_id}/${d.document_id}.pdf`,
    content_sha256: 'a5f3c9e17b04d2118ee6c0f4b7d9a3128c55e0f7',
    timeline: states,
    assertions: MOCK_ASSERTIONS.filter((a) => a.source_locator.document_id === d.document_id),
  }
}

export const MOCK_METRICS: Metric[] = [
  {
    metric_id: 'm_001',
    name: 'fees_billed',
    definition: 'Total fees invoiced, excluding disbursements and VAT.',
    expression: 'SUM(i.fee_amount)',
    source_table: 'firm_billing.invoice',
    source_id: 'src_glue_billing',
    grain: ['matter_id', 'practice_area'],
    time_grain_column: 'issued_date',
    time_grains: ['month', 'quarter', 'year'],
    aggregation: 'additive',
    parameters: [
      { column: 'practice_area', operator: '=', required: false, description: 'Restrict to one department.' },
    ],
    filters: ['i.status = \'ISSUED\''],
    synonyms: ['billings', 'revenue', 'turnover'],
    status: 'approved',
    version: 4,
    owner: 'finance',
    updated_by: 'r.okonjo@thornevaux.example',
    updated_at: iso(12, 11),
  },
  {
    metric_id: 'm_002',
    name: 'recorded_hours',
    definition: 'Hours recorded against a matter, billable and non-billable.',
    expression: 'SUM(t.hours)',
    source_table: 'firm_billing.time_entry',
    source_id: 'src_glue_billing',
    grain: ['matter_id', 'fee_earner_id'],
    time_grain_column: 'entry_date',
    time_grains: ['day', 'week', 'month'],
    aggregation: 'additive',
    parameters: [],
    filters: [],
    synonyms: ['time recorded', 'hours'],
    status: 'approved',
    version: 2,
    owner: 'finance',
    updated_by: 'r.okonjo@thornevaux.example',
    updated_at: iso(30, 10),
  },
  {
    metric_id: 'm_003',
    name: 'work_in_progress',
    definition: 'Unbilled work in progress at period end.',
    expression: 'SUM(w.wip_value)',
    source_table: 'finance.wip_snapshot',
    source_id: 'src_rs_finance',
    grain: ['matter_id'],
    time_grain_column: 'snapshot_date',
    time_grains: ['month'],
    aggregation: 'semi_additive',
    parameters: [],
    filters: [],
    synonyms: ['WIP', 'unbilled'],
    status: 'draft',
    version: 1,
    owner: 'finance',
    updated_by: 'k.iyer@thornevaux.example',
    updated_at: iso(2, 16),
  },
  {
    metric_id: 'm_004',
    name: 'open_matters',
    definition: 'Count of matters open at the end of the period.',
    expression: 'COUNT(DISTINCT m.matter_id)',
    source_table: 'firm_matters.matter',
    source_id: 'src_glue_matters',
    grain: ['practice_area'],
    time_grain_column: 'opened_date',
    time_grains: ['month', 'quarter'],
    aggregation: 'non_additive',
    parameters: [],
    filters: ['m.closed_date IS NULL'],
    synonyms: ['live matters', 'caseload'],
    status: 'approved',
    version: 3,
    owner: 'operations',
    updated_by: 'a.mensah@thornevaux.example',
    updated_at: iso(20, 9),
  },
]

export function mockCompiledSql(m: Metric): string {
  const filters = m.filters.length ? `\nWHERE ${m.filters.join('\n  AND ')}` : ''
  const dims = [...(m.time_grain_column ? [`DATE_TRUNC('month', ${m.time_grain_column}) AS period`] : []), ...m.grain]
  return `-- Compiled from governed metric ${m.metric_id} v${m.version}. No model involved.
SELECT
  ${dims.join(',\n  ')},
  ${m.expression} AS ${m.name}
FROM ${m.source_table}${filters}
GROUP BY ${dims.map((_, i) => i + 1).join(', ')}
ORDER BY 1 DESC
LIMIT 100`
}

export const MOCK_QUERY_RESULTS: Record<string, QueryResult> = {
  tier1: {
    question: 'What were fees billed by practice area last quarter?',
    tier: 1,
    tier_reason:
      'The question matched the governed metric fees_billed (approved, v4) on the synonym "billed".',
    answer:
      'Fees billed in the last completed quarter were 4,182,900 across four practice areas, led by Corporate at 1,904,220.',
    metric_id: 'm_001',
    sql: mockCompiledSql(MOCK_METRICS[0]),
    citations: [],
    rows: {
      columns: ['period', 'practice_area', 'fees_billed'],
      rows: [
        ['2026-Q2', 'Corporate', 1904220],
        ['2026-Q2', 'Litigation', 1220480],
        ['2026-Q2', 'Real Estate', 702100],
        ['2026-Q2', 'Private Client', 356100],
      ],
    },
  },
  tier2: {
    question: 'Does acting for Halveston create a conflict?',
    tier: 2,
    tier_reason:
      'No governed metric applies. The answer came from walking assertions along governing predicates only, above the 0.80 trust floor.',
    answer:
      'Yes — a potential conflict is flagged. The firm is recorded as representing Meridian Capital Partners LLP on MAT-1590, and Meridian is recorded as adverse to the Halveston matter.[1][2]',
    path: [
      {
        subject_label: 'Thorne Vaux LLP',
        predicate: 'REPRESENTS',
        object_label: 'Meridian Capital Partners LLP',
        assertion_id: 'a_prem_represents_01',
        epistemic_class: 'DECLARED',
        confidence: 1.0,
      },
      {
        subject_label: 'Halveston Group — Series C',
        predicate: 'ADVERSE_TO',
        object_label: 'Meridian Capital Partners LLP',
        assertion_id: 'a_prem_adverse_01',
        epistemic_class: 'DECLARED',
        confidence: 1.0,
      },
      {
        subject_label: 'Halveston Group — Series C',
        predicate: 'POTENTIAL_CONFLICT',
        object_label: 'Meridian Capital Partners LLP',
        assertion_id: 'a_inf_conflict_0091',
        epistemic_class: 'INFERRED',
        confidence: 0.95,
      },
    ],
    citations: [
      {
        assertion_id: 'a_prem_represents_01',
        label: 'Thorne Vaux LLP REPRESENTS Meridian Capital Partners LLP',
        epistemic_class: 'DECLARED',
        confidence: 1.0,
      },
      {
        assertion_id: 'a_mentions_meridian_01',
        label: 'Subscription Agreement MENTIONS Meridian Capital Partners LLP',
        epistemic_class: 'EXTRACTED_DET',
        confidence: 0.99,
        document_id: 'doc_8f21a0',
        filename: 'Halveston — Subscription Agreement (executed).pdf',
        page: 3,
        quote: 'Meridian Capital Partners LLP (the "Adverse Party")',
      },
    ],
  },
  tier3: {
    question: 'Which open matters have unbilled work and an adverse party we also act for?',
    tier: 3,
    tier_reason:
      'The entities and relationships came from a graph traversal; the unbilled figures came from the governed metric work_in_progress.',
    answer:
      'One matter meets both tests: Halveston Group — Series C, with 214,800 unbilled, adverse to Meridian Capital Partners LLP, which the firm acts for on MAT-1590.[1]',
    metric_id: 'm_003',
    sql: mockCompiledSql(MOCK_METRICS[2]),
    path: [
      {
        subject_label: 'Halveston Group — Series C',
        predicate: 'ADVERSE_TO',
        object_label: 'Meridian Capital Partners LLP',
        assertion_id: 'a_prem_adverse_01',
        epistemic_class: 'DECLARED',
        confidence: 1.0,
      },
    ],
    citations: [
      {
        assertion_id: 'a_inf_conflict_0091',
        label: 'Potential conflict on MAT-2041',
        epistemic_class: 'INFERRED',
        confidence: 0.95,
      },
    ],
    rows: {
      columns: ['matter_id', 'matter_name', 'wip_value'],
      rows: [['MAT-2041', 'Halveston Group — Series C', 214800]],
    },
  },
  tier4: {
    question: 'What is the average number of days between opening a matter and the first invoice?',
    tier: 4,
    tier_reason:
      'No governed metric covers days-to-first-invoice, so SQL was generated against the real schema and checked by the query firewall before running.',
    answer:
      'The mean interval between a matter opening and its first invoice is 47.3 days across 3,918 matters with at least one invoice.',
    sql: `-- Generated by llm:claude-sonnet-5. Read before relying on this figure.
WITH first_invoice AS (
  SELECT matter_id, MIN(issued_date) AS first_issued
  FROM firm_billing.invoice
  WHERE status = 'ISSUED'
  GROUP BY matter_id
)
SELECT
  AVG(DATE_DIFF('day', m.opened_date, f.first_issued)) AS avg_days_to_first_invoice,
  COUNT(*) AS matters
FROM firm_matters.matter m
JOIN first_invoice f ON f.matter_id = m.matter_id`,
    citations: [],
    rows: {
      columns: ['avg_days_to_first_invoice', 'matters'],
      rows: [[47.3, 3918]],
    },
  },
}

export const MOCK_NEIGHBOURHOOD: Neighbourhood = {
  nodes: [
    { id: 'MAT-2041', label: 'Halveston — Series C', type: 'Matter', matter_id: 'MAT-2041' },
    { id: 'p_halveston', label: 'Halveston Group Ltd', type: 'Party' },
    { id: 'p_meridian', label: 'Meridian Capital Partners LLP', type: 'Party' },
    { id: 'c_firm', label: 'Thorne Vaux LLP', type: 'Counsel' },
    { id: 'c_dalgleish', label: 'Dalgleish & Rowe LLP', type: 'Counsel' },
    { id: 'd_8f21a0', label: 'Subscription Agreement', type: 'Document', matter_id: 'MAT-2041' },
    { id: 'd_5c0f8e', label: 'Board minutes 12 Mar', type: 'Document', matter_id: 'MAT-2041' },
    { id: 'auth_wilkes', label: 'Wilkes v Argonaut Shipping', type: 'Authority' },
    { id: 'auth_cogsa', label: 'Carriage of Goods by Sea Act 1971', type: 'Authority' },
    { id: 'MAT-2088', label: 'Rowe v Castleton Freight', type: 'Matter', matter_id: 'MAT-2088' },
    { id: 'p_castleton', label: 'Castleton Freight Ltd', type: 'Party' },
    { id: 'p_rowe', label: 'Rowe (Claimant)', type: 'Party' },
    { id: 'd_44c9de', label: 'Particulars of Claim', type: 'Document', matter_id: 'MAT-2088' },
    { id: 'court_qb', label: 'King’s Bench Division', type: 'Court' },
    { id: 'dl_northmoor_1', label: 'Appointment deadline', type: 'Deadline', matter_id: 'MAT-1977' },
    { id: 'MAT-1977', label: 'Northmoor Estate', type: 'Matter', matter_id: 'MAT-1977' },
    { id: 'MAT-1590', label: 'Ashcombe — Disposal', type: 'Matter', matter_id: 'MAT-1590' },
    { id: 'cl_indemnity', label: 'Clause 12.4 (indemnity)', type: 'Clause', matter_id: 'MAT-2041' },
    { id: 't_carriage', label: 'Carriage of goods by sea', type: 'Topic' },
    { id: 'p_ashcombe', label: 'Ashcombe Holdings plc', type: 'Party' },
  ],
  // Only MENTIONS edges may be EXTRACTED_DET: a verified quote establishes that words
  // appear and nothing about what they imply. Anything asserting significance —
  // CITES, FILED_IN, OVERRULES — is model-extracted and reviewed.
  edges: [
    { assertion_id: 'e1', source: 'p_halveston', target: 'MAT-2041', predicate: 'PARTY_TO', epistemic_class: 'DECLARED', confidence: 1, review_state: 'AUTO_ASSERTED', governing: true },
    { assertion_id: 'a_prem_adverse_01', source: 'MAT-2041', target: 'p_meridian', predicate: 'ADVERSE_TO', epistemic_class: 'DECLARED', confidence: 1, review_state: 'AUTO_ASSERTED', governing: true },
    { assertion_id: 'a_prem_represents_01', source: 'c_firm', target: 'p_meridian', predicate: 'REPRESENTS', epistemic_class: 'DECLARED', confidence: 1, review_state: 'AUTO_ASSERTED', governing: true },
    { assertion_id: 'a_inf_conflict_0091', source: 'MAT-2041', target: 'p_meridian', predicate: 'POTENTIAL_CONFLICT', epistemic_class: 'INFERRED', confidence: 0.95, review_state: 'PENDING', governing: true },
    { assertion_id: 'e5', source: 'd_8f21a0', target: 'MAT-2041', predicate: 'RELATES_TO_MATTER', epistemic_class: 'DECLARED', confidence: 1, review_state: 'AUTO_ASSERTED', governing: true },
    { assertion_id: 'a_1f90d7ac55be4402', source: 'd_8f21a0', target: 'MAT-2041', predicate: 'SUBJECT_TO_PRIVILEGE', epistemic_class: 'EXTRACTED_MODEL', confidence: 0.64, review_state: 'PENDING', governing: true },
    { assertion_id: 'e7', source: 'd_8f21a0', target: 'cl_indemnity', predicate: 'CONTAINS_CLAUSE', epistemic_class: 'EXTRACTED_MODEL', confidence: 0.79, review_state: 'APPROVED', governing: true },
    { assertion_id: 'a_mentions_meridian_01', source: 'd_8f21a0', target: 'p_meridian', predicate: 'MENTIONS', epistemic_class: 'EXTRACTED_DET', confidence: 0.99, review_state: 'AUTO_ASSERTED', governing: false },
    { assertion_id: 'a_7c41e9b2f0a34d18', source: 'p_halveston', target: 'p_meridian', predicate: 'ADVERSE_TO', epistemic_class: 'EXTRACTED_MODEL', confidence: 0.71, review_state: 'PENDING', governing: true },
    { assertion_id: 'a_c17f4e6b8a2d4915', source: 'd_5c0f8e', target: 'p_ashcombe', predicate: 'MENTIONS', epistemic_class: 'EXTRACTED_DET', confidence: 0.99, review_state: 'AUTO_ASSERTED', governing: false },
    { assertion_id: 'e10', source: 'd_5c0f8e', target: 'MAT-2041', predicate: 'RELATES_TO_MATTER', epistemic_class: 'DECLARED', confidence: 1, review_state: 'AUTO_ASSERTED', governing: true },
    { assertion_id: 'a_mentions_wilkes_01', source: 'd_44c9de', target: 'auth_wilkes', predicate: 'MENTIONS', epistemic_class: 'EXTRACTED_DET', confidence: 0.99, review_state: 'AUTO_ASSERTED', governing: false },
    { assertion_id: 'a_cite_wilkes_01', source: 'd_44c9de', target: 'auth_wilkes', predicate: 'CITES', epistemic_class: 'EXTRACTED_MODEL', confidence: 0.79, review_state: 'APPROVED', governing: true },
    { assertion_id: 'e12', source: 'd_44c9de', target: 'auth_cogsa', predicate: 'CITES', epistemic_class: 'EXTRACTED_MODEL', confidence: 0.79, review_state: 'APPROVED', governing: true },
    { assertion_id: 'a_5da0417ee2cb49aa', source: 'd_44c9de', target: 'auth_wilkes', predicate: 'DISTINGUISHES', epistemic_class: 'EXTRACTED_MODEL', confidence: 0.79, review_state: 'PENDING', governing: true },
    { assertion_id: 'e14', source: 'd_44c9de', target: 'court_qb', predicate: 'FILED_IN', epistemic_class: 'EXTRACTED_MODEL', confidence: 0.79, review_state: 'APPROVED', governing: true },
    { assertion_id: 'e15', source: 'p_rowe', target: 'MAT-2088', predicate: 'PARTY_TO', epistemic_class: 'DECLARED', confidence: 1, review_state: 'AUTO_ASSERTED', governing: true },
    { assertion_id: 'e16', source: 'p_castleton', target: 'MAT-2088', predicate: 'PARTY_TO', epistemic_class: 'DECLARED', confidence: 1, review_state: 'AUTO_ASSERTED', governing: true },
    { assertion_id: 'a_b3e8225c91df4770', source: 'c_dalgleish', target: 'p_castleton', predicate: 'REPRESENTS', epistemic_class: 'EXTRACTED_MODEL', confidence: 0.58, review_state: 'PENDING', governing: true },
    { assertion_id: 'e18', source: 'MAT-2088', target: 'auth_cogsa', predicate: 'GOVERNED_BY', epistemic_class: 'DECLARED', confidence: 1, review_state: 'AUTO_ASSERTED', governing: true },
    { assertion_id: 'a_902ab7f14c0e4e63', source: 'dl_northmoor_1', target: 'MAT-1977', predicate: 'DEADLINE_FOR', epistemic_class: 'EXTRACTED_MODEL', confidence: 0.45, review_state: 'PENDING', governing: true },
    { assertion_id: 'a_pred_link_01', source: 'p_castleton', target: 't_carriage', predicate: 'CONCERNS_TOPIC', epistemic_class: 'PREDICTED', confidence: 0.52, review_state: 'PENDING', governing: false },
    { assertion_id: 'e21', source: 'MAT-2088', target: 't_carriage', predicate: 'CONCERNS_TOPIC', epistemic_class: 'EXTRACTED_MODEL', confidence: 0.79, review_state: 'APPROVED', governing: false },
    { assertion_id: 'e22', source: 'p_meridian', target: 'MAT-1590', predicate: 'PARTY_TO', epistemic_class: 'DECLARED', confidence: 1, review_state: 'AUTO_ASSERTED', governing: true },
    { assertion_id: 'e23', source: 'auth_cogsa', target: 'auth_wilkes', predicate: 'OVERRULES', epistemic_class: 'EXTRACTED_MODEL', confidence: 0.79, review_state: 'APPROVED', governing: true },
  ],
}

const MOCK_ACTIVITY: ActivityEvent[] = [
  { event_id: 'ev1', timestamp: iso(0, 14), actor: 'a.mensah@thornevaux.example', action: 'Approved', detail: 'Particulars of Claim DISTINGUISHES Wilkes v Argonaut Shipping', epistemic_class: 'EXTRACTED_MODEL' },
  { event_id: 'ev2', timestamp: iso(0, 13), actor: 'llm:claude-sonnet-5', action: 'Staged 9 claims', detail: 'Halveston — Subscription Agreement (executed).pdf', epistemic_class: 'EXTRACTED_MODEL' },
  { event_id: 'ev3', timestamp: iso(0, 12), actor: 'rule:conflict_check@v1', action: 'Inferred', detail: 'Potential conflict: MAT-2041 and Meridian Capital Partners LLP', epistemic_class: 'INFERRED' },
  { event_id: 'ev4', timestamp: iso(0, 11), actor: VERIFIED, action: 'Asserted 14 verified quotes', detail: 'Rowe — Particulars of Claim.pdf', epistemic_class: 'EXTRACTED_DET' },
  { event_id: 'ev5', timestamp: iso(1, 16), actor: 'k.iyer@thornevaux.example', action: 'Rejected', detail: 'Dalgleish & Rowe LLP REPRESENTS Rowe — parties inverted', epistemic_class: 'EXTRACTED_MODEL' },
  { event_id: 'ev6', timestamp: iso(1, 9), actor: 'glue:catalog_scan@v2', action: 'Declared 18 tables', detail: 'firm_matters, firm_billing', epistemic_class: 'DECLARED' },
  { event_id: 'ev7', timestamp: iso(2, 15), actor: 'system', action: 'Ingest failed', detail: 'Northmoor — Trust Deed 1994 (scanned).pdf — no text layer', epistemic_class: null },
  { event_id: 'ev8', timestamp: iso(2, 10), actor: 'r.okonjo@thornevaux.example', action: 'Approved metric', detail: 'fees_billed v4', epistemic_class: null },
]

export const MOCK_DASHBOARD: DashboardStats = {
  assertions_by_class: {
    DECLARED: 8412,
    EXTRACTED_DET: 3106,
    EXTRACTED_MODEL: 942,
    INFERRED: 118,
    PREDICTED: 269,
  },
  pending_review: 25,
  documents_by_state: {
    LIVE: 182,
    PENDING_REVIEW: 6,
    EXTRACTING: 2,
    EMBEDDING: 1,
    PARSE_FAILED: 1,
    REGISTERED: 3,
  },
  matters: 4,
  metrics: { total: 4, approved: 3 },
  recent_activity: MOCK_ACTIVITY,
}

export const MOCK_SETTINGS: TenantSettings = {
  tenant_id: 'demo-firm',
  name: 'Thorne Vaux LLP',
  ontology_domain: 'legal',
  min_confidence: 0.8,
  block_ungoverned_queries: false,
  extraction_model: 'anthropic.claude-sonnet-5',
  synthesis_model: 'anthropic.claude-sonnet-5',
  embedding_model: 'amazon.titan-embed-text-v2',
  available_models: [
    { id: 'anthropic.claude-sonnet-5', label: 'Claude Sonnet 5' },
    { id: 'anthropic.claude-haiku-4-5', label: 'Claude Haiku 4.5' },
    { id: 'anthropic.claude-opus-4-5', label: 'Claude Opus 4.5' },
  ],
  available_domains: ['legal', 'healthcare'],
}

/** Mirrors ontologies/legal.yaml. The real endpoint serves this from the pack. */
export const MOCK_ONTOLOGY: Ontology = {
  domain: 'legal',
  version: 1,
  entity_types: [
    { id: 'Matter', label: 'Matter', description: 'An engagement — a case, transaction, or advisory instruction.', help: 'The organising unit of legal work. Most other entities hang off a Matter.' },
    { id: 'Party', label: 'Party', description: 'A person or organisation with an interest in a matter.', help: 'Clients, counterparties, and third parties are all Parties. The role is on the edge, not the node — the same company can be a client on one matter and adverse on another.' },
    { id: 'Counsel', label: 'Counsel', description: 'A lawyer or firm acting on a matter.', help: 'Includes internal fee earners and external firms.' },
    { id: 'Document', label: 'Document', description: 'A filing, contract, opinion, or correspondence item.', help: 'Documents are immutable in S3. What you see here is derived metadata plus extracted assertions.' },
    { id: 'Authority', label: 'Authority', description: 'A cited case, statute, or regulation.', help: 'The words of a citation are quoted from the document and checked; whether the document relies on the authority is a judgement, so it is reviewed.' },
    { id: 'Court', label: 'Court', description: 'A forum or tribunal.', help: null },
    { id: 'Deadline', label: 'Deadline', description: 'A date-bound obligation.', help: 'Limitation periods and filing deadlines. Governing predicates, because a missed one is negligence.' },
    { id: 'Clause', label: 'Clause', description: 'A provision within a contract.', help: null },
  ],
  governing_predicates: [
    { id: 'REPRESENTS', label: 'represents', description: 'Counsel acts for this party.', governing: true, domain: ['Counsel'], range: ['Party'], help: 'Half of every conflict check. Kept closed so `acts_for` and `is_counsel_to` can never fragment it.' },
    { id: 'ADVERSE_TO', label: 'adverse to', description: 'Positioned against this party.', governing: true, domain: ['Party', 'Matter'], range: ['Party'], help: 'The other half of a conflict check. If this predicate fragments, conflict checks silently under-report.', symmetric: true },
    { id: 'PARTY_TO', label: 'party to', description: 'Has an interest in this matter.', governing: true, domain: ['Party'], range: ['Matter'], help: null },
    { id: 'CITES', label: 'cites', description: 'Refers to this authority.', governing: true, domain: ['Document', 'Authority'], range: ['Authority'], help: 'Implies reliance, which finding the words on the page does not establish — "the court declined to follow Brown" names Brown without relying on it. So this is always reviewed.' },
    { id: 'OVERRULES', label: 'overrules', description: 'Displaces the cited authority.', governing: true, domain: ['Authority'], range: ['Authority'], help: 'Governing because relying on overruled authority is a substantive error.' },
    { id: 'DISTINGUISHES', label: 'distinguishes', description: 'Argues the cited authority does not apply.', governing: true, domain: ['Authority', 'Document'], range: ['Authority'], help: null },
    { id: 'FILED_IN', label: 'filed in', description: 'Lodged with this forum.', governing: true, domain: ['Document', 'Matter'], range: ['Court'], help: null },
    { id: 'DEADLINE_FOR', label: 'deadline for', description: 'A date-bound obligation on this matter.', governing: true, domain: ['Deadline'], range: ['Matter', 'Document'], help: null },
    { id: 'SUBJECT_TO_PRIVILEGE', label: 'subject to privilege', description: 'Privileged material on this matter.', governing: true, domain: ['Document', 'Clause'], range: ['Matter'], help: 'Drives redaction and disclosure. Never model-extracted without review.' },
    { id: 'SUPERSEDES', label: 'supersedes', description: 'Replaces the earlier instrument.', governing: true, domain: ['Document', 'Clause'], range: ['Document', 'Clause'], help: null },
    { id: 'GOVERNED_BY', label: 'governed by', description: 'Subject to this law or jurisdiction.', governing: true, domain: ['Matter', 'Document', 'Clause'], range: ['Authority'], help: null },
    { id: 'CONTAINS_CLAUSE', label: 'contains clause', description: 'This document includes the provision.', governing: true, domain: ['Document'], range: ['Clause'], help: null },
    { id: 'RELATES_TO_MATTER', label: 'relates to matter', description: 'Belongs to this matter.', governing: true, domain: ['Document', 'Party', 'Deadline'], range: ['Matter'], help: null },
    { id: 'INSTRUCTS', label: 'instructs', description: 'Retains this counsel.', governing: true, domain: ['Party'], range: ['Counsel'], help: null },
  ],
  descriptive_predicates: [
    { id: 'CONCERNS_TOPIC', label: 'concerns topic', description: 'Subject-matter tag.', governing: false, domain: [], range: [], help: null },
    { id: 'IN_INDUSTRY', label: 'in industry', description: 'Sector classification.', governing: false, domain: [], range: [], help: null },
    { id: 'HAS_DEAL_TYPE', label: 'has deal type', description: 'Transaction category.', governing: false, domain: [], range: [], help: null },
    { id: 'MENTIONS', label: 'mentions', description: 'Named in the text without a stronger relationship being established.', governing: false, domain: [], range: [], help: 'The only relationship a verified quote can support on its own: the words are on the page, and nothing is read into them.' },
  ],
  rules: [
    {
      id: 'conflict_check',
      version: 'v1',
      description: 'Flag a potential conflict where the firm both represents and opposes a party.',
      when: ['(c:Counsel)-[:REPRESENTS]->(p:Party)', '(m:Matter)-[:ADVERSE_TO]->(p:Party)'],
      then: '(m)-[:POTENTIAL_CONFLICT]->(p)',
      min_premise_class: 'EXTRACTED_DET',
      help: 'Fires only on facts declared by a system of record or confirmed by a check — a conflict flag resting on an unreviewed guess would be worse than none.',
    },
    {
      id: 'authority_stale',
      version: 'v1',
      description: 'Mark reliance on authority that has since been overruled.',
      when: ['(d:Document)-[:CITES]->(a:Authority)', '(x:Authority)-[:OVERRULES]->(a:Authority)'],
      then: '(d)-[:RELIES_ON_STALE_AUTHORITY]->(a)',
      min_premise_class: 'EXTRACTED_DET',
      help: null,
    },
  ],
}
