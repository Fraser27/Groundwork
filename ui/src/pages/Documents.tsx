import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  api,
  type DocumentDetail,
  type DocumentSummary,
  type IngestState,
  type Matter,
} from '../api'
import { getTenantId } from '../auth'
import { HELP, ingestPhase } from '../epistemic'
import { fallback, mockDocumentDetail, MOCK_DOCUMENTS, MOCK_MATTERS_RESPONSE, MOCK_SETTINGS } from '../mocks'
import ConfidenceBar from '../components/ConfidenceBar'
import DocumentViewer from '../components/DocumentViewer'
import EpistemicBadge from '../components/EpistemicBadge'
import FieldHelp from '../components/FieldHelp'
import { SourceSpan } from '../components/ProvenancePanel'
import { EmptyState, IngestPill, MockFlag, Pipeline, Spinner, Toast } from '../components/Shared'
import { fmtBytes, fmtDateTime, fmtNum } from '../format'

/** The user asked for 30s. On-demand refresh is the Refresh button and every upload. */
const POLL_INTERVAL_MS = 30_000

/** Stop tracking an ingest we have heard nothing about for this long. */
const STALE_AFTER_MS = 15 * 60_000

const TERMINAL_STATES = new Set<IngestState>([
  'LIVE',
  'PENDING_REVIEW',
  'FETCH_FAILED',
  'PARSE_FAILED',
  'CHUNK_FAILED',
  'EXTRACT_FAILED',
  'EMBED_FAILED',
])

interface ActiveIngest {
  filename: string
  state: IngestState
  at: number
  document_id?: string
  reason?: string | null
}

export default function Documents() {
  const tenant = getTenantId()
  const [docs, setDocs] = useState<DocumentSummary[]>([])
  const [matters, setMatters] = useState<Matter[]>([])
  const [floor, setFloor] = useState(0.8)
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState('')
  const [stateFilter, setStateFilter] = useState('__all__')
  const [matterFilter, setMatterFilter] = useState('__all__')
  const [detail, setDetail] = useState<DocumentDetail | null>(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const [uploadMatter, setUploadMatter] = useState('')
  const [viewing, setViewing] = useState<{ page: number; quote: string | null } | null>(null)
  const [toast, setToast] = useState<{ msg: string; type: string } | null>(null)
  const [progress, setProgress] = useState<{ name: string; fraction: number } | null>(null)
  const [active, setActive] = useState<ActiveIngest[]>([])
  const [liveEvents, setLiveEvents] = useState(false)
  const fileInput = useRef<HTMLInputElement>(null)

  const showToast = (msg: string, type = 'success') => {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 4000)
  }

  const load = () => {
    Promise.all([
      fallback(api.listDocuments(tenant), MOCK_DOCUMENTS),
      fallback(api.listMatters(tenant), MOCK_MATTERS_RESPONSE),
      fallback(api.getSettings(tenant), MOCK_SETTINGS),
    ])
      .then(([d, m, s]) => {
        setDocs(d)
        setMatters(m.matters)
        setFloor(s.min_confidence)
      })
      .catch(console.error)
      .finally(() => setLoading(false))
  }

  useEffect(load, [tenant])

  /**
   * Live progress, with a poll underneath it.
   *
   * Both, not either. The socket is served by a single task, so a client can be connected
   * to a task that is not running its ingest and never receive an event. The poll is
   * therefore the correctness guarantee and the socket only removes the wait — which is
   * why `liveEvents` changes a label and nothing else.
   */
  useEffect(() => {
    const close = api.subscribeIngestEvents(
      tenant,
      (event) => {
        setLiveEvents(true)
        setActive((prev) =>
          prev.map((a) =>
            a.document_id === event.document_id || a.state === 'REGISTERED'
              ? { ...a, document_id: event.document_id, state: event.state, reason: event.reason }
              : a,
          ),
        )
        if (TERMINAL_STATES.has(event.state)) load()
      },
      () => setLiveEvents(false),
    )
    return close
  }, [tenant])

  useEffect(() => {
    if (active.length === 0) return
    const id = setInterval(() => {
      load()
      // Drop anything finished or long enough ago that its absence means it is done and
      // already in the table. Without this the tracker grows for the whole session.
      setActive((prev) =>
        prev.filter((a) => !TERMINAL_STATES.has(a.state) && Date.now() - a.at < STALE_AFTER_MS),
      )
    }, POLL_INTERVAL_MS)
    return () => clearInterval(id)
  }, [active.length, tenant])

  const filtered = useMemo(() => {
    let out = docs
    if (filter.trim()) {
      const q = filter.toLowerCase()
      out = out.filter(
        (d) => d.filename.toLowerCase().includes(q) || d.document_id.toLowerCase().includes(q),
      )
    }
    if (stateFilter !== '__all__') out = out.filter((d) => ingestPhase(d.state) === stateFilter)
    if (matterFilter !== '__all__') out = out.filter((d) => d.matter_id === matterFilter)
    return out
  }, [docs, filter, stateFilter, matterFilter])

  const openDetail = async (id: string) => {
    setDetailLoading(true)
    try {
      setDetail(await fallback(api.getDocument(tenant, id), mockDocumentDetail(id)))
    } catch (e) {
      showToast((e as Error).message, 'error')
    } finally {
      setDetailLoading(false)
    }
  }

  /**
   * Upload direct to S3, then track ingestion separately.
   *
   * The file does not pass through the API, so a large bundle is no longer bounded by
   * the 60s origin timeout. The upload finishing does *not* mean ingestion finished —
   * that is what `active` and the poll below are for.
   */
  const upload = async (files: FileList | null) => {
    if (!files || files.length === 0) return
    setUploading(true)
    const list = Array.from(files)
    try {
      for (const f of list) {
        setProgress({ name: f.name, fraction: 0 })
        await api.uploadViaPresignedPost(tenant, f, uploadMatter || undefined, (fraction) =>
          setProgress({ name: f.name, fraction }),
        )
        setActive((prev) => [...prev, { filename: f.name, state: 'REGISTERED', at: Date.now() }])
      }
      showToast(
        `${list.length} file${list.length === 1 ? '' : 's'} uploaded. Reading the pages happens in the background, progress appears below.`,
      )
      load()
    } catch (e) {
      showToast((e as Error).message.replace(/^\d+:\s*/, ''), 'error')
    } finally {
      setUploading(false)
      setProgress(null)
      if (fileInput.current) fileInput.current.value = ''
    }
  }

  const failing = docs.filter((d) => ingestPhase(d.state) === 'failed')

  if (loading) return <Spinner />

  return (
    <>
      <div className="page-header">
        <div className="page-header-row">
          <div>
            <h2>Documents</h2>
            <p>
              Uploads are stored unaltered and never overwritten; everything else is a derived index
              that can be rebuilt. A bad extraction run is therefore never a data-loss event.
            </p>
          </div>
          <MockFlag />
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <h3>
            Upload
            <FieldHelp text={HELP.ingestState} />
          </h3>
          <div className="toolbar-field" style={{ marginBottom: 0 }}>
            <label>
              Attach to matter
              <FieldHelp text={HELP.matterWall} align="right" />
            </label>
            <select value={uploadMatter} onChange={(e) => setUploadMatter(e.target.value)}>
              <option value="">Unassigned</option>
              {matters
                .filter((m) => !m.walled)
                .map((m) => (
                  <option key={m.matter_id} value={m.matter_id}>
                    {m.matter_id} — {m.name}
                  </option>
                ))}
            </select>
          </div>
        </div>
        <div
          className={`dropzone${dragOver ? ' over' : ''}`}
          onDragOver={(e) => {
            e.preventDefault()
            setDragOver(true)
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragOver(false)
            upload(e.dataTransfer.files)
          }}
          onClick={() => fileInput.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') fileInput.current?.click()
          }}
        >
          <strong>{uploading ? 'Uploading…' : 'Drop files here, or click to choose'}</strong>
          PDF, DOCX or TIFF. Verbatim text becomes searchable as soon as it is parsed; anything a
          model reads into it is held back until someone reviews it.
          <input
            ref={fileInput}
            type="file"
            multiple
            accept=".pdf,.docx,.doc,.tif,.tiff,.txt"
            style={{ display: 'none' }}
            onChange={(e) => upload(e.target.files)}
          />
        </div>

        {progress && (
          <div className="upload-progress">
            <div className="upload-progress-row">
              <span className="mono">{progress.name}</span>
              <span>{Math.round(progress.fraction * 100)}%</span>
            </div>
            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${progress.fraction * 100}%` }} />
            </div>
          </div>
        )}
      </div>

      {active.length > 0 && (
        <div className="card" style={{ marginTop: 16 }}>
          <div className="card-header">
            <h3>Reading pages</h3>
            <span className="muted small">
              {liveEvents ? 'Live' : `Checking every ${POLL_INTERVAL_MS / 1000}s`}
              {' · '}
              <button className="link-button" onClick={load}>
                Refresh now
              </button>
            </span>
          </div>
          <table className="data-table">
            <thead>
              <tr>
                <th>File</th>
                <th>Stage</th>
                <th>Detail</th>
              </tr>
            </thead>
            <tbody>
              {active.map((a, i) => (
                <tr key={`${a.filename}-${i}`}>
                  <td className="mono">{a.filename}</td>
                  <td>
                    <IngestPill state={a.state} />
                  </td>
                  <td className="muted">
                    {a.reason ||
                      (a.state === 'PARSING'
                        ? 'One model call per page, several at a time.'
                        : '')}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {failing.length > 0 && (
        <div className="banner banner-error" style={{ marginTop: 16 }}>
          <span>
            <strong>
              {failing.length} document{failing.length === 1 ? '' : 's'} failed to ingest.
            </strong>{' '}
            Nothing from a failed document has entered the graph. The original file is untouched, so
            re-running the pipeline is safe.
          </span>
        </div>
      )}

      <div className="toolbar" style={{ marginTop: 16 }}>
        <div className="toolbar-field" style={{ flex: 1, minWidth: 240 }}>
          <label>Search</label>
          <input
            placeholder="Filter by filename or id…"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            style={{ width: '100%' }}
          />
        </div>
        <div className="toolbar-field">
          <label>
            Stage
            <FieldHelp text={HELP.ingestState} />
          </label>
          <select value={stateFilter} onChange={(e) => setStateFilter(e.target.value)}>
            <option value="__all__">All stages</option>
            <option value="pending">Registered</option>
            <option value="running">In progress</option>
            <option value="review">Awaiting review</option>
            <option value="live">Live</option>
            <option value="failed">Failed</option>
          </select>
        </div>
        <div className="toolbar-field">
          <label>Matter</label>
          <select value={matterFilter} onChange={(e) => setMatterFilter(e.target.value)}>
            <option value="__all__">All matters</option>
            {matters
              .filter((m) => !m.walled)
              .map((m) => (
                <option key={m.matter_id} value={m.matter_id}>
                  {m.matter_id}
                </option>
              ))}
          </select>
        </div>
        <div className="toolbar-field toolbar-spacer">
          <label>&nbsp;</label>
          <span className="search-count">
            {filtered.length} of {docs.length}
          </span>
        </div>
      </div>

      <div className="card">
        <table className="data-table data-table-hover">
          <thead>
            <tr>
              <th>File</th>
              <th>Matter</th>
              <th>
                State
                <FieldHelp text={HELP.ingestState} />
              </th>
              <th className="num">Pages</th>
              <th className="num">Size</th>
              <th className="num">Facts</th>
              <th className="num">
                Pending
                <FieldHelp text={HELP.reviewState} align="right" />
              </th>
              <th>Uploaded</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((d) => (
              <tr key={d.document_id} onClick={() => openDetail(d.document_id)}>
                <td>
                  <strong>{d.filename}</strong>
                  <div className="dim" style={{ fontSize: 11.5 }}>
                    <code>{d.document_id}</code>
                  </div>
                </td>
                <td className="nowrap dim">{d.matter_id || '-'}</td>
                <td>
                  <IngestPill state={d.state} />
                  {d.error && (
                    <div style={{ fontSize: 11, color: 'var(--red)', marginTop: 4, maxWidth: 260 }}>
                      {d.error}
                    </div>
                  )}
                </td>
                <td className="num">{fmtNum(d.page_count)}</td>
                <td className="num nowrap">{fmtBytes(d.size_bytes)}</td>
                <td className="num">{fmtNum(d.assertion_count)}</td>
                <td className="num">
                  {d.pending_review_count > 0 ? (
                    <span className="tag tag-orange">{d.pending_review_count}</span>
                  ) : (
                    '-'
                  )}
                </td>
                <td className="nowrap dim">{fmtDateTime(d.uploaded_at)}</td>
              </tr>
            ))}
            {filtered.length === 0 && (
              <tr>
                <td colSpan={8}>
                  <EmptyState title="No documents match">
                    Upload a file above to start the pipeline.
                  </EmptyState>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {(detail || detailLoading) && (
        <div className="modal-overlay" onClick={() => setDetail(null)}>
          <div className="modal modal-wide" onClick={(e) => e.stopPropagation()}>
            {detailLoading || !detail ? (
              <Spinner />
            ) : (
              <>
                <h3>{detail.filename}</h3>
                <p className="modal-sub">
                  <code>{detail.s3_uri}</code>
                </p>

                <div className="prov-section-title">
                  Ingest pipeline
                  <FieldHelp text={HELP.ingestState} />
                </div>
                <Pipeline state={detail.state} />

                {detail.error && (
                  <div className="banner banner-error">
                    <span>
                      <strong>Ingest failed.</strong> {detail.error}
                    </span>
                  </div>
                )}

                <div className="detail-grid-3" style={{ marginTop: 16 }}>
                  <div className="detail-field">
                    <div className="label">State</div>
                    <div className="value">
                      <IngestPill state={detail.state} />
                    </div>
                  </div>
                  <div className="detail-field">
                    <div className="label">Matter</div>
                    <div className="value">{detail.matter_id || 'Unassigned'}</div>
                  </div>
                  <div className="detail-field">
                    <div className="label">Pages</div>
                    <div className="value">{fmtNum(detail.page_count)}</div>
                  </div>
                  <div className="detail-field">
                    <div className="label">Size</div>
                    <div className="value">{fmtBytes(detail.size_bytes)}</div>
                  </div>
                  <div className="detail-field">
                    <div className="label">
                      Content hash
                      <FieldHelp text="Taken on upload. It is how a re-upload of the same file is recognised as a no-op rather than a second copy." />
                    </div>
                    <div className="value">
                      <code style={{ fontSize: 11 }}>{detail.content_sha256?.slice(0, 20)}</code>
                    </div>
                  </div>
                  <div className="detail-field">
                    <div className="label">Uploaded</div>
                    <div className="value">{fmtDateTime(detail.uploaded_at)}</div>
                  </div>
                </div>

                <div className="prov-section-title" style={{ marginTop: 6 }}>
                  Transitions
                </div>
                <div className="timeline">
                  {detail.timeline.map((t, i) => (
                    <div className="timeline-row" key={`${t.state}-${i}`}>
                      <span className="timeline-when">{fmtDateTime(t.at)}</span>
                      <span>
                        <IngestPill state={t.state} />
                      </span>
                      <span className="timeline-detail">{t.detail || '-'}</span>
                    </div>
                  ))}
                </div>

                <div className="prov-section-title" style={{ marginTop: 20 }}>
                  Facts extracted from this document
                  <FieldHelp text={HELP.epistemicClass} />
                </div>
                {detail.assertions.length === 0 ? (
                  <p className="card-note">
                    Nothing has been extracted yet. A quote the system confirms is on the page it
                    names is asserted directly; anything read into the text appears here as pending.
                  </p>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                    {detail.assertions.map((a) => (
                      <div
                        key={a.assertion_id}
                        style={{
                          border: '1px solid var(--border)',
                          borderRadius: 'var(--radius-sm)',
                          padding: '11px 13px',
                        }}
                      >
                        <div
                          style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            gap: 10,
                            flexWrap: 'wrap',
                            marginBottom: 8,
                          }}
                        >
                          <span>
                            <strong>{a.subject_label || a.subject_id}</strong>{' '}
                            <span className="prov-pred">{a.predicate}</span>{' '}
                            <strong>{a.object_label || a.object_id}</strong>
                          </span>
                          <span style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                            <EpistemicBadge
                              epistemicClass={a.epistemic_class}
                              size="sm"
                              tipAlign="right"
                              tipPlacement="above"
                            />
                            <ConfidenceBar value={a.confidence} floor={floor} width={56} />
                          </span>
                        </div>
                        {a.source_locator.quote && <SourceSpan text={a.source_locator.quote} />}
                        <div className="prov-locator">
                          {a.source_locator.page != null && (
                            <span>page {a.source_locator.page}</span>
                          )}
                          <span>{a.method}</span>
                          <span>{a.review_state.replace('_', '-').toLowerCase()}</span>
                        </div>
                        {a.source_locator.page != null && (
                          <button
                            className="btn btn-ghost btn-sm"
                            style={{ marginTop: 8 }}
                            onClick={() =>
                              setViewing({
                                page: a.source_locator.page as number,
                                quote: a.source_locator.quote ?? null,
                              })
                            }
                          >
                            Open at page {a.source_locator.page}
                          </button>
                        )}
                      </div>
                    ))}
                  </div>
                )}

                <div className="modal-actions">
                  <button
                    className="btn btn-ghost btn-sm"
                    onClick={() => setViewing({ page: 1, quote: null })}
                  >
                    Open the document
                  </button>
                  {detail.pending_review_count > 0 && (
                    <Link to="/review" className="btn btn-primary btn-sm">
                      Review {detail.pending_review_count} pending
                    </Link>
                  )}
                  <button className="btn btn-ghost" onClick={() => setDetail(null)}>
                    Close
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {viewing && detail && (
        <DocumentViewer
          tenant={tenant}
          documentId={detail.document_id}
          filename={detail.filename}
          page={viewing.page}
          quote={viewing.quote}
          onClose={() => setViewing(null)}
        />
      )}

      <Toast toast={toast} />
    </>
  )
}
