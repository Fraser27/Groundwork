/**
 * DocumentViewer — opens the original file at the cited page.
 *
 * The browser's own PDF viewer does the rendering, reached through `#page=N`. That
 * fragment is honoured by Chrome, Safari, Firefox and Edge, so a citation resolves
 * with no PDF library shipped to the client and no coordinate mapping.
 *
 * The quote is shown above the file rather than highlighted inside it: the native
 * viewer exposes no highlight API, and a lawyer can read it off or search for it.
 *
 * Links are presigned and short-lived, so one is fetched per open and refetched when
 * it lapses — never cached.
 */

import { useEffect, useState } from 'react'
import { api, type DocumentDownload } from '../api'
import { HELP } from '../epistemic'
import FieldHelp from './FieldHelp'

/** Whether the browser will render this inline. Anything else is offered as a download. */
function isPdf(filename: string, contentType?: string | null): boolean {
  if (contentType) return contentType.toLowerCase().includes('pdf')
  return filename.toLowerCase().endsWith('.pdf')
}

function fileKind(filename: string): string {
  const dot = filename.lastIndexOf('.')
  return dot > 0 ? filename.slice(dot + 1).toUpperCase() : 'file'
}

type Outcome =
  | { status: 'ready'; link: DocumentDownload }
  | { status: 'unavailable'; reason: string; detail?: string }

function describeFailure(e: Error): Outcome {
  return {
    status: 'unavailable',
    reason: /403|forbidden|denied/i.test(e.message)
      ? 'You do not have access to this file.'
      : /404|not found/i.test(e.message)
        ? 'The file is no longer in storage.'
        : 'The file could not be retrieved.',
    detail: e.message.replace(/^\d+:\s*/, '') || undefined,
  }
}

export default function DocumentViewer({
  tenant,
  documentId,
  filename,
  page,
  quote,
  onClose,
}: {
  tenant: string
  documentId: string
  /** Shown while the link is still being fetched, so the panel is never nameless. */
  filename: string
  page: number
  quote?: string | null
  onClose: () => void
}) {
  // The result is tagged with the request it answers, so a result for a previous
  // document or a previous attempt reads as "still loading" rather than being shown.
  const [attempt, setAttempt] = useState(0)
  const [loaded, setLoaded] = useState<{ key: string; outcome: Outcome } | null>(null)
  /** Which link lapsed, not merely that one did — a fresh link must not inherit it. */
  const [expiredKey, setExpiredKey] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)

  const key = `${tenant}/${documentId}#${attempt}`
  const outcome = loaded?.key === key ? loaded.outcome : null

  const load = () => setAttempt((n) => n + 1)

  useEffect(() => {
    let live = true
    api
      .documentDownload(tenant, documentId)
      .then((link) => {
        if (!live) return
        setLoaded({
          key,
          outcome: link.download_url
            ? { status: 'ready', link }
            : {
                status: 'unavailable',
                reason: 'The system did not return a link to this file.',
                detail:
                  'The record of the file is here, but the file itself could not be reached. Its storage may not be configured for this firm. The quote and page below are still the citation.',
              },
        })
      })
      .catch((e: Error) => {
        if (live) setLoaded({ key, outcome: describeFailure(e) })
      })
    return () => {
      live = false
    }
  }, [tenant, documentId, key])

  // Capture phase and stop propagation: opened from inside the provenance modal, one
  // Escape must close the viewer only, not both layers at once.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      e.stopPropagation()
      onClose()
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [onClose])

  // A presigned link stops working mid-read with no visible change, so the panel
  // watches the clock and says so rather than showing a dead frame.
  const expiresAt = outcome?.status === 'ready' ? outcome.link.expires_at : null
  const expired = expiredKey === key
  useEffect(() => {
    if (!expiresAt) return
    const ms = new Date(expiresAt).getTime() - Date.now()
    const t = setTimeout(() => setExpiredKey(key), Math.max(0, ms))
    return () => clearTimeout(t)
  }, [expiresAt, key])

  const link = outcome?.status === 'ready' ? outcome.link : null
  const name = link?.filename || filename
  const renderable = link ? isPdf(name, link.content_type) : false
  const src = link?.download_url ? `${link.download_url}#page=${page}` : null

  const copyQuote = () => {
    if (!quote) return
    navigator.clipboard.writeText(quote).then(
      () => {
        setCopied(true)
        setTimeout(() => setCopied(false), 2000)
      },
      () => setCopied(false),
    )
  }

  return (
    <div
      className="modal-overlay doc-viewer-overlay"
      onClick={(e) => {
        // Nested inside another overlay: without this, one backdrop click dismisses
        // the provenance modal underneath as well.
        e.stopPropagation()
        onClose()
      }}
    >
      <div className="doc-viewer" onClick={(e) => e.stopPropagation()}>
        <div className="doc-viewer-head">
          <div className="doc-viewer-title">
            <h4>{name}</h4>
            <div className="doc-viewer-sub">
              Page {page}
              {link?.page_count ? ` of ${link.page_count}` : ''}
              <FieldHelp title="How to check this" text={HELP.pageCitation} />
            </div>
          </div>
          <div className="doc-viewer-actions">
            {src && !expired && (
              <a className="btn btn-ghost btn-sm" href={src} target="_blank" rel="noreferrer">
                Open in a new tab
              </a>
            )}
            <button className="btn btn-ghost btn-sm" onClick={onClose} aria-label="Close">
              &#x2715;
            </button>
          </div>
        </div>

        {quote && (
          <div className="doc-viewer-quote">
            <div className="prov-section-title">
              The words cited
              <FieldHelp text={HELP.quote} />
            </div>
            <blockquote className="prov-quote">
              <mark>{quote}</mark>
            </blockquote>
            <div className="doc-viewer-quote-foot">
              <button className="btn btn-ghost btn-sm" onClick={copyQuote}>
                {copied ? 'Copied' : 'Copy the quote'}
              </button>
              <span className="card-note">
                Search for it on page {page} of the file to confirm the citation.
              </span>
            </div>
          </div>
        )}

        <div className="doc-viewer-body">
          {!outcome && (
            <div className="doc-viewer-message">
              <div className="spinner" />
              <p className="card-note">Getting a link to the file.</p>
            </div>
          )}

          {outcome?.status === 'unavailable' && (
            <div className="doc-viewer-message">
              <strong>{outcome.reason}</strong>
              {outcome.detail && <p className="card-note">{outcome.detail}</p>}
              <button className="btn btn-ghost btn-sm" onClick={load}>
                Try again
              </button>
            </div>
          )}

          {link && expired && (
            <div className="doc-viewer-message">
              <strong>The link to this file has expired.</strong>
              <p className="card-note">
                Links are deliberately short-lived, so one is issued each time you open a
                document rather than stored. Nothing about the citation has changed.
              </p>
              <button className="btn btn-primary btn-sm" onClick={load}>
                Get a fresh link
              </button>
            </div>
          )}

          {link && !expired && !renderable && (
            <div className="doc-viewer-message">
              <strong>This is a {fileKind(name)} file, which the browser cannot show inline.</strong>
              <p className="card-note">
                Download it and go to page {page}. Page numbering follows the file as it was
                uploaded.
              </p>
              {link.download_url && (
                <a className="btn btn-primary btn-sm" href={link.download_url} download={name}>
                  Download the file
                </a>
              )}
            </div>
          )}

          {src && !expired && renderable && (
            <iframe className="doc-viewer-frame" src={src} title={`${name}, page ${page}`} />
          )}
        </div>
      </div>
    </div>
  )
}
