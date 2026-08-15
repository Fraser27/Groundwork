"""Load the shipped legal documents through the real ingest pipeline.

Deliberately not a graph seeder. Writing assertions straight into Neptune would be faster
and would produce a demo that looks identical — right up to the moment somebody clicks a
citation, at which point provenance points at a document that does not exist and the
central claim of the product ("every fact carries the reason it is believed") is visibly
false.

So this uploads real files to S3 and runs the same path an operator's upload takes. Every
resulting assertion cites a real document, page and character span, and the Provenance page
resolves it.

Idempotent by construction rather than by a guard: keys are content-addressed, so re-running
this produces the same `document_id`, and therefore the same assertion ids. Loading twice
converges instead of duplicating.

The documents describe one coherent scenario — an engagement, the advice given under it, and
the conflict check that preceded it — because a demo where the facts do not interlock cannot
show conflict detection or cross-document inference at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample" / "documents"

#: Five demo matters as PDFs, committed so a fresh clone needs no PDF toolchain. Regenerate
#: with `python sample/generate_demo_pdfs.py` when the content changes.
SAMPLE_ZIP = Path(__file__).resolve().parents[1] / "sample" / "legal-demo.zip"

#: The default matter, used for a document whose filename carries no matter reference.
SAMPLE_MATTER_ID = "NTL-2026-0114"

SAMPLE_MATTER_NAME = "Northwind Trading Ltd v Calder Shipping AG"

#: What each demo matter is, for the load report. The three NTL documents share a matter on
#: purpose: an engagement, the advice given under it, and the conflict check that preceded
#: it, so matter-scoped reads and the ethical wall are both demonstrable.
SAMPLE_MATTERS = {
    "NTL-2026-0114": "Northwind Trading Ltd v Calder Shipping AG",
    "MBC-2024-0431": "Meridian Bulk Carriers SA: secured facility",
    "HAL-2025-0092": "Halveston Chartering Ltd: know-how",
}


def _matter_of(filename: str) -> str:
    """The matter a demo document belongs to, read from its filename.

    Encoding it in the name keeps the mapping visible in the zip rather than in a table
    somebody has to find, and a document added to the zip needs no code change.
    """
    stem = filename.rsplit("/", 1)[-1]
    for matter_id in SAMPLE_MATTERS:
        if stem.startswith(matter_id):
            return matter_id
    return SAMPLE_MATTER_ID


@dataclass
class SampleLoadReport:
    documents_loaded: int = 0
    documents_skipped: int = 0
    assertions_live: int = 0
    assertions_pending: int = 0
    chunks: int = 0
    matter_id: str = SAMPLE_MATTER_ID
    errors: list[str] = field(default_factory=list)
    details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "documents_loaded": self.documents_loaded,
            "documents_skipped": self.documents_skipped,
            "assertions_live": self.assertions_live,
            "assertions_pending": self.assertions_pending,
            "chunks": self.chunks,
            "matter_id": self.matter_id,
            "matters": SAMPLE_MATTERS,
            "errors": self.errors,
            "documents": self.details,
            "note": (
                "Loaded through the real ingest pipeline, so every assertion cites a page "
                "and a verbatim span in a document you can open. Re-running is safe: keys "
                "are content-addressed, so the same bytes yield the same ids."
            ),
        }


def sample_documents() -> list[tuple[str, bytes]]:
    """The shipped demo documents, as (filename, bytes).

    Prefers the PDF zip, which is the realistic demo: PDFs go through page rasterisation and
    the vision model, which is the path a real upload takes. Falls back to the plain-text
    documents so the demo still loads on a deployment with no Bedrock access, where a PDF
    could be stored but never read.
    """
    import zipfile

    if SAMPLE_ZIP.is_file():
        try:
            with zipfile.ZipFile(SAMPLE_ZIP) as archive:
                return [
                    (name, archive.read(name))
                    for name in sorted(archive.namelist())
                    if name.lower().endswith(".pdf")
                ]
        except (zipfile.BadZipFile, OSError) as e:
            logger.warning("could not read %s: %s", SAMPLE_ZIP, e)

    if SAMPLE_DIR.is_dir():
        return [(path.name, path.read_bytes()) for path in sorted(SAMPLE_DIR.glob("*.txt"))]
    return []


def load_sample_data(
    services: Any, ctx: Any, *, run_model_extraction: bool = True
) -> SampleLoadReport:
    """Ingest every sample document for this tenant.

    `run_model_extraction` is honoured rather than forced: without Bedrock the documents
    still land, chunk and embed, so the demo degrades to searchable-but-not-extracted rather
    than failing outright. That is the same degradation the upload route already chooses.
    """
    report = SampleLoadReport()
    documents = sample_documents()
    if not documents:
        report.errors.append(f"no sample documents found in {SAMPLE_DIR}")
        return report

    from src.api.routes_documents import _require_runner

    runner = _require_runner(services)

    for filename, body in documents:
        matter_id = _matter_of(filename)
        is_pdf = filename.lower().endswith(".pdf")
        try:
            doc = runner.storage.put_document(
                ctx,
                filename=filename,
                body=body,
                matter_id=matter_id,
                media_type="application/pdf" if is_pdf else "text/plain",
            )
            parsed = services_parse(services, doc, body)
            if parsed is None:
                report.documents_skipped += 1
                report.errors.append(
                    f"{filename}: stored, but no vision model is available to read it"
                )
                continue
            result = runner.pipeline(
                ctx,
                parsed,
                matter_id=matter_id,
                run_model_extraction=run_model_extraction,
                job_id=doc.document_id,
            )
            report.documents_loaded += 1
            report.chunks += int(result.get("chunks") or 0)
            report.assertions_live += int(result.get("assertions_live") or 0)
            report.assertions_pending += int(result.get("pending_review") or 0)
            report.details.append(
                {
                    "filename": filename,
                    "document_id": doc.document_id,
                    "matter_id": matter_id,
                    "chunks": result.get("chunks"),
                    "extraction": result.get("extraction"),
                }
            )
        except Exception as e:
            logger.exception("sample document %s failed", filename)
            report.errors.append(f"{filename}: {e}")

    logger.info(
        "loaded %d sample documents for %s (%d chunks)",
        report.documents_loaded,
        ctx.tenant_id,
        report.chunks,
    )
    return report


def services_parse(services: Any, doc: Any, body: bytes) -> Any:
    """Parse a sample document.

    All samples are plain text, so this never needs a vision model — which is the reason the
    samples are `.txt` rather than PDF. A demo that cannot load without Bedrock credentials
    is a demo that fails on a fresh account.
    """
    from src.documents.parse import parse_plain_text

    text = body.decode("utf-8", errors="replace")
    return parse_plain_text(doc.document_id, text, filename=doc.filename)
