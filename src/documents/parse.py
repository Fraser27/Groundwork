"""Read a document into text, one page at a time, with a vision model.

There is no Textract here, deliberately. Textract reads *text*, and a legal document
carries meaning in things that are not text: a cap table, an org chart, a signature
block, a handwritten marginal note, a redaction box. Textract returns nothing useful
for those, and "nothing" is indistinguishable downstream from "no such content".

A vision model reads all of it, and describes a chart in prose that can then be
quoted, embedded and cited exactly like any other passage.

The output shape is unchanged and is still the point. Everything downstream — chunking,
extraction, highlighting — needs one buffer plus a way to map any offset back to a
page:

    ParsedDocument.text     one buffer, pages joined by a known separator
    ParsedDocument.pages    [(page_no, char_start, char_end), ...]

**Page numbers are now exact by construction.** Each page is rendered and transcribed
separately, so a page number is the page we sent rather than something inferred from a
coordinate stream. Since provenance is file + page + quote, that is the property that
matters most.

The OCR model is configured *separately* from the extraction model, because they are
different jobs at different prices: transcription is mechanical and a cheap model does
it well; deciding whether a holding undercuts an argument is not. Both are editable in
Admin.
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol

from src.constants import MAX_PAGE_CONCURRENCY, PAGE_BATCH_SIZE

logger = logging.getLogger(__name__)

#: Pages are joined with a form feed so a page boundary is findable in the text and is
#: exactly one character wide. A newline would be ambiguous with a line break.
PAGE_SEPARATOR = "\f"

#: Rendering resolution. 150 DPI is legible to a vision model without inflating the
#: payload; below ~110 small print starts being transcribed *wrongly*, which is worse
#: than failing because it looks like text.
RENDER_DPI = 150

#: Ceiling on one document. One model call per page means a 400-page bundle is a real
#: cost, so the limit is explicit here rather than discovered on an invoice.
MAX_PAGES = 400


OCR_SYSTEM_PROMPT = """\
You transcribe pages of legal documents for a system where every claim must be \
traceable to the exact words on the page.

Transcribe the page as plain text, preserving reading order, paragraph breaks, and the \
text of headings, captions and signature blocks.

Rules:
- Transcribe verbatim. Do not summarise, correct spelling, expand abbreviations or \
modernise punctuation. Later steps quote this text and cite it back to this page, so an \
"improvement" becomes a citation to words that were never written.
- For a table, transcribe cells in reading order, one row per line, columns separated \
by " | ". Keep the header row.
- For a chart, diagram, org chart or image, write a description in square brackets \
stating what it shows and any labelled values, e.g. [Bar chart: fees by practice group. \
Litigation 1.2m, Corporate 0.8m, Tax 0.3m].
- For a stamp, seal or handwritten annotation, transcribe the words and mark it, e.g. \
[Handwritten: approved 3 May].
- If a region is redacted or illegible, say so: [Redacted] or [Illegible].
- Output only the transcription. No preamble, no commentary about the page.
"""


class PageRenderer(Protocol):
    """Turns document bytes into one PNG per page."""

    def render(self, data: bytes, *, dpi: int = RENDER_DPI) -> list[bytes]: ...


class VisionLike(Protocol):
    def invoke_model(self, **kwargs: Any) -> dict[str, Any]: ...


class ParseFailed(RuntimeError):
    """The document could not be rendered or transcribed."""


@dataclass(frozen=True)
class PageSpan:
    page: int
    char_start: int
    char_end: int

    def contains(self, offset: int) -> bool:
        return self.char_start <= offset < self.char_end


@dataclass
class ParsedDocument:
    document_id: str
    text: str
    pages: tuple[PageSpan, ...]
    method: str
    """Versioned and specific, e.g. `vision:claude-haiku-4-5@v1`. Recorded on every
    assertion derived from this text, so changing the transcription model supersedes
    old output rather than silently mixing generations."""

    filename: str | None = None
    page_texts: tuple[str, ...] = field(default=(), repr=False)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def page_at(self, offset: int) -> int:
        """Which page a global character offset falls on.

        An offset on a separator or past the end resolves to the nearest real page
        rather than raising: a citation landing one character beyond a page break
        should still name a page.
        """
        for span in self.pages:
            if span.contains(offset):
                return span.page
        return self.pages[-1].page if self.pages else 1

    def text_of(self, page: int) -> str:
        for span in self.pages:
            if span.page == page:
                return self.text[span.char_start : span.char_end]
        return ""


def assemble(
    document_id: str,
    page_texts: Sequence[str],
    *,
    method: str,
    filename: str | None = None,
) -> ParsedDocument:
    """Join per-page transcriptions into one buffer plus its page map.

    The map is built from the join rather than parsed back out of the finished text, so
    the two cannot disagree.
    """
    buffer: list[str] = []
    spans: list[PageSpan] = []
    cursor = 0

    for index, page_text in enumerate(page_texts, start=1):
        if index > 1:
            buffer.append(PAGE_SEPARATOR)
            cursor += len(PAGE_SEPARATOR)
        start = cursor
        buffer.append(page_text)
        cursor += len(page_text)
        spans.append(PageSpan(page=index, char_start=start, char_end=cursor))

    return ParsedDocument(
        document_id=document_id,
        text="".join(buffer),
        pages=tuple(spans),
        method=method,
        filename=filename,
        page_texts=tuple(page_texts),
    )


def parse_plain_text(document_id: str, text: str, *, filename: str | None = None) -> ParsedDocument:
    """Treat already-extracted text as a single page.

    Used for text uploads and tests. No model call, and the method records that nothing
    transcribed it.
    """
    return assemble(document_id, [text], method="text:inline@v1", filename=filename)


class PyMuPDFRenderer:
    """Rasterises PDF pages with PyMuPDF.

    PyMuPDF is AGPL-3.0 and is used here *only* to turn a page into a PNG — trivially
    replaceable by `pdftoppm`, Ghostscript, or a Lambda layer, and it must be replaced
    before this ships commercially. Isolating it behind `PageRenderer` keeps that a
    one-class change rather than a refactor.
    """

    def render(self, data: bytes, *, dpi: int = RENDER_DPI) -> list[bytes]:
        try:
            import pymupdf
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ParseFailed(
                "no PDF renderer available: install pymupdf, or pass a PageRenderer"
            ) from e

        images: list[bytes] = []
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            if doc.page_count > MAX_PAGES:
                raise ParseFailed(
                    f"{doc.page_count} pages exceeds the {MAX_PAGES}-page ceiling; "
                    "split the document or raise MAX_PAGES deliberately"
                )
            for page in doc:
                images.append(page.get_pixmap(dpi=dpi).tobytes("png"))
        return images


class VisionParser:
    """Transcribes a document page by page with a vision model."""

    def __init__(
        self,
        *,
        model_id: str,
        bedrock: VisionLike | None = None,
        bedrock_factory: Callable[[], VisionLike] | None = None,
        renderer: PageRenderer | None = None,
        dpi: int = RENDER_DPI,
        max_pages: int = MAX_PAGES,
        max_tokens: int = 8192,
        version: str = "v1",
        batch_size: int = PAGE_BATCH_SIZE,
        max_concurrency: int = MAX_PAGE_CONCURRENCY,
    ) -> None:
        self.model_id = model_id
        self._bedrock = bedrock
        self._bedrock_factory = bedrock_factory
        self._renderer = renderer
        self.dpi = dpi
        self.max_pages = max_pages
        self.max_tokens = max_tokens
        self.version = version
        self.batch_size = max(1, batch_size)
        self.max_concurrency = max(1, max_concurrency)

    @property
    def method(self) -> str:
        return f"vision:{self.model_id}@{self.version}"

    @property
    def renderer(self) -> PageRenderer:
        # Constructed lazily so importing this module does not require PyMuPDF.
        if self._renderer is None:
            self._renderer = PyMuPDFRenderer()
        return self._renderer

    @property
    def bedrock(self) -> VisionLike:
        if self._bedrock is None:
            if self._bedrock_factory is None:
                raise ParseFailed("no vision client configured")
            self._bedrock = self._bedrock_factory()
        return self._bedrock

    def _read_page(self, image: bytes, page: int) -> str:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "system": OCR_SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.b64encode(image).decode(),
                            },
                        },
                        {"type": "text", "text": f"Transcribe page {page}."},
                    ],
                }
            ],
        }
        # `temperature` is deliberately absent: newer Anthropic models on Bedrock reject
        # it outright, and a rejected call would fail the whole parse.
        try:
            response = self.bedrock.invoke_model(modelId=self.model_id, body=json.dumps(body))
            payload = json.loads(response["body"].read())
        except Exception as e:
            raise ParseFailed(f"transcription failed on page {page}: {e}") from e

        blocks = payload.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        if not text.strip():
            # A blank page is legitimate — a divider, an empty verso. Record it as empty
            # rather than failing the document.
            logger.info("page %d transcribed empty", page)
        return text

    def parse(
        self, document_id: str, data: bytes, *, filename: str | None = None
    ) -> ParsedDocument:
        images = self.renderer.render(data, dpi=self.dpi)
        if not images:
            raise ParseFailed("renderer produced no pages")
        logger.info("transcribing %d pages of %s with %s", len(images), document_id, self.model_id)
        return self.parse_pages(document_id, images, filename=filename)

    def parse_pages(
        self,
        document_id: str,
        images: Sequence[bytes],
        *,
        filename: str | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> ParsedDocument:
        """Transcribe pre-rendered pages in batches, several calls in flight at once.

        The ceiling is enforced here as well as in `PyMuPDFRenderer`, because the cost
        is one model call per page and a substituted renderer must not be able to
        bypass it.

        Pages are transcribed concurrently but **reassembled in page order**. That is
        not a nicety: a page number is provenance, so a page landing at the wrong offset
        would make every citation into it wrong.

        `on_progress(done, total)` is called after each batch. A 400-page bundle takes
        minutes, and a caller with no way to report progress can only show a spinner.
        """
        if len(images) > self.max_pages:
            raise ParseFailed(
                f"{len(images)} pages exceeds the {self.max_pages}-page ceiling; "
                "split the document or raise max_pages deliberately"
            )

        page_texts: list[str] = []
        total = len(images)
        for start in range(0, total, self.batch_size):
            batch = images[start : start + self.batch_size]
            page_texts.extend(self._read_batch(batch, first_page=start + 1))
            if on_progress is not None:
                on_progress(len(page_texts), total)
        return assemble(document_id, page_texts, method=self.method, filename=filename)

    def _read_batch(self, images: Sequence[bytes], *, first_page: int) -> list[str]:
        """Transcribe one batch, in parallel, returned in page order.

        Serial when concurrency is 1 so tests and single-page documents do not pay for a
        thread pool. `map` rather than `as_completed` because the result order *is* the
        page order, and the first exception propagates — a page that failed to transcribe
        must fail its document rather than silently leaving a gap where text should be.
        """
        if self.max_concurrency <= 1 or len(images) == 1:
            return [self._read_page(img, first_page + i) for i, img in enumerate(images)]

        with ThreadPoolExecutor(max_workers=min(self.max_concurrency, len(images))) as pool:
            return list(
                pool.map(
                    lambda pair: self._read_page(pair[1], first_page + pair[0]),
                    enumerate(images),
                )
            )


def iter_page_texts(parsed: ParsedDocument) -> Iterator[tuple[int, str]]:
    for span in parsed.pages:
        yield span.page, parsed.text[span.char_start : span.char_end]
