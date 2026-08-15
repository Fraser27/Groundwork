"""Split a parsed document into chunks without ever losing an offset.

The invariant this module exists to hold:

    parsed.text[chunk.char_start:chunk.char_end] == chunk.text

for every chunk, always. `Chunk` validates the length half of that; the tests assert
the full identity. It sounds trivial and it is the easiest thing in the pipeline to
break — one `.strip()` on a chunk body shifts every offset after it, and the failure
surfaces months later as a highlight landing on the wrong paragraph in a filing
someone is relying on.

Two rules follow from it:

- Whitespace is never trimmed off a chunk. If a boundary should skip leading
  whitespace, the *offset moves with it*.
- Chunks never span a page boundary, because a chunk carries a single page number and
  a citation that says "page 4 or maybe 5" is not a citation.

Boundaries are preferred at paragraph breaks, then sentence ends, then whitespace,
and only then mid-token — a citation split across two chunks is invisible to the
deterministic extractors, which is a correctness problem rather than a cosmetic one.
"""

from __future__ import annotations

import re

from src.documents.models import Chunk
from src.documents.parse import ParsedDocument

DEFAULT_TARGET_CHARS = 1200
DEFAULT_OVERLAP_CHARS = 150
#: Below this a boundary search is not worth it — better a slightly long chunk than
#: one cut 200 characters early at a marginally nicer break.
MIN_CHARS = 200
#: Overlap is capped at half the target because chunk count scales as
#: `length / (target - overlap)`. At 95% overlap a document produces twenty times the
#: vectors, which is a silent twenty-fold embedding bill from a one-line config typo.
MAX_OVERLAP_RATIO = 0.5

_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"(?<=[.!?;:])[\s]")
_WHITESPACE = re.compile(r"\s")


def _best_boundary(text: str, lo: int, hi: int) -> int:
    """Pick a cut point in `text[lo:hi]`, returning an index into `text`.

    Searches backward from `hi` so the chunk stays under target. Returns `hi` when
    nothing better exists, which is the mid-token case.
    """
    window = text[lo:hi]
    for pattern in (_PARAGRAPH, _SENTENCE, _WHITESPACE):
        matches = list(pattern.finditer(window))
        if matches:
            # Cut *after* the separator so it belongs to the earlier chunk and no
            # character is dropped between chunks.
            cut = lo + matches[-1].end()
            if cut - lo >= MIN_CHARS:
                return cut
    return hi


def chunk_page(
    parsed: ParsedDocument,
    page: int,
    *,
    target_chars: int,
    overlap_chars: int,
) -> list[tuple[int, int]]:
    """Global (char_start, char_end) pairs for one page."""
    span = next(s for s in parsed.pages if s.page == page)
    start, end = span.char_start, span.char_end
    if start >= end:
        return []

    spans: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        # Skip leading whitespace by moving the offset, never by trimming the text.
        while cursor < end and parsed.text[cursor].isspace():
            cursor += 1
        if cursor >= end:
            break

        hard_stop = min(cursor + target_chars, end)
        cut = hard_stop if hard_stop >= end else _best_boundary(parsed.text, cursor, hard_stop)
        spans.append((cursor, cut))
        if cut >= end:
            break

        step = cut - overlap_chars
        # Overlap must not stall: a boundary that lands inside the overlap window
        # would re-emit the same chunk forever.
        cursor = max(step, cursor + 1) if step > cursor else cut
    return spans


def chunk_document(
    parsed: ParsedDocument,
    *,
    tenant_id: str,
    matter_id: str | None = None,
    filename: str | None = None,
    target_chars: int = DEFAULT_TARGET_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[Chunk]:
    ceiling = int(target_chars * MAX_OVERLAP_RATIO)
    if overlap_chars > ceiling:
        raise ValueError(
            f"overlap_chars {overlap_chars} exceeds {MAX_OVERLAP_RATIO:.0%} of "
            f"target_chars ({ceiling}), see MAX_OVERLAP_RATIO"
        )

    chunks: list[Chunk] = []
    ordinal = 0
    for page_span in parsed.pages:
        for start, stop in chunk_page(
            parsed, page_span.page, target_chars=target_chars, overlap_chars=overlap_chars
        ):
            chunks.append(
                Chunk(
                    document_id=parsed.document_id,
                    tenant_id=tenant_id,
                    matter_id=matter_id,
                    filename=filename,
                    ordinal=ordinal,
                    page=page_span.page,
                    char_start=start,
                    char_end=stop,
                    text=parsed.text[start:stop],
                )
            )
            ordinal += 1
    return chunks


def verify_offsets(parsed: ParsedDocument, chunks: list[Chunk]) -> None:
    """Assert the module invariant against a real parse.

    Called at the end of the chunking phase rather than only in tests: this is the
    cheapest possible check, and a silent offset drift poisons every assertion made
    from the document.
    """
    for chunk in chunks:
        actual = parsed.text[chunk.char_start : chunk.char_end]
        if actual != chunk.text:
            raise ValueError(
                f"offset drift in {chunk.chunk_id}: "
                f"expected {chunk.text[:40]!r}, buffer holds {actual[:40]!r}"
            )
        if parsed.page_at(chunk.char_start) != chunk.page:
            raise ValueError(f"{chunk.chunk_id} claims page {chunk.page} but offset disagrees")
