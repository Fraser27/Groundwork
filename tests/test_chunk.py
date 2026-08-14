"""Tests that offsets survive chunking.

One property does almost all the work here:

    parsed.text[chunk.char_start:chunk.char_end] == chunk.text

Everything downstream depends on it — a `source_locator` is only meaningful if the
offsets index the text that was actually parsed. The failure mode is silent and
delayed: chunking trims a space, every offset after it shifts by one, and months later
a highlight lands on the wrong paragraph of a filing someone is relying on.

So the property is asserted on every chunk of every fixture, not sampled.
"""

from __future__ import annotations

import pytest

from src.documents.chunk import (
    DEFAULT_OVERLAP_CHARS,
    DEFAULT_TARGET_CHARS,
    chunk_document,
    verify_offsets,
)
from src.documents.models import Chunk
from src.documents.parse import PAGE_SEPARATOR, ParsedDocument, assemble, parse_plain_text

PARA = (
    "The plaintiff alleges that the defendant breached the agreement dated "
    "March 4, 2021, and that such breach was material under the governing law. "
)


def _doc(pages: list[str]) -> ParsedDocument:
    """A multi-page document built the way the parser builds one.

    Through `assemble` rather than by joining pages into a string and reparsing: page
    numbers now come from the pages that were transcribed, so a fixture that
    joins-then-splits would exercise a recovery step that no longer exists.
    """
    return assemble("doc-1", pages, method="vision:test@v1")


def _assert_offsets_exact(parsed, chunks) -> None:
    assert chunks, "fixture produced no chunks"
    for chunk in chunks:
        assert parsed.text[chunk.char_start : chunk.char_end] == chunk.text


class TestOffsetsSurvive:
    def test_single_page_round_trips(self):
        parsed = parse_plain_text("doc-1", PARA * 20)
        chunks = chunk_document(parsed, tenant_id="firm-acme")
        _assert_offsets_exact(parsed, chunks)
        verify_offsets(parsed, chunks)

    def test_multi_page_round_trips(self):
        parsed = _doc([PARA * 8, PARA * 12, PARA * 3])
        chunks = chunk_document(parsed, tenant_id="firm-acme")
        _assert_offsets_exact(parsed, chunks)
        verify_offsets(parsed, chunks)

    @pytest.mark.parametrize("target,overlap", [(200, 0), (300, 50), (1200, 150), (5000, 400)])
    def test_offsets_hold_at_every_chunk_size(self, target, overlap):
        parsed = _doc([PARA * 6, PARA * 9])
        chunks = chunk_document(
            parsed, tenant_id="firm-acme", target_chars=target, overlap_chars=overlap
        )
        _assert_offsets_exact(parsed, chunks)

    def test_short_document_is_one_chunk_covering_it(self):
        text = "A short memo about the matter."
        parsed = parse_plain_text("doc-1", text)
        chunks = chunk_document(parsed, tenant_id="firm-acme")
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert (chunks[0].char_start, chunks[0].char_end) == (0, len(text))

    def test_leading_whitespace_moves_the_offset_rather_than_being_trimmed(self):
        """A `.strip()` here shifts every downstream offset. The offset must move."""
        text = "\n\n\n   Held for the plaintiff."
        parsed = parse_plain_text("doc-1", text)
        chunk = chunk_document(parsed, tenant_id="firm-acme")[0]
        assert chunk.text == "Held for the plaintiff."
        assert chunk.char_start == text.index("Held")
        assert parsed.text[chunk.char_start : chunk.char_end] == chunk.text

    def test_unicode_offsets_are_character_not_byte(self):
        text = "Ratified — see § 1983 — by the Bundesgerichtshof. " * 40
        parsed = parse_plain_text("doc-1", text)
        chunks = chunk_document(parsed, tenant_id="firm-acme", target_chars=300, overlap_chars=40)
        _assert_offsets_exact(parsed, chunks)

    def test_verify_offsets_catches_drift(self):
        parsed = parse_plain_text("doc-1", PARA * 10)
        chunks = chunk_document(parsed, tenant_id="firm-acme")
        drifted = chunks[:-1] + [
            Chunk(
                document_id="doc-1",
                tenant_id="firm-acme",
                ordinal=chunks[-1].ordinal,
                page=1,
                char_start=chunks[-1].char_start + 3,
                char_end=chunks[-1].char_end + 3,
                text=chunks[-1].text,
            )
        ]
        with pytest.raises(ValueError, match="offset drift"):
            verify_offsets(parsed, drifted)


class TestPageBoundaries:
    def test_no_chunk_spans_two_pages(self):
        """A chunk carries one page number; "page 4 or maybe 5" is not a citation."""
        parsed = _doc([PARA * 4, PARA * 4, PARA * 4])
        chunks = chunk_document(parsed, tenant_id="firm-acme", target_chars=5000, overlap_chars=100)
        for chunk in chunks:
            span = next(s for s in parsed.pages if s.page == chunk.page)
            assert span.char_start <= chunk.char_start
            assert chunk.char_end <= span.char_end

    def test_page_numbers_are_correct_and_one_based(self):
        parsed = _doc([PARA, PARA, PARA])
        chunks = chunk_document(parsed, tenant_id="firm-acme")
        assert [c.page for c in chunks] == [1, 2, 3]

    def test_page_separator_is_never_inside_a_chunk(self):
        parsed = _doc([PARA * 3, PARA * 3])
        chunks = chunk_document(parsed, tenant_id="firm-acme", target_chars=10000, overlap_chars=0)
        assert all(PAGE_SEPARATOR not in c.text for c in chunks)

    def test_empty_page_produces_no_chunk(self):
        parsed = _doc([PARA, "", PARA])
        chunks = chunk_document(parsed, tenant_id="firm-acme")
        assert {c.page for c in chunks} == {1, 3}

    def test_page_lookup_agrees_with_chunk_page(self):
        parsed = _doc([PARA * 5, PARA * 5])
        for chunk in chunk_document(parsed, tenant_id="firm-acme"):
            assert parsed.page_at(chunk.char_start) == chunk.page


class TestCoverageAndProgress:
    def test_every_non_whitespace_character_appears_in_some_chunk(self):
        parsed = _doc([PARA * 7, PARA * 4])
        chunks = chunk_document(parsed, tenant_id="firm-acme", target_chars=400, overlap_chars=60)
        covered = set()
        for chunk in chunks:
            covered.update(range(chunk.char_start, chunk.char_end))
        missing = [
            i
            for i, ch in enumerate(parsed.text)
            if i not in covered and not ch.isspace() and ch != PAGE_SEPARATOR
        ]
        assert missing == []

    def test_chunks_are_ordered_and_ordinals_are_dense(self):
        parsed = _doc([PARA * 6, PARA * 6])
        chunks = chunk_document(parsed, tenant_id="firm-acme", target_chars=400, overlap_chars=50)
        assert [c.ordinal for c in chunks] == list(range(len(chunks)))
        assert [c.char_start for c in chunks] == sorted(c.char_start for c in chunks)

    def test_overlap_does_not_stall_or_duplicate(self):
        """A boundary landing inside the overlap window could re-emit forever."""
        parsed = parse_plain_text("doc-1", "word " * 2000)
        chunks = chunk_document(parsed, tenant_id="firm-acme", target_chars=250, overlap_chars=125)
        assert len({c.chunk_id for c in chunks}) == len(chunks)
        assert len(chunks) < 200

    def test_pathological_overlap_is_rejected_not_absorbed(self):
        """Chunk count scales as length/(target-overlap): 95% overlap is a 20x bill."""
        parsed = parse_plain_text("doc-1", PARA * 5)
        with pytest.raises(ValueError, match="MAX_OVERLAP_RATIO"):
            chunk_document(parsed, tenant_id="firm-acme", target_chars=250, overlap_chars=240)

    def test_chunks_respect_the_target_size(self):
        parsed = parse_plain_text("doc-1", PARA * 30)
        chunks = chunk_document(parsed, tenant_id="firm-acme", target_chars=600, overlap_chars=80)
        assert all(len(c.text) <= 600 for c in chunks)

    def test_text_without_whitespace_still_terminates(self):
        parsed = parse_plain_text("doc-1", "x" * 3000)
        chunks = chunk_document(parsed, tenant_id="firm-acme", target_chars=500, overlap_chars=50)
        _assert_offsets_exact(parsed, chunks)
        assert len(chunks) >= 6


class TestBoundaryQuality:
    def test_prefers_a_paragraph_break(self):
        first = "The agreement was executed by both parties in good faith. " * 5
        text = first.rstrip() + "\n\n" + "The second recital follows here. " * 20
        parsed = parse_plain_text("doc-1", text)
        chunks = chunk_document(parsed, tenant_id="firm-acme", target_chars=340, overlap_chars=0)
        assert chunks[0].text.rstrip().endswith("in good faith.")

    def test_falls_back_past_a_break_that_would_cut_too_early(self):
        """MIN_CHARS: better a slightly long chunk than one cut 200 chars early."""
        text = "Short intro.\n\n" + "The body of the clause continues at length. " * 20
        parsed = parse_plain_text("doc-1", text)
        chunks = chunk_document(parsed, tenant_id="firm-acme", target_chars=400, overlap_chars=0)
        assert len(chunks[0].text) > 300

    def test_prefers_a_sentence_end_over_mid_word(self):
        text = "The motion is denied. " * 40
        parsed = parse_plain_text("doc-1", text)
        chunks = chunk_document(parsed, tenant_id="firm-acme", target_chars=300, overlap_chars=0)
        assert chunks[0].text.rstrip().endswith("denied.")

    def test_citation_not_split_across_chunks(self):
        """A citation split across two chunks is invisible to extraction, and a quote
        spanning the cut can never be verified against either chunk."""
        text = PARA * 4 + "The Court in Roe v. Wade, 410 U.S. 113 (1973), so held. " + PARA * 4
        parsed = parse_plain_text("doc-1", text)
        chunks = chunk_document(parsed, tenant_id="firm-acme", target_chars=350, overlap_chars=120)
        assert any("Roe v. Wade, 410 U.S. 113 (1973)" in c.text for c in chunks)


class TestChunkModelInvariants:
    def test_offsets_must_match_text_length(self):
        with pytest.raises(ValueError, match="do not match text"):
            Chunk(
                document_id="doc-1",
                tenant_id="firm-acme",
                ordinal=0,
                page=1,
                char_start=0,
                char_end=5,
                text="much longer than five",
            )

    def test_page_is_one_based(self):
        with pytest.raises(ValueError, match="1-based"):
            Chunk(
                document_id="doc-1",
                tenant_id="firm-acme",
                ordinal=0,
                page=0,
                char_start=0,
                char_end=3,
                text="abc",
            )

    def test_chunk_id_encodes_the_span(self):
        chunk = Chunk(
            document_id="doc-1",
            tenant_id="firm-acme",
            ordinal=2,
            page=4,
            char_start=100,
            char_end=103,
            text="abc",
        )
        assert chunk.chunk_id == "doc-1:p4:100-103"

    def test_span_hash_detects_text_changing_underneath(self):
        base = dict(document_id="doc-1", tenant_id="firm-acme", ordinal=0, page=1, char_start=0)
        a = Chunk(**base, char_end=3, text="abc")
        b = Chunk(**base, char_end=3, text="abd")
        assert a.span_sha256 != b.span_sha256

    def test_locator_rejects_a_span_outside_the_chunk(self):
        chunk = Chunk(
            document_id="doc-1",
            tenant_id="firm-acme",
            ordinal=0,
            page=1,
            char_start=100,
            char_end=110,
            text="0123456789",
        )
        with pytest.raises(ValueError, match="outside chunk"):
            chunk.to_locator(90, 105)

    def test_locator_narrows_to_a_subspan(self):
        chunk = Chunk(
            document_id="doc-1",
            tenant_id="firm-acme",
            ordinal=0,
            page=7,
            char_start=100,
            char_end=110,
            text="0123456789",
        )
        loc = chunk.to_locator(102, 105)
        assert (loc.page, loc.quote) == (7, "234")
        assert loc.chunk_id == chunk.chunk_id


class TestReparseIsIdempotent:
    def test_same_input_same_chunk_ids(self):
        """Rebuilding a derived index must converge, not accumulate."""
        pages = [PARA * 5, PARA * 5]
        first = chunk_document(_doc(pages), tenant_id="firm-acme")
        second = chunk_document(_doc(pages), tenant_id="firm-acme")
        assert [c.chunk_id for c in first] == [c.chunk_id for c in second]

    def test_defaults_are_sane(self):
        assert DEFAULT_OVERLAP_CHARS < DEFAULT_TARGET_CHARS
