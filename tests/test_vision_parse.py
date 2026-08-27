"""Tests for the vision page reader.

The property that matters most is that **a page number is the page we sent**. Provenance
is file + page + quote, so a page number that is inferred rather than known is a
citation that can be wrong without anything looking wrong. Because each page is rendered
and transcribed separately, that is now testable directly: give the fake renderer three
pages and the parse must say three, in order.

Both AWS-facing collaborators are faked. No Bedrock call, and no PyMuPDF import — the
renderer is injected, which is the whole reason `PageRenderer` is a protocol.
"""

from __future__ import annotations

import re
import threading
import time

import pytest

from src.documents.parse import (
    MAX_PAGES,
    OCR_SYSTEM_PROMPT,
    PAGE_SEPARATOR,
    ParseFailed,
    VisionParser,
    assemble,
    parse_plain_text,
)

CHART = (
    "[Bar chart: fees by practice group. Litigation 1.2m, Corporate 0.8m, Tax 0.3m]"
)


class FakeRenderer:
    """Returns one opaque blob per page. Content is irrelevant; the count is not."""

    def __init__(self, pages: int) -> None:
        self.pages = pages
        self.dpi: int | None = None

    def render(self, data: bytes, *, dpi: int = 150) -> list[bytes]:
        self.dpi = dpi
        return [f"page-{i}".encode() for i in range(1, self.pages + 1)]


class FakeVision:
    """Answers with the transcription for the page it was actually asked about.

    Keyed off the page number in the prompt rather than off call order, because pages
    are transcribed concurrently and arrival order is not page order. A fake that
    counted calls would attribute page 3's text to page 1 and make a correct parse look
    broken — or worse, hide a real mix-up.

    `fail_on` raises for a given 1-based page so the error path can be tested without
    making every other test go through a failure branch.
    """

    def __init__(self, texts: list[str], *, fail_on: int | None = None) -> None:
        self.texts = texts
        self.fail_on = fail_on
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def converse(self, **kw):
        prompt = kw["messages"][0]["content"][-1]["text"]
        page = int(re.search(r"page (\d+)", prompt).group(1))
        with self._lock:
            self.calls.append({"modelId": kw["modelId"], "request": kw, "page": page})
        if page == self.fail_on:
            raise RuntimeError("ThrottlingException: rate exceeded")
        text = self.texts[page - 1] if page <= len(self.texts) else ""
        return {"output": {"message": {"content": [{"text": text}]}}}


def _parser(texts: list[str], *, pages: int | None = None, **kw) -> VisionParser:
    renderer = FakeRenderer(pages if pages is not None else len(texts))
    return VisionParser(
        model_id="test-ocr-model",
        bedrock=FakeVision(texts, fail_on=kw.pop("fail_on", None)),
        renderer=renderer,
        **kw,
    )


class TestPageNumbersAreExact:
    def test_page_count_matches_the_rendered_pages(self):
        parsed = _parser(["one", "two", "three"]).parse("doc-1", b"%PDF")
        assert parsed.page_count == 3

    def test_page_numbers_are_one_based_and_in_order(self):
        """The page number is the page that was sent, not one recovered from a buffer."""
        parsed = _parser(["one", "two", "three"]).parse("doc-1", b"%PDF")
        assert [span.page for span in parsed.pages] == [1, 2, 3]

    def test_each_page_asks_for_the_page_it_is(self):
        parser = _parser(["one", "two"])
        parser.parse("doc-1", b"%PDF")
        prompts = [c["request"]["messages"][0]["content"][1]["text"] for c in parser.bedrock.calls]
        assert prompts == ["Transcribe page 1.", "Transcribe page 2."]

    def test_one_model_call_per_page(self):
        parser = _parser(["a", "b", "c", "d"])
        parser.parse("doc-1", b"%PDF")
        assert len(parser.bedrock.calls) == 4

    def test_text_of_returns_that_page_only(self):
        parsed = _parser(["alpha", "beta", "gamma"]).parse("doc-1", b"%PDF")
        assert parsed.text_of(2) == "beta"


class TestOffsetMapping:
    def test_page_at_maps_an_offset_inside_each_page(self):
        parsed = _parser(["alpha", "beta", "gamma"]).parse("doc-1", b"%PDF")
        assert parsed.page_at(0) == 1
        assert parsed.page_at(parsed.pages[1].char_start) == 2
        assert parsed.page_at(parsed.pages[2].char_end - 1) == 3

    def test_page_at_on_a_separator_resolves_to_a_real_page(self):
        """A citation landing one character past a page break must still name a page."""
        parsed = _parser(["alpha", "beta"]).parse("doc-1", b"%PDF")
        separator = parsed.pages[0].char_end
        assert parsed.text[separator] == PAGE_SEPARATOR
        assert parsed.page_at(separator) in (1, 2)

    def test_page_at_past_the_end_does_not_raise(self):
        parsed = _parser(["alpha"]).parse("doc-1", b"%PDF")
        assert parsed.page_at(10_000) == 1

    def test_spans_are_disjoint_and_ascending(self):
        parsed = _parser(["one", "two", "three"]).parse("doc-1", b"%PDF")
        for earlier, later in zip(parsed.pages, parsed.pages[1:]):
            assert earlier.char_end < later.char_start

    def test_contains_excludes_the_end_offset(self):
        parsed = _parser(["alpha"]).parse("doc-1", b"%PDF")
        span = parsed.pages[0]
        assert span.contains(span.char_end - 1)
        assert not span.contains(span.char_end)


class TestTheMapNeverDisagreesWithTheText:
    """`assemble` builds the map from the join, so the two cannot drift apart."""

    @pytest.mark.parametrize(
        "pages",
        [
            ["one"],
            ["one", "two"],
            ["", "body", ""],
            ["a" * 3000, "b"],
            ["Ratified — see § 1983 —", "Bundesgerichtshof"],
            [f"table\nrow 1 | 2\n{CHART}", "signed /s/ A. Partner"],
        ],
    )
    def test_every_span_slices_back_to_its_page(self, pages):
        parsed = assemble("doc-1", pages, method="vision:test@v1")
        assert [parsed.text[s.char_start : s.char_end] for s in parsed.pages] == pages

    def test_separator_is_exactly_one_char_between_pages(self):
        parsed = assemble("doc-1", ["a", "b", "c"], method="vision:test@v1")
        assert parsed.text == f"a{PAGE_SEPARATOR}b{PAGE_SEPARATOR}c"

    def test_a_single_page_gets_no_separator(self):
        parsed = assemble("doc-1", ["only"], method="vision:test@v1")
        assert PAGE_SEPARATOR not in parsed.text

    def test_the_map_survives_a_transcription_that_contains_a_form_feed(self):
        """A model echoing a form feed must not be able to invent a page boundary."""
        parsed = assemble("doc-1", [f"before{PAGE_SEPARATOR}after", "two"], method="m")
        assert parsed.page_count == 2
        assert parsed.text_of(1) == f"before{PAGE_SEPARATOR}after"


class TestNonTextContentSurvives:
    def test_a_chart_description_reaches_the_text(self):
        """The point of the vision model: a chart becomes prose that can be quoted."""
        parsed = _parser([f"Fee analysis\n\n{CHART}"]).parse("doc-1", b"%PDF")
        assert CHART in parsed.text

    def test_a_chart_description_is_citable_to_its_page(self):
        parsed = _parser(["cover page", f"exhibit B\n{CHART}"]).parse("doc-1", b"%PDF")
        assert parsed.page_at(parsed.text.index(CHART)) == 2


class TestDegradedPages:
    def test_an_empty_page_does_not_fail_the_document(self):
        """A divider or an empty verso is legitimate. Record it, do not raise."""
        parsed = _parser(["one", "", "three"]).parse("doc-1", b"%PDF")
        assert parsed.page_count == 3
        assert parsed.text_of(2) == ""

    def test_an_empty_page_keeps_later_page_numbers_right(self):
        parsed = _parser(["one", "", "three"]).parse("doc-1", b"%PDF")
        assert parsed.page_at(parsed.text.index("three")) == 3

    def test_every_page_empty_still_parses(self):
        parsed = _parser(["", "", ""]).parse("doc-1", b"%PDF")
        assert (parsed.page_count, parsed.text.strip()) == (3, "")


class TestFailureModes:
    def test_a_bedrock_failure_names_the_page(self):
        """Which page failed is the whole diagnostic — the retry is per document."""
        with pytest.raises(ParseFailed, match="page 2"):
            _parser(["one", "two", "three"], fail_on=2).parse("doc-1", b"%PDF")

    def test_the_underlying_error_is_not_swallowed(self):
        with pytest.raises(ParseFailed, match="Throttling"):
            _parser(["one"], fail_on=1).parse("doc-1", b"%PDF")

    def test_a_renderer_producing_no_pages_raises(self):
        with pytest.raises(ParseFailed, match="no pages"):
            _parser([], pages=0).parse("doc-1", b"not-a-pdf")

    def test_no_vision_client_raises_rather_than_returning_empty_text(self):
        """Silently empty text is indistinguishable from a blank document."""
        parser = VisionParser(model_id="m", renderer=FakeRenderer(1))
        with pytest.raises(ParseFailed, match="no vision client"):
            parser.parse("doc-1", b"%PDF")


class TestPageCeiling:
    def test_over_the_ceiling_raises(self):
        """One model call per page: a 400-page bundle is a cost, not an accident."""
        parser = _parser([], pages=MAX_PAGES + 1)
        with pytest.raises(ParseFailed, match="ceiling"):
            parser.parse("doc-1", b"%PDF")

    def test_the_ceiling_is_checked_before_any_model_call(self):
        parser = _parser([], pages=MAX_PAGES + 1)
        with pytest.raises(ParseFailed):
            parser.parse("doc-1", b"%PDF")
        assert parser.bedrock.calls == []

    def test_at_the_ceiling_is_allowed(self):
        parser = _parser([], pages=MAX_PAGES)
        assert parser.parse("doc-1", b"%PDF").page_count == MAX_PAGES

    def test_a_lowered_ceiling_is_honoured(self):
        """A substituted renderer must not be able to bypass the limit."""
        parser = _parser(["a", "b", "c"], max_pages=2)
        with pytest.raises(ParseFailed, match="2-page ceiling"):
            parser.parse("doc-1", b"%PDF")


class TestMethodAndMetadata:
    def test_method_records_model_and_version(self):
        """Stored on every assertion, so changing the model supersedes rather than mixes."""
        parser = _parser(["one"])
        assert parser.method == "vision:test-ocr-model@v1"
        assert parser.parse("doc-1", b"%PDF").method == "vision:test-ocr-model@v1"

    def test_filename_is_carried_onto_the_parse(self):
        """`Chunk.filename` comes from here, and `SourceLocator.filename` from that."""
        parsed = _parser(["one"]).parse("doc-1", b"%PDF", filename="motion.pdf")
        assert parsed.filename == "motion.pdf"

    def test_dpi_is_passed_to_the_renderer(self):
        parser = _parser(["one"], dpi=200)
        parser.parse("doc-1", b"%PDF")
        assert parser.renderer.dpi == 200

    def test_the_request_carries_the_page_image_as_raw_png_bytes(self):
        """Raw bytes, not base64: botocore encodes blob members, so pre-encoding would have
        the model read a base64 string as if it were the page."""
        parser = _parser(["one"])
        parser.parse("doc-1", b"%PDF")
        image = parser.bedrock.calls[0]["request"]["messages"][0]["content"][0]["image"]
        assert image["format"] == "png"
        assert isinstance(image["source"]["bytes"], bytes)

    def test_the_request_uses_converse_not_a_vendor_native_body(self):
        """The OCR model is admin-selectable across Nova and Anthropic, whose native bodies
        are mutually invalid. Converse is the only shape both accept."""
        parser = _parser(["one"])
        parser.parse("doc-1", b"%PDF")
        request = parser.bedrock.calls[0]["request"]
        assert "body" not in request
        assert request["system"] == [{"text": OCR_SYSTEM_PROMPT}]

    def test_temperature_is_never_sent(self):
        """Newer Anthropic models on Bedrock reject it, failing the whole parse."""
        parser = _parser(["one"])
        parser.parse("doc-1", b"%PDF")
        assert "temperature" not in parser.bedrock.calls[0]["request"]["inferenceConfig"]


class TestBatchedTranscription:
    """Pages are transcribed concurrently in batches. Reassembly must stay in page
    order, because a page number is provenance: a page landing at the wrong offset makes
    every citation into it wrong while looking perfectly healthy."""

    def test_pages_are_reassembled_in_order_despite_concurrency(self):
        texts = [f"page {i} text" for i in range(1, 21)]
        parsed = _parser(texts, batch_size=5, max_concurrency=8).parse("doc-1", b"%PDF")
        assert parsed.page_count == 20
        for i, expected in enumerate(texts, start=1):
            assert parsed.text_of(i) == expected

    def test_a_slow_early_page_does_not_shift_later_pages(self):
        """Finishing out of order is the whole risk concurrency introduces, so the fake
        deliberately makes page 1 the last to return."""

        class SlowFirstPage(FakeVision):
            def converse(self, **kw):
                prompt = kw["messages"][0]["content"][-1]["text"]
                if int(re.search(r"page (\d+)", prompt).group(1)) == 1:
                    time.sleep(0.05)
                return super().converse(**kw)

        texts = ["first", "second", "third", "fourth"]
        parser = VisionParser(
            model_id="test-ocr-model",
            bedrock=SlowFirstPage(texts),
            renderer=FakeRenderer(4),
            batch_size=4,
            max_concurrency=4,
        )
        parsed = parser.parse("doc-1", b"%PDF")
        assert [parsed.text_of(i) for i in range(1, 5)] == texts

    def test_every_page_is_transcribed_exactly_once(self):
        """A retried or duplicated page is a double Bedrock charge, and a dropped one is
        a silent gap where text should be."""
        parser = _parser([f"p{i}" for i in range(1, 13)], batch_size=5)
        parser.parse("doc-1", b"%PDF")
        assert sorted(c["page"] for c in parser.bedrock.calls) == list(range(1, 13))

    def test_concurrency_is_bounded(self):
        """Unbounded fan-out earns throttling and a burst of retries, so the cap has to
        be real rather than nominal."""
        peak = 0
        current = 0
        lock = threading.Lock()

        class CountingVision(FakeVision):
            def converse(self, **kw):
                nonlocal peak, current
                with lock:
                    current += 1
                    peak = max(peak, current)
                time.sleep(0.01)
                try:
                    return super().converse(**kw)
                finally:
                    with lock:
                        current -= 1

        parser = VisionParser(
            model_id="test-ocr-model",
            bedrock=CountingVision([f"p{i}" for i in range(1, 21)]),
            renderer=FakeRenderer(20),
            batch_size=20,
            max_concurrency=3,
        )
        parser.parse("doc-1", b"%PDF")
        assert peak <= 3

    def test_progress_is_reported_per_batch(self):
        """A 400-page bundle takes minutes; without this the UI can only spin."""
        seen: list[tuple[int, int]] = []
        parser = _parser([f"p{i}" for i in range(1, 13)], batch_size=5)
        parser.parse_pages(
            "doc-1",
            [f"page-{i}".encode() for i in range(1, 13)],
            on_progress=lambda done, total: seen.append((done, total)),
        )
        assert seen == [(5, 12), (10, 12), (12, 12)]

    def test_a_failed_page_fails_the_document(self):
        """Confining a failure to its batch would leave a gap that reads as a blank page,
        which is indistinguishable downstream from a page that really is blank."""
        parser = _parser([f"p{i}" for i in range(1, 11)], batch_size=5, fail_on=7)
        with pytest.raises(ParseFailed, match="page 7"):
            parser.parse("doc-1", b"%PDF")

    def test_a_single_page_document_needs_no_pool(self):
        parsed = _parser(["only"], batch_size=5, max_concurrency=8).parse("doc-1", b"%PDF")
        assert parsed.page_count == 1
        assert parsed.text_of(1) == "only"

    def test_the_page_ceiling_still_applies(self):
        """Batching must not become a way around the per-document cost ceiling."""
        parser = _parser([], pages=5, max_pages=3, batch_size=2)
        with pytest.raises(ParseFailed, match="exceeds"):
            parser.parse("doc-1", b"%PDF")


class TestPlainText:
    def test_inline_text_is_one_page_and_names_no_model(self):
        parsed = parse_plain_text("doc-1", "already extracted")
        assert (parsed.page_count, parsed.method) == (1, "text:inline@v1")

    def test_inline_text_round_trips_exactly(self):
        text = "Held for the plaintiff.\n\n  Costs reserved. "
        parsed = parse_plain_text("doc-1", text)
        assert parsed.text == text
        assert parsed.text_of(1) == text

    def test_a_form_feed_in_inline_text_does_not_split_pages(self):
        """Page numbers come from pages transcribed, never from separators in a buffer."""
        parsed = parse_plain_text("doc-1", f"one{PAGE_SEPARATOR}two")
        assert parsed.page_count == 1

    def test_filename_is_carried(self):
        parsed = parse_plain_text("doc-1", "text", filename="memo.txt")
        assert parsed.filename == "memo.txt"
