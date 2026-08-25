"""Rendering an answer as text without weakening what it says.

The risk this file guards is specific. A caller can ask for `markdown` because it wants
something to show a person, and a renderer that dropped the awkward parts would be a way to ask
the server for a cleaner-looking answer than the truth. So every format has to carry the
governance label, the blocks and the lanes that did not run, and the structured answer has to
survive alongside the prose rather than being replaced by it.
"""

from __future__ import annotations

import pytest

from src.query.rendering import DEFAULT_FORMAT, FORMATS, UnknownFormat, render, validate_format

COMPOSED = {
    "parts": [
        {
            "lane": "metric",
            "provenance": "deterministic",
            "tier": 1,
            "content": {"columns": ["month", "fees"], "rows": [["2026-01", 120]]},
            "sql": "SELECT 1",
            "assertion_ids": [],
            "confidence": None,
            "error": None,
        },
        {
            "lane": "passages",
            "provenance": "verbatim",
            "tier": 3,
            "content": [
                {
                    "document_id": "doc-1",
                    "filename": "letter.pdf",
                    "page": 2,
                    "text": "Calder is the adverse party",
                }
            ],
            "sql": None,
            "assertion_ids": ["a1", "a2"],
            "confidence": 0.91,
            "error": None,
        },
    ],
    "blocks": [
        {
            "reason": "Potential conflict involving matter:NTL",
            "effect": "notify",
            "matter_id": "NTL",
            "contact": "risk@firm.example",
        }
    ],
    "lanes_skipped": {"graph": "tier 2 is not permitted for this tenant"},
    "governance": "deterministic + verbatim",
    "fully_deterministic": False,
}

RESOLUTION = {
    "tier": 1,
    "tier_name": "GOVERNED_METRIC",
    "governed": True,
    "sql": "SELECT 1",
    "answer": {"columns": ["fees"], "rows": [[120]]},
    "assertions_used": [],
    "blocks": [],
}


class TestTheFormatArgument:
    def test_data_is_the_default_and_renders_nothing(self):
        """An existing client must see a byte-identical response, so the default adds no key."""
        assert validate_format(None) == DEFAULT_FORMAT
        assert render(COMPOSED, "data") is None

    def test_an_unknown_format_is_refused_by_name(self):
        with pytest.raises(UnknownFormat, match="markdown"):
            validate_format("yaml")

    def test_case_and_whitespace_are_tolerated(self):
        assert validate_format(" Markdown ") == "markdown"


class TestEveryFormatCarriesWhatWeakensTheAnswer:
    @pytest.mark.parametrize("fmt", [f for f in FORMATS if f != "data"])
    def test_the_governance_label_is_present(self, fmt):
        """It stops saying "governed" the moment a model contributed, so a rendering that lost
        it would let a caller show a stronger claim than the data supports."""
        assert "deterministic + verbatim" in render(COMPOSED, fmt)

    @pytest.mark.parametrize("fmt", [f for f in FORMATS if f != "data"])
    def test_a_block_is_named_with_its_matter_and_contact(self, fmt):
        """A lawyer who cannot see which client to ask about has been told nothing useful."""
        out = render(COMPOSED, fmt)
        assert "Potential conflict involving matter:NTL" in out
        assert "NTL" in out
        assert "risk@firm.example" in out

    @pytest.mark.parametrize("fmt", [f for f in FORMATS if f != "data"])
    def test_a_skipped_lane_says_why(self, fmt):
        """ "We did not look there" and "we looked and found nothing" are different answers."""
        out = render(COMPOSED, fmt)
        assert "graph" in out
        assert "not permitted" in out

    def test_a_notify_block_does_not_read_as_a_withholding(self):
        """Every legal rule finding is `notify`. Calling one "withheld" would tell a reader
        evidence was suppressed when it is all there."""
        out = render(COMPOSED, "markdown")
        assert "Finding to consider:" in out
        assert "Withheld:" not in out

    def test_a_withholding_block_says_so(self):
        withheld = {**COMPOSED, "blocks": [{"reason": "ethical screen", "effect": "withhold"}]}
        assert "Withheld:" in render(withheld, "markdown")


class TestBothToolShapesRender:
    def test_a_single_tier_answer_is_not_reported_as_empty(self):
        """`ask` returns a tier and an answer, with no `parts`. Rendering it through the
        part-wise path printed "nothing was found" for an answer that existed."""
        out = render(RESOLUTION, "markdown")
        assert "Nothing was found" not in out
        assert "Governed metric" in out
        assert "SELECT 1" in out

    def test_a_genuinely_empty_answer_still_says_so(self):
        """The opposite failure. An empty result has to read as empty, and has to point at the
        skipped lanes rather than at an absence."""
        empty = {"parts": [], "blocks": [], "lanes_skipped": {}, "governance": "no answer"}
        out = render(empty, "markdown")
        assert "Nothing was found" in out

    def test_a_table_cell_cannot_break_the_table(self):
        """Content is a document quote or SQL, either of which can hold a pipe or a newline."""
        nasty = {
            **COMPOSED,
            "parts": [
                {
                    "lane": "passages",
                    "provenance": "verbatim",
                    "tier": 3,
                    "content": [{"text": "a | b\nc"}],
                    "sql": None,
                    "assertion_ids": [],
                    "confidence": 0.5,
                    "error": None,
                }
            ],
        }
        rendered = render(nasty, "table").splitlines()
        row = next(ln for ln in rendered if ln.startswith("| passages"))
        # Six columns means seven pipes. A leaked pipe would silently add a column.
        assert row.count("|") == 7


class TestAFailedPartIsNotHidden:
    def test_an_error_on_a_part_is_stated(self):
        """A hallucinated column errors at Athena, and reporting that as no rows would read as
        "no data", which is the silent empty this codebase refuses."""
        failed = {
            **COMPOSED,
            "parts": [
                {
                    "lane": "sql",
                    "provenance": "model_written",
                    "tier": 3,
                    "content": None,
                    "sql": "SELECT bad_column FROM matters",
                    "assertion_ids": [],
                    "confidence": None,
                    "error": "COLUMN_NOT_FOUND: bad_column",
                }
            ],
        }
        out = render(failed, "markdown")
        assert "COLUMN_NOT_FOUND" in out
        assert "model written" in out
