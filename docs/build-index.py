#!/usr/bin/env python3
"""Rebuild the documentation search index.

Reads every content page in this directory, extracts its title, headings and visible
body text, and writes two files:

    search-index.json   the readable, diffable copy
    search-index.js     the same data assigned to a global, which is what the browser
                        actually loads

Both, not one, and the duplication is deliberate. A page opened from `file://` is
treated as an opaque origin, so `fetch()` on a sibling JSON file fails the CORS check
in Chrome and Safari with no way to allow it. A `<script>` tag has none of that
restriction. The `.json` exists because it is what a human or another tool should read;
the `.js` exists because it is what works offline.

Standard library only, on purpose — a documentation build that needs `pip install` is a
documentation build that stops working.

    python3 docs/build-index.py
"""

from __future__ import annotations

import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

DOCS = Path(__file__).resolve().parent

#: Page order matches the sidebar in assets/nav.js. Kept explicit rather than globbed
#: so a stray HTML file in this directory cannot silently enter the index, and so the
#: order of results for an equal-scoring match is stable.
PAGES = [
    "index.html",
    "assertion-contract.html",
    "extraction.html",
    "provenance.html",
    "tenancy.html",
    "predicates.html",
    "asking-questions.html",
    "review.html",
    "governance.html",
    "demo-data.html",
    "glossary.html",
    "architecture.html",
]

#: Content of these elements is never body text. Script and style would otherwise dump
#: code into the index; `head` covers everything in it, including the void `meta` and
#: `link` tags, which is why those are NOT listed here — a void tag never produces an
#: end tag, so counting it would leave the skip depth permanently raised and index
#: nothing at all.
SKIP = {"script", "style", "head", "title"}

#: Elements whose boundaries are a word boundary. Without this, "…the page</p><p>A
#: citation…" concatenates into "pageA citation" and a search for "a citation" misses.
BLOCK = {
    "p", "div", "li", "td", "th", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
    "dt", "dd", "pre", "blockquote", "section", "article", "aside", "main",
    "table", "thead", "tbody", "ul", "ol", "dl", "br", "hr", "span",
}


class PageParser(HTMLParser):
    """Pulls the title, the h2 headings with their ids, and the body text.

    Only `main` contributes body text. The sidebar is rendered by JavaScript so it is
    not in the source anyway, but scoping to `main` also keeps the eyebrow label and
    the page-nav links out of the index, neither of which is content.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.summary = ""
        self.headings: list[dict[str, str]] = []
        self._chunks: list[str] = []

        self._skip_depth = 0
        self._in_main = False
        self._in_title = False
        self._heading: dict[str, str] | None = None
        self._capture_lede = False
        self._lede: list[str] = []

    # ── tags ──────────────────────────────────────────────────────────────────

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {k: (v or "") for k, v in attrs}

        if tag in SKIP:
            self._skip_depth += 1
            self._in_title = tag == "title"
            return

        if tag == "main":
            self._in_main = True
        if tag in BLOCK:
            self._chunks.append(" ")

        if not self._in_main:
            return

        # h2 ids are the anchors search results link to, so a body hit can point at a
        # section rather than the top of the page.
        #
        # The heading's offset into the body is recorded here rather than being found
        # by searching for its text later. Searching is wrong: a heading's own words
        # usually also appear earlier on the page as a cross-reference link, so the
        # search finds the link and the section boundaries come out in the wrong order.
        if tag == "h2" and attr.get("id"):
            self._heading = {
                "id": attr["id"],
                "text": "",
                "at": len(collapse("".join(self._chunks))),
            }

        # The lede paragraph doubles as the page summary shown when a match has no
        # useful surrounding text.
        if tag == "p" and "lede" in attr.get("class", ""):
            self._capture_lede = True

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
            self._in_title = False
            return

        if tag == "main":
            self._in_main = False
        if tag in BLOCK:
            self._chunks.append(" ")

        if tag == "h2" and self._heading is not None:
            text = collapse(self._heading["text"])
            if text:
                self.headings.append(
                    {
                        "id": self._heading["id"],
                        "text": text,
                        "at": self._heading["at"],
                    }
                )
            self._heading = None

        if tag == "p" and self._capture_lede:
            self._capture_lede = False
            self.summary = collapse("".join(self._lede))

    # ── text ──────────────────────────────────────────────────────────────────

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
            return
        if self._skip_depth or not self._in_main:
            return
        self._chunks.append(data)
        if self._heading is not None:
            self._heading["text"] += data
        if self._capture_lede:
            self._lede.append(data)

    @property
    def body(self) -> str:
        return collapse("".join(self._chunks))


def collapse(text: str) -> str:
    """One space between words, no leading or trailing space."""
    return re.sub(r"\s+", " ", text).strip()


def clean_title(raw: str) -> str:
    """`Provenance — LexGraph documentation` -> `Provenance`.

    The suffix is useful in a browser tab and noise in a result list, where every hit
    would otherwise end with the same six words.
    """
    return collapse(raw.split("—")[0])


def build() -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    missing: list[str] = []

    for name in PAGES:
        path = DOCS / name
        if not path.exists():
            missing.append(name)
            continue

        parser = PageParser()
        parser.feed(path.read_text(encoding="utf-8"))
        parser.close()

        entries.append(
            {
                "path": name,
                "title": clean_title(parser.title),
                "summary": parser.summary,
                "headings": parser.headings,
                "body": parser.body,
            }
        )

    if missing:
        # A missing page means the index silently stops covering it, so this is loud.
        print(f"warning: not found, skipped: {', '.join(missing)}", file=sys.stderr)

    return entries


def main() -> int:
    entries = build()
    if not entries:
        print("error: no pages indexed", file=sys.stderr)
        return 1

    payload = json.dumps(entries, indent=2, ensure_ascii=False)
    (DOCS / "search-index.json").write_text(payload + "\n", encoding="utf-8")

    (DOCS / "search-index.js").write_text(
        "/* Generated by build-index.py — do not edit.\n"
        " *\n"
        " * A script rather than a fetch of search-index.json, because a page opened\n"
        " * from file:// cannot fetch a sibling file. Regenerate with:\n"
        " *     python3 docs/build-index.py\n"
        " */\n"
        "window.LEXGRAPH_SEARCH_INDEX = " + payload + ";\n",
        encoding="utf-8",
    )

    words = sum(len(e["body"].split()) for e in entries)  # type: ignore[union-attr]
    heads = sum(len(e["headings"]) for e in entries)  # type: ignore[arg-type]
    print(f"indexed {len(entries)} pages, {heads} sections, {words:,} words")
    print("wrote search-index.json and search-index.js")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
