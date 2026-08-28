"""Render an answer as text, for an agent that has to show it to somebody.

**No model here.** This is string building over a dict that is already the answer. A renderer
that called a model would be a second ungoverned layer inside a tool whose whole purpose is
saying how much each part can be trusted, and nobody could attribute a sentence to either.

**Rendering is additive, never a substitution.** The caller gets the structured answer exactly
as before plus a `formatted` string. A tool that returned prose *instead of* the data would let
a client display a governance label with none of the parts behind it, which is the failure the
trace exists to prevent -- so `structuredContent` always holds the full shape and this is a
convenience on top.

Three things every format must carry, because each is a way the answer could be read as
stronger than it is:

- the **governance label**, which stops saying "governed" the moment a model contributed;
- the **blocks**, which are findings the graph made rather than omissions;
- the **skipped lanes**, because "we did not look there" and "we looked and found nothing" are
  different answers and only one of them is reassuring.
"""

from __future__ import annotations

from typing import Any

#: What a caller may ask for. `data` is the default and adds nothing, so an existing client sees
#: byte-identical responses.
FORMATS = ("data", "markdown", "table")

DEFAULT_FORMAT = "data"


class UnknownFormat(ValueError):
    """An unsupported `response_format`. Named so a tool can turn it into a readable refusal."""


def validate_format(value: str | None) -> str:
    fmt = (value or DEFAULT_FORMAT).strip().lower()
    if fmt not in FORMATS:
        raise UnknownFormat(f"response_format must be one of {', '.join(FORMATS)}, got {value!r}")
    return fmt


def render(answer: dict[str, Any], fmt: str) -> str | None:
    """The answer as text, or None when the caller wanted only the data.

    Handles both shapes a tool can return. `compose` produces `parts`, each with its own trust;
    `ask` produces one tier and one answer. Rendering the thin shape through the part-wise path
    printed "nothing was found" for an answer that existed, which is worse than not rendering it.
    """
    if fmt == "data":
        return None
    normalised = answer if "parts" in answer else _as_parts(answer)
    if fmt == "markdown":
        return _markdown(normalised)
    return _table(normalised)


#: Lists a multi-store answer can carry, and what to call each one to a reader. A traversal tier
#: returns one dict holding several of them -- graph-first returns four -- and a single part wrapping
#: that dict renders as `str(dict)`, so the reader is shown a Python repr instead of an answer.
_EVIDENCE_LANES = {
    "facts": "facts",
    "passages": "passages",
    "related": "related facts",
    "tables": "schema",
}


def _as_parts(resolution: dict[str, Any]) -> dict[str, Any]:
    """A single-tier `Resolution` in the part-wise shape, so one renderer serves both tools.

    Not a general adapter: it exists so `ask` can be rendered at all. The tier name becomes the
    part's lane because that is what a reader of a single-tier answer recognises, except where the
    tier searched several stores -- then each list becomes its own part, the way compose reports
    them, because "quoted from a document" and "a relationship in the graph" are not the same claim
    and a reader has to be able to tell which they are looking at.
    """
    answer = resolution.get("answer")
    common: dict[str, Any] = {
        "provenance": "deterministic" if resolution.get("governed") else "model_written",
        "tier": resolution.get("tier"),
        "confidence": None,
        "error": None,
        "sql": None,
        "assertion_ids": [],
    }

    parts: list[dict[str, Any]] = []
    if isinstance(answer, dict) and any(key in answer for key in _EVIDENCE_LANES):
        for key, lane in _EVIDENCE_LANES.items():
            rows = answer.get(key)
            if not rows:
                continue
            parts.append(
                {
                    **common,
                    "lane": lane,
                    "content": rows,
                    "assertion_ids": [
                        r["assertion_id"]
                        for r in rows
                        if isinstance(r, dict) and r.get("assertion_id")
                    ],
                }
            )
        generated = answer.get("generated")
        if isinstance(generated, dict) and generated.get("sql"):
            parts.append(
                {
                    **common,
                    # Never deterministic, whatever the tier says: nothing approved this query.
                    "provenance": "model_written",
                    "lane": "generated query",
                    "content": generated.get("rows"),
                    "sql": generated.get("sql"),
                    "error": generated.get("error"),
                }
            )
    elif bool(answer) or bool(resolution.get("sql")):
        parts.append(
            {
                **common,
                "lane": str(resolution.get("tier_name", "result")).lower(),
                "content": answer,
                "sql": resolution.get("sql"),
                "assertion_ids": resolution.get("assertions_used") or [],
            }
        )
    return {
        "parts": parts,
        "blocks": resolution.get("blocks") or [],
        "lanes_skipped": {},
        "governance": "governed" if resolution.get("governed") else "a model contributed",
        "fully_deterministic": bool(resolution.get("governed")),
        "explanation": resolution.get("explanation"),
    }


def _markdown(answer: dict[str, Any]) -> str:
    lines: list[str] = []
    governance = answer.get("governance") or "unknown"
    lines.append(f"**Trust:** {governance}")
    if answer.get("fully_deterministic"):
        lines.append("")
        lines.append("No model contributed to this answer.")

    for block in answer.get("blocks") or []:
        lines.append("")
        lines.append(f"> **{_block_headline(block)}** {_block_detail(block)}")

    for part in answer.get("parts") or []:
        lines.append("")
        lines.append(f"### {_part_title(part)}")
        lines.append(_part_trust(part))
        if part.get("error"):
            lines.append("")
            lines.append(f"This part failed: {part['error']}")
        body = _part_body(part)
        if body:
            lines.append("")
            lines.extend(body)

    skipped = answer.get("lanes_skipped") or {}
    if skipped:
        lines.append("")
        lines.append("### Not searched")
        for lane, reason in sorted(skipped.items()):
            lines.append(f"- **{lane}**: {reason}")

    if not answer.get("parts"):
        lines.append("")
        lines.append("Nothing was found. Read the skipped lanes above before reporting an absence.")

    return "\n".join(lines).strip()


def _table(answer: dict[str, Any]) -> str:
    """One row per part. For a client that wants to show where an answer came from at a glance."""
    rows = [
        ("Source", "Trust", "Tier", "Confidence", "Facts", "Detail"),
        ("---", "---", "---", "---", "---", "---"),
    ]
    for part in answer.get("parts") or []:
        confidence = part.get("confidence")
        rows.append(
            (
                str(part.get("lane", "")),
                str(part.get("provenance", "")),
                str(part.get("tier", "")),
                f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "exact",
                str(len(part.get("assertion_ids") or [])),
                _one_line(part),
            )
        )

    out = [f"**Trust:** {answer.get('governance') or 'unknown'}", ""]
    out.extend("| " + " | ".join(r) + " |" for r in rows)

    for block in answer.get("blocks") or []:
        out.append("")
        out.append(f"{_block_headline(block)} {_block_detail(block)}")

    skipped = answer.get("lanes_skipped") or {}
    if skipped:
        out.append("")
        out.append("Not searched: " + "; ".join(f"{k} ({v})" for k, v in sorted(skipped.items())))
    return "\n".join(out).strip()


def _block_headline(block: dict[str, Any]) -> str:
    """A withheld block and a reported one are different events and must not read alike."""
    if str(block.get("effect") or "withhold") == "withhold":
        return "Withheld:"
    return "Finding to consider:"


def _block_detail(block: dict[str, Any]) -> str:
    parts = [str(block.get("reason") or "no reason recorded")]
    if block.get("matter_id"):
        parts.append(f"Matter {block['matter_id']}.")
    if block.get("contact"):
        parts.append(f"Contact {block['contact']}.")
    return " ".join(parts)


def _part_title(part: dict[str, Any]) -> str:
    lane = str(part.get("lane", "result")).replace("_", " ")
    return lane[:1].upper() + lane[1:]


def _part_trust(part: dict[str, Any]) -> str:
    provenance = str(part.get("provenance", "unknown")).replace("_", " ")
    confidence = part.get("confidence")
    if isinstance(confidence, (int, float)):
        return f"{provenance}, confidence {confidence:.2f}"
    return provenance


def _part_body(part: dict[str, Any]) -> list[str]:
    """The part's own content, in whatever shape that lane produces."""
    if part.get("sql"):
        return ["```sql", str(part["sql"]).strip(), "```"]

    content = part.get("content")
    if isinstance(content, dict) and "columns" in content:
        return _rows_table(content)
    if isinstance(content, list):
        return [f"- {_summarise(item)}" for item in content[:20]]
    if content:
        return [str(content)]
    return []


def _rows_table(content: dict[str, Any]) -> list[str]:
    columns = [str(c) for c in content.get("columns") or []]
    rows = content.get("rows") or []
    if not columns:
        return []
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows[:20]:
        out.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return out


def _summarise(item: Any) -> str:
    """One line for a passage or a fact, whichever this is."""
    if not isinstance(item, dict):
        return str(item)
    if item.get("predicate"):
        return f"{item.get('subject_id', '?')} {item['predicate']} {item.get('object_id', '?')}"
    if item.get("text"):
        page = item.get("page")
        where = f" (p{page})" if page else ""
        return f"{item.get('filename') or item.get('document_id', 'document')}{where}: {item['text'][:200]}"
    return ", ".join(f"{k}={v}" for k, v in list(item.items())[:4])


def _one_line(part: dict[str, Any]) -> str:
    """A table cell, so no newlines and no pipes."""
    body = " ".join(_part_body(part))
    return body.replace("|", "/").replace("\n", " ")[:120] or "-"
