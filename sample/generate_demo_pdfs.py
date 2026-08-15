"""Build the demo matter PDFs that ship as `sample/legal-demo.zip`.

Run this only when the demo content changes. The output is committed, so a fresh clone can
load the demo without generating anything and without a PDF toolchain.

The five matters interlock on purpose. A demo whose facts do not connect cannot show the
things this product exists to show: the conflict check only fires because Meridian appears as
a lender in one matter and a shareholder of the opposing party in another, and the stale
authority rule only fires because one advice note cites a case another document overrules.
Five unrelated documents would render the same and demonstrate nothing.

    .venv/bin/python sample/generate_demo_pdfs.py
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pymupdf

OUT_ZIP = Path(__file__).resolve().parent / "legal-demo.zip"

#: Body text is 10.5pt on 15pt leading, which is close enough to a real letter that page
#: numbers and character offsets in a citation land where a reader expects them.
_BODY = 10.5
_LEAD = 15
_MARGIN = 56


def _page(doc: pymupdf.Document, lines: list[tuple[str, str]]) -> None:
    """One page. `lines` is (style, text) where style is head, sub, body, or space."""
    page = doc.new_page(width=595, height=842)  # A4
    y = _MARGIN
    for style, text in lines:
        if style == "space":
            y += _LEAD * 0.6
            continue
        size, font = {
            "head": (15, "hebo"),
            "sub": (11.5, "hebo"),
            "body": (_BODY, "helv"),
        }[style]
        # Wrap by hand: insert_textbox reflows but returns a negative height when the text
        # does not fit, which is silent data loss in a document meant to be cited.
        width = 595 - 2 * _MARGIN
        for chunk in _wrap(text, width, size):
            page.insert_text((_MARGIN, y), chunk, fontsize=size, fontname=font)
            y += _LEAD if style == "body" else _LEAD * 1.3
    return None


def _wrap(text: str, width: float, size: float) -> list[str]:
    approx = int(width / (size * 0.5))
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > approx:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    out.append(line)
    return out or [""]


DOCUMENTS: dict[str, list[tuple[str, str]]] = {
    "NTL-2026-0114-engagement-letter.pdf": [
        ("head", "THORNE VAUX LLP"),
        ("body", "14 Bedford Row, London WC1R 4EH"),
        ("space", ""),
        ("body", "12 March 2026"),
        ("space", ""),
        ("body", "Northwind Trading Limited, Attn: Ms Ada Okafor, General Counsel"),
        ("space", ""),
        ("sub", "ENGAGEMENT: NORTHWIND TRADING LIMITED v CALDER SHIPPING AG"),
        ("body", "Our reference: NTL-2026-0114"),
        ("space", ""),
        (
            "body",
            "Thorne Vaux LLP represents Northwind Trading Limited in the charterparty "
            "dispute with Calder Shipping AG. Northwind Trading Limited is our client. "
            "Calder Shipping AG is the adverse party.",
        ),
        ("space", ""),
        ("sub", "1. SCOPE"),
        (
            "body",
            "We will advise on and conduct proceedings arising from the alleged repudiatory "
            "breach of the time charterparty dated 4 August 2025 for the vessel MV Aurelia, "
            "including the LMAA arbitration reference and any enforcement.",
        ),
        (
            "body",
            "We do not advise on the tax treatment of any settlement sum. That exclusion is "
            "deliberate and survives any variation of this engagement.",
        ),
        ("space", ""),
        ("sub", "2. THE TEAM"),
        (
            "body",
            "Ms Perrine Duval, partner, supervises. Mr Kavi Iyer, senior associate, has "
            "day-to-day conduct. Ms Adaeze Mensah assists on disclosure.",
        ),
        ("space", ""),
        ("sub", "3. FEES"),
        (
            "body",
            "Partner GBP 650, senior associate GBP 420, associate GBP 310, paralegal GBP 165. "
            "We estimate GBP 180,000 to the conclusion of the arbitration hearing, excluding "
            "counsel's fees. This is an estimate and not a cap.",
        ),
        ("space", ""),
        ("sub", "4. CONFLICTS"),
        (
            "body",
            "We act for Meridian Bulk Carriers SA on unrelated financing work. Meridian is a "
            "minority shareholder in Calder Shipping AG. An information barrier is in place "
            "between the two teams.",
        ),
        ("space", ""),
        ("body", "Perrine Duval, Partner, for and on behalf of Thorne Vaux LLP"),
    ],
    "NTL-2026-0114-advice-on-prospects.pdf": [
        ("head", "PRIVILEGED AND CONFIDENTIAL"),
        ("sub", "ADVICE ON PROSPECTS"),
        ("space", ""),
        ("body", "To: Ms Ada Okafor, General Counsel, Northwind Trading Limited"),
        ("body", "From: Kavi Iyer, Senior Associate, Thorne Vaux LLP"),
        ("body", "Date: 28 March 2026"),
        ("body", "Matter: NTL-2026-0114"),
        ("space", ""),
        ("sub", "1. SUMMARY"),
        (
            "body",
            "Northwind has a strong claim that Calder Shipping AG repudiated the charterparty "
            "by withdrawing MV Aurelia on 19 January 2026. We assess prospects on liability at "
            "70 to 75 per cent. Quantum is less certain.",
        ),
        ("space", ""),
        ("sub", "2. THE WITHDRAWAL"),
        (
            "body",
            "Clause 11(a) entitles owners to withdraw for non-payment of hire, but only after "
            "three banking days' notice. Calder gave notice on 16 January 2026 and withdrew on "
            "19 January 2026. Given the intervening weekend that is fewer than three banking days.",
        ),
        ("space", ""),
        ("sub", "3. THE OFF-HIRE DISPUTE"),
        (
            "body",
            "The engine logs disclosed on 22 March 2026 record the main engine inoperable from "
            "03:40 on 2 January until 18:15 on 13 January, which supports Northwind. The "
            "surveyor's report by Halvorsen Marine reaches the opposite conclusion but appears "
            "not to have had the logs.",
        ),
        ("space", ""),
        ("sub", "4. AUTHORITY"),
        (
            "body",
            "We rely on The Aquitaine [2019] EWHC 2211 (Comm) on the construction of clause 11 "
            "notice periods.",
        ),
        ("space", ""),
        ("sub", "5. QUANTUM"),
        (
            "body",
            "The charter rate was USD 18,500 per day against a market rate averaging USD 24,200 "
            "over the remaining 214 days, giving a prima facie loss near USD 1.22 million before "
            "mitigation. Northwind fixed the MV Sirocco on 4 February 2026.",
        ),
        ("space", ""),
        ("body", "Kavi Iyer, Senior Associate"),
    ],
    "NTL-2026-0114-conflict-memorandum.pdf": [
        ("head", "THORNE VAUX LLP: RISK AND COMPLIANCE"),
        ("sub", "CONFLICT CHECK MEMORANDUM"),
        ("space", ""),
        ("body", "Date: 6 March 2026"),
        ("body", "Prepared by: Rita Okonjo, Head of Risk"),
        ("body", "Matter: NTL-2026-0114"),
        ("space", ""),
        ("sub", "1. FINDINGS"),
        (
            "body",
            "The firm acts for Meridian Bulk Carriers SA on a secured lending facility, led by "
            "James Trelawney and open since 11 September 2024.",
        ),
        (
            "body",
            "Meridian Bulk Carriers SA holds 18 per cent of the issued share capital of Calder "
            "Shipping AG. Meridian is not a party to the dispute.",
        ),
        ("body", "The firm has never acted for Calder Shipping AG."),
        ("space", ""),
        ("sub", "2. ANALYSIS"),
        (
            "body",
            "Acting for Northwind against Calder does not put the firm in opposition to a current "
            "client. Meridian is a shareholder in the opposing party, not the opposing party. The "
            "residual risk is confidential financial information about Meridian reaching the "
            "disputes team.",
        ),
        ("space", ""),
        ("sub", "3. DECISION"),
        (
            "body",
            "The engagement may be accepted subject to an information barrier. Kavi Iyer, Adaeze "
            "Mensah and Perrine Duval are screened from the Meridian matter. James Trelawney is "
            "screened from matter NTL-2026-0114. Neither team may discuss the other's matter.",
        ),
        ("space", ""),
        ("body", "Rita Okonjo, Head of Risk"),
    ],
    "MBC-2024-0431-facility-agreement.pdf": [
        ("head", "FACILITY AGREEMENT: SUMMARY SHEET"),
        ("space", ""),
        ("body", "Matter: MBC-2024-0431"),
        ("body", "Client: Meridian Bulk Carriers SA"),
        ("body", "Partner: James Trelawney"),
        ("body", "Opened: 11 September 2024"),
        ("space", ""),
        ("sub", "1. THE FACILITY"),
        (
            "body",
            "Thorne Vaux LLP represents Meridian Bulk Carriers SA on a USD 240 million secured "
            "revolving facility arranged by Kestrel Bank AG. Meridian Bulk Carriers SA is our "
            "client on this matter.",
        ),
        ("space", ""),
        ("sub", "2. SECURITY"),
        (
            "body",
            "Security includes a first-priority mortgage over four vessels and a charge over "
            "Meridian's 18 per cent shareholding in Calder Shipping AG.",
        ),
        ("space", ""),
        ("sub", "3. INFORMATION BARRIER"),
        (
            "body",
            "This matter is subject to an information barrier raised on 6 March 2026. The disputes "
            "team acting on NTL-2026-0114 is screened from this matter and from the financial "
            "information it holds.",
        ),
        ("space", ""),
        ("body", "James Trelawney, Partner"),
    ],
    "HAL-2025-0092-authority-note.pdf": [
        ("head", "THORNE VAUX LLP: KNOW-HOW NOTE"),
        ("sub", "CHARTERPARTY WITHDRAWAL: THE AUTHORITIES"),
        ("space", ""),
        ("body", "Matter: HAL-2025-0092"),
        ("body", "Client: Halveston Chartering Limited"),
        ("body", "Author: Sian Aldridge"),
        ("body", "Date: 4 February 2026"),
        ("space", ""),
        ("sub", "1. THE CURRENT POSITION"),
        (
            "body",
            "The Marisol [2025] UKSC 14 overrules The Aquitaine [2019] EWHC 2211 (Comm) on the "
            "construction of notice periods in withdrawal clauses. The Supreme Court held that "
            "banking days exclude the day of service.",
        ),
        ("space", ""),
        ("sub", "2. CONSEQUENCE"),
        (
            "body",
            "Any advice relying on The Aquitaine for a notice-period calculation should be "
            "revisited. The change narrows rather than widens the window available to owners.",
        ),
        ("space", ""),
        ("sub", "3. RELATED PARTIES"),
        (
            "body",
            "Halveston Chartering Limited is our client on this matter. Calder Shipping AG has "
            "appeared as a counterparty in two unrelated Halveston fixtures.",
        ),
        ("space", ""),
        ("body", "Sian Aldridge"),
    ],
}


def build() -> Path:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, lines in DOCUMENTS.items():
            doc = pymupdf.open()
            _page(doc, lines)
            archive.writestr(name, doc.tobytes())
            doc.close()
    OUT_ZIP.write_bytes(buffer.getvalue())
    return OUT_ZIP


if __name__ == "__main__":
    path = build()
    size = path.stat().st_size
    print(f"wrote {path} ({size:,} bytes, {len(DOCUMENTS)} matters)")
