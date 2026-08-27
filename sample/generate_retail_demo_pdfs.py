"""Build the retail demo PDFs for the AnyCorp returns dataset.

The third sibling of `generate_demo_pdfs.py` (legal) and `generate_pack_demo_pdfs.py`
(healthcare, fintech). Separate for the same reason those two are: each pack's documents
have to interlock on *its own* rules, and there is nothing shared to factor out except
the page layout, which is imported rather than copied.

    .venv/bin/python sample/generate_retail_demo_pdfs.py

These four documents are written against a dataset that already exists. The workshop
loads an Aurora seed into Iceberg with Sam Parker as customer 47: nineteen purchases,
eight returns, a 42.10% return rate, a fraud risk score of 85 and an account status of
`under_review`. Every return in that table is a high-value electronics item, opened, sent
back on day 13 or 14 of a 14-day window. The scanned `return-policy.pdf` in the same
workshop is AnyCorp's real manual, and Section 6 of it lists "Pattern of returning
high-value electronics" as a red flag.

So the numbers here are not invented, and that is the point. A governed metric over
`returns` must reconcile with a fact extracted from a page, and it cannot if the two
describe different customers.

What each document is built to make happen, because a demo whose facts do not connect
renders identically to one that demonstrates nothing:

**`exception_on_superseded_policy`** -- the return approval relies on the Electronics
provision of Section 2 by name (14-day opened window, 15% restocking fee). Policy
Bulletin 2026-03, issued a fortnight earlier, withdrew that provision and replaced it
with a stricter one. Neither document mentions the other. The approval was wrong by
$299.99 and reads as routine on its own page.

**`exception_during_investigation`** -- loss prevention opened a file on Sam Parker on
6 March. The returns desk granted a goodwill refund to Sam Parker on 16 March. Two
ordinary records ten days apart in two systems, and the join is the whole finding.

**`related_party_resale`** -- the control chain is deliberately two links, Sam Parker ->
Northgate Holdings -> PixelPerfect Resale, so that walking `CONTROLS` as a bounded path
is what finds the storefront. A one-hop rule finds only Northgate Holdings, which is a
dormant holding company AnyCorp does not trade with, and misses the seller it actually
pays out to. The registered address is Sam Parker's own, which is the detail a reviewer
can check.

One absence is deliberate and load-bearing. **No document waives a fee on a return that
is in the warehouse table.** Every seeded return of Sam Parker's had its 15% fee charged
correctly -- $194.99 on a $1,299.99 television, and so on. The refund here is a ninth
return, in 2026, after the seeded history ends. A document contradicting the rows would
make the demo incoherent rather than interesting.

Ids are left to the extractor. These documents name entities in prose the way a real one
does, because the point is to exercise extraction rather than to hand it a pre-parsed
answer.
"""

from __future__ import annotations

import importlib.util
import io
import zipfile
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent

# By path, because `sample/` is not a package and these scripts are run as files. Importing the
# layout rather than copying it keeps one definition of what a demo page looks like.
_spec = importlib.util.spec_from_file_location("_legal_demo", HERE / "generate_demo_pdfs.py")
assert _spec is not None and _spec.loader is not None
_legal_demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_legal_demo)
_page = _legal_demo._page

RETAIL: dict[str, list[tuple[str, str]]] = {
    # ── The supersession. Issued before the approval below, and never referenced by it. ──
    "policy-bulletin-2026-03.pdf": [
        ("head", "ANYCORP RETAIL"),
        ("body", "Returns and Customer Care -- Policy Bulletin"),
        ("space", ""),
        ("sub", "POLICY BULLETIN 2026-03"),
        ("body", "Issued: 2 March 2026"),
        ("body", "Effective: 9 March 2026"),
        ("body", "Applies to: all stores, contact centre and online returns"),
        ("body", "Issued by: Delia Marchetti, Director of Returns Policy"),
        ("space", ""),
        ("sub", "1. PURPOSE"),
        (
            "body",
            "Policy Bulletin 2026-03 amends the Return Policy Manual 2025. Opened electronics "
            "accounted for 61% of refund value and 74% of confirmed abuse cases in the 2025 "
            "financial year. The Electronics provision of Section 2, Return Windows by Category, "
            "is the provision under which almost all of that value left the business.",
        ),
        ("space", ""),
        ("sub", "2. PROVISION WITHDRAWN"),
        (
            "body",
            "Policy Bulletin 2026-03 supersedes the Electronics opened-returns provision of "
            "Section 2 of the Return Policy Manual 2025. That provision allowed an opened "
            "electronics item to be returned within a 14-day window from the purchase date "
            "subject to a 15% restocking fee. It is withdrawn in full and may not be relied on "
            "for any return accepted on or after 9 March 2026.",
        ),
        (
            "body",
            "The unopened Electronics provision is unchanged. A sealed item in original packaging "
            "is still refunded in full within 30 days with no restocking fee.",
        ),
        ("space", ""),
        ("sub", "3. PROVISION SUBSTITUTED"),
        (
            "body",
            "An opened electronics item is not returnable except where the item is faulty. Where "
            "a fault is established the return is handled under Section 5, Defective Items, at no "
            "restocking fee.",
        ),
        (
            "body",
            "Where an opened electronics item is accepted on any other ground, the return requires "
            "store manager approval recorded on the return record, and a restocking fee of 25% "
            "applies. The 25% fee may not be waived at desk level.",
        ),
        ("space", ""),
        ("sub", "4. FRAUD PREVENTION UNAFFECTED"),
        (
            "body",
            "Section 6, Fraud Prevention, is unaffected by Policy Bulletin 2026-03 and remains in "
            "force in its entirety.",
        ),
        ("space", ""),
        ("body", "Delia Marchetti, Director of Returns Policy, AnyCorp Retail"),
    ],
    # ── The open case. Names the red flag the warehouse rows actually support. ──
    "LP-2026-0088-investigation-memo.pdf": [
        ("head", "ANYCORP RETAIL -- LOSS PREVENTION"),
        ("body", "Confidential. Internal distribution only."),
        ("space", ""),
        ("sub", "CASE OPENING MEMORANDUM"),
        ("body", "Case: LP-2026-0088"),
        ("body", "Opened: 6 March 2026"),
        ("body", "Subject: Sam Parker, account 47, 980 Maple St, Omaha, NE 68101"),
        ("body", "Prepared by: Ada Okonjo, Loss Prevention Analyst"),
        ("space", ""),
        ("sub", "1. GROUND FOR OPENING"),
        (
            "body",
            "Case LP-2026-0088 investigates Sam Parker for suspected return abuse. The ground is "
            "the red flag at Section 6 of the Return Policy Manual 2025, Red Flags for Fraudulent "
            "Returns, item 2: \"Pattern of returning high-value electronics\".",
        ),
        (
            "body",
            "Sam Parker has nineteen recorded purchases and eight returns, a return rate of "
            "42.10%. The account carries a fraud risk score of 85 and its status is under review. "
            "Three suspicious-activity notes are on file.",
        ),
        ("space", ""),
        ("sub", "2. PATTERN"),
        (
            "body",
            "Every one of the eight returns was an electronics item in opened condition. Seven of "
            "the eight were lodged on day 13 or day 14 of the 14-day opened window. The items "
            "were a Samsung 65\" QLED TV on three separate occasions, an Apple iPhone 15 Pro "
            "twice, a Sony PlayStation 5, a MacBook Pro 14\" and a set of Bose Headphones.",
        ),
        (
            "body",
            "Stated reasons vary and do not corroborate one another: changed mind, not as "
            "described, defective, too expensive, found better price. The defective claim on the "
            "Sony PlayStation 5 was the only one to attract no restocking fee.",
        ),
        ("space", ""),
        ("sub", "3. PAYMENTS"),
        (
            "body",
            "A chargeback was raised against order ORD-2024-00008 after the refund had already "
            "been issued. Loss Prevention treats a chargeback on a refunded order as an "
            "indicator, not a finding.",
        ),
        ("space", ""),
        ("sub", "4. WHAT THIS MEMORANDUM DOES NOT SAY"),
        (
            "body",
            "No conclusion is drawn. Sam Parker has not been notified and no privilege is "
            "suspended. Case LP-2026-0088 remains open pending review of the resale enquiry "
            "running separately.",
        ),
        ("space", ""),
        ("body", "Ada Okonjo, Loss Prevention Analyst"),
    ],
    # ── The exception. Relies on the withdrawn provision by name; open case not consulted. ──
    "LP-2026-0088-return-approval.pdf": [
        ("head", "ANYCORP RETAIL"),
        ("body", "Returns Desk -- Goodwill Approval Record"),
        ("space", ""),
        ("sub", "RETURN APPROVAL RTN-2026-00912"),
        ("body", "Date: 16 March 2026"),
        ("body", "Customer: Sam Parker, account 47"),
        ("body", "Order: ORD-2026-04417, placed 2 March 2026"),
        ("body", "Item: Bose Headphones, 299.99 USD, opened, all accessories present"),
        ("body", "Approved by: Curtis Lindgren, Returns Desk Supervisor"),
        ("space", ""),
        ("sub", "1. DECISION"),
        (
            "body",
            "Return RTN-2026-00912 is approved in favour of Sam Parker. The full purchase price "
            "of 299.99 USD is refunded to the original payment method. The restocking fee is "
            "waived in full as a goodwill gesture.",
        ),
        ("space", ""),
        ("sub", "2. AUTHORITY RELIED ON"),
        (
            "body",
            "This approval is made under the Electronics opened-returns provision of Section 2, "
            "Return Windows by Category, of the Return Policy Manual 2025: a 14-day window from "
            "the purchase date, subject to a 15% restocking fee. Return RTN-2026-00912 was lodged "
            "on day 14 and so falls inside that window.",
        ),
        (
            "body",
            "The 15% restocking fee of 44.99 USD is waived under the Refund Adjustments note at "
            "Section 4, which permits a fee to be waived where a supervisor records a reason. The "
            "reason recorded is customer goodwill.",
        ),
        ("space", ""),
        ("sub", "3. CUSTOMER STANDING"),
        (
            "body",
            "Sam Parker is described on the account as a long-standing customer at Silver loyalty "
            "tier with a lifetime value of 12,450 USD. The desk treated that history as the "
            "reason for goodwill.",
        ),
        ("space", ""),
        ("sub", "4. CHECKS PERFORMED"),
        (
            "body",
            "Receipt verified. Serial number matched the receipt. Packaging intact. No manager "
            "escalation was raised and no loss prevention check was requested.",
        ),
        ("space", ""),
        ("body", "Curtis Lindgren, Returns Desk Supervisor, Store 118, Omaha NE"),
    ],
    # ── The ownership chain. Two links on purpose. ──
    "merchant-onboarding-pixelperfect.pdf": [
        ("head", "ANYCORP MARKETPLACE"),
        ("body", "Seller Onboarding -- Approval Memorandum"),
        ("space", ""),
        ("sub", "SELLER APPROVAL MEM-2026-0231"),
        ("body", "Date: 11 February 2026"),
        ("body", "Seller: PixelPerfect Resale"),
        ("body", "Category: Electronics, refurbished and open-box"),
        ("body", "Prepared by: Naomi Ferreira, Marketplace Onboarding"),
        ("space", ""),
        ("sub", "1. APPROVAL"),
        (
            "body",
            "PixelPerfect Resale is approved to trade on AnyCorp Marketplace from 18 February "
            "2026. PixelPerfect Resale sells refurbished consumer electronics. The listings "
            "submitted at application were a Samsung 65\" QLED TV, an Apple iPhone 15 Pro and a "
            "Sony PlayStation 5, all described as open-box.",
        ),
        ("space", ""),
        ("sub", "2. OWNERSHIP"),
        (
            "body",
            "PixelPerfect Resale is wholly owned by Northgate Holdings. Northgate Holdings is a "
            "non-trading company whose sole director is Sam Parker. Northgate Holdings controls "
            "PixelPerfect Resale and holds no other subsidiary.",
        ),
        (
            "body",
            "Sam Parker controls Northgate Holdings. No further beneficial owner was declared on "
            "the application.",
        ),
        ("space", ""),
        ("sub", "3. REGISTERED ADDRESS"),
        (
            "body",
            "Northgate Holdings and PixelPerfect Resale share a registered address of 980 Maple "
            "St, Omaha, NE 68101. The address is residential.",
        ),
        ("space", ""),
        ("sub", "4. CHECKS PERFORMED"),
        (
            "body",
            "Company registration verified. Bank account verified in the name of Northgate "
            "Holdings. Tax identifier supplied. Onboarding does not screen a declared director "
            "against the customer file, and no such check was performed here.",
        ),
        ("space", ""),
        ("body", "Naomi Ferreira, Marketplace Onboarding, AnyCorp Marketplace"),
    ],
}


def build() -> tuple[Path, Path]:
    """Write the loose PDFs and the zip. Both, because they are used differently.

    The zip is what a person uploads through the Documents page in one go. The loose
    directory is what the workshop serves as downloadable assets, and what a diff shows
    when one document's wording changes.
    """
    loose = HERE / "retail-demo"
    loose.mkdir(exist_ok=True)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, lines in RETAIL.items():
            doc = pymupdf.open()
            _page(doc, lines)
            payload = doc.tobytes()
            doc.close()
            archive.writestr(name, payload)
            (loose / name).write_bytes(payload)

    out = HERE / "retail-demo.zip"
    out.write_bytes(buffer.getvalue())
    return out, loose


if __name__ == "__main__":
    zip_path, loose_dir = build()
    print(f"wrote {zip_path} ({zip_path.stat().st_size:,} bytes, {len(RETAIL)} documents)")
    for name in RETAIL:
        print(f"  {loose_dir / name}")
