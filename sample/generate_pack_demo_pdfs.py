"""Build the healthcare and fintech demo PDFs, one zip per pack.

The legal demo has `generate_demo_pdfs.py`. This is its sibling for the other two packs, and
the reason it exists separately is that each pack's documents have to interlock on *its own*
rules -- a healthcare document cannot demonstrate a conflict check, and there is nothing
shared to factor out except the page layout, which is imported rather than copied.

    .venv/bin/python sample/generate_pack_demo_pdfs.py

What each set is built to make happen, because a demo whose facts do not connect renders
identically to one that demonstrates nothing:

**healthcare** -- `contraindication_via_ingredient` is the three-premise rule, the analogue of
the legal pack's `conflict_via_affiliate`. So one document records the allergy against a brand
name, a second prescribes a *different* brand, and a third is the formulary note saying the two
share an active ingredient. The direct rule (`contraindication_alert`) cannot fire on those three
because no one document has both ends; only the ingredient path finds it. That is the whole
point: the risk is real and invisible to a single-document reader.

**fintech** -- `group_exposure_via_control` walks `CONTROLS` as a bounded path, so the control
chain is deliberately two links (Meridian -> Pallas -> Calder Freight) to prove an intermediate
holding company does not hide the group. `related_party_lending` is the three-premise rule and
needs a guarantor who is also a related party, which the credit memo records.

Ids are left to the extractor. These documents name entities in prose the way a real one does,
because the point is to exercise extraction rather than to hand it a pre-parsed answer.
"""

from __future__ import annotations

import importlib.util
import io
import zipfile
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent

# By path, because `sample/` is not a package and both scripts are run as files. Importing the
# layout rather than copying it keeps one definition of what a demo page looks like.
_spec = importlib.util.spec_from_file_location("_legal_demo", HERE / "generate_demo_pdfs.py")
assert _spec is not None and _spec.loader is not None
_legal_demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_legal_demo)
_page = _legal_demo._page

HEALTHCARE: dict[str, list[tuple[str, str]]] = {
    "ENC-2026-0451-admission-note.pdf": [
        ("head", "ST OSWALD'S TEACHING HOSPITAL"),
        ("body", "Department of Acute Medicine"),
        ("space", ""),
        ("sub", "ADMISSION NOTE"),
        ("body", "Encounter: ENC-2026-0451"),
        ("body", "Patient: Mr Tobias Renner, DOB 14 May 1961, hospital number 88-40217"),
        ("body", "Admitted: 3 February 2026, 21:40"),
        ("body", "Clinician: Dr Aoife Brennan, Consultant Physician"),
        ("space", ""),
        ("sub", "1. PRESENTING COMPLAINT"),
        (
            "body",
            "Mr Tobias Renner presented with a productive cough, fever of 38.9C and right basal "
            "crackles. Chest radiography shows consolidation of the right lower lobe. "
            "Mr Tobias Renner has a diagnosis of community acquired pneumonia.",
        ),
        ("space", ""),
        ("sub", "2. ALLERGY HISTORY"),
        (
            "body",
            "Mr Tobias Renner is allergic to Amoxil. The reaction recorded in 2019 was urticaria "
            "with facial swelling, which the patient describes as requiring emergency treatment. "
            "This allergy is to be treated as absolute.",
        ),
        (
            "body",
            "No other drug allergy is reported. The patient is unable to name the drug class and "
            "recalls only the brand on the packet.",
        ),
        ("space", ""),
        ("sub", "3. RELEVANT HISTORY"),
        (
            "body",
            "Type 2 diabetes, diagnosed 2011. Hypertension. No known renal impairment, with an "
            "eGFR of 74 on admission bloods.",
        ),
        ("space", ""),
        ("body", "Dr Aoife Brennan, Consultant Physician"),
    ],
    "ENC-2026-0451-prescription-chart.pdf": [
        ("head", "ST OSWALD'S TEACHING HOSPITAL"),
        ("body", "Inpatient Prescription Chart"),
        ("space", ""),
        ("body", "Encounter: ENC-2026-0451"),
        ("body", "Patient: Mr Tobias Renner, hospital number 88-40217"),
        ("body", "Date: 4 February 2026"),
        ("space", ""),
        ("sub", "1. REGULAR MEDICATION"),
        (
            "body",
            "Dr Ravi Nadkarni prescribed Almodan 500mg three times daily for seven days, to treat "
            "the community acquired pneumonia. First dose given at 08:15 on 4 February 2026.",
        ),
        (
            "body",
            "Metformin 1g twice daily, continued from the patient's own supply. Ramipril 5mg once "
            "daily, continued.",
        ),
        ("space", ""),
        ("sub", "2. AS REQUIRED"),
        (
            "body",
            "Paracetamol 1g up to four times daily for fever. Dr Ravi Nadkarni prescribed "
            "Paracetamol on the same round.",
        ),
        ("space", ""),
        ("sub", "3. NOTE FROM PHARMACY"),
        (
            "body",
            "The admission note records an allergy to a penicillin brand. The prescribing entry "
            "here uses a different brand name and the chart carries no cross-check. Pharmacy has "
            "asked for confirmation before the second dose.",
        ),
        ("space", ""),
        ("body", "Dr Ravi Nadkarni, Specialty Registrar"),
    ],
    "FORM-2026-penicillin-formulary-note.pdf": [
        ("head", "ST OSWALD'S TEACHING HOSPITAL"),
        ("body", "Medicines Information Service"),
        ("space", ""),
        ("sub", "FORMULARY NOTE: PENICILLIN BRAND EQUIVALENCE"),
        ("body", "Reference: FORM-2026-014"),
        ("body", "Issued: 9 January 2026"),
        ("body", "Author: Ms Priya Raghunathan, Lead Antimicrobial Pharmacist"),
        ("space", ""),
        ("sub", "1. PURPOSE"),
        (
            "body",
            "Several penicillin products are stocked under different brand names. Allergy "
            "histories are frequently recorded against a brand the patient remembers rather than "
            "against the active ingredient, so a brand-to-brand check will miss a real allergy.",
        ),
        ("space", ""),
        ("sub", "2. EQUIVALENCE"),
        (
            "body",
            "Almodan has the same active ingredient as Amoxil. Both are amoxicillin "
            "preparations and differ only in presentation and manufacturer.",
        ),
        (
            "body",
            "Amoxil has the same active ingredient as Amix. Any allergy recorded against one of "
            "these brands must be treated as applying to all of them.",
        ),
        ("space", ""),
        ("sub", "3. ACTION"),
        (
            "body",
            "Prescribers must check the active ingredient and not the brand. This note is issued "
            "because two incidents in the last year involved a brand-name mismatch of exactly "
            "this kind.",
        ),
        ("space", ""),
        ("body", "Ms Priya Raghunathan, Lead Antimicrobial Pharmacist"),
    ],
    "ENC-2026-0388-consent-form.pdf": [
        ("head", "ST OSWALD'S TEACHING HOSPITAL"),
        ("body", "Department of General Surgery"),
        ("space", ""),
        ("sub", "CONSENT FORM"),
        ("body", "Encounter: ENC-2026-0388"),
        ("body", "Patient: Mrs Helena Voss, DOB 2 October 1974, hospital number 71-33095"),
        ("body", "Date: 22 January 2026"),
        ("space", ""),
        ("sub", "1. PROCEDURE"),
        (
            "body",
            "Mrs Helena Voss has given consent for laparoscopic cholecystectomy. The procedure, "
            "its benefits and the risk of conversion to open surgery were explained in full.",
        ),
        ("space", ""),
        ("sub", "2. DIAGNOSIS"),
        (
            "body",
            "Mrs Helena Voss has a diagnosis of symptomatic cholelithiasis, confirmed on "
            "ultrasound dated 14 January 2026.",
        ),
        ("space", ""),
        ("sub", "3. OPERATOR"),
        (
            "body",
            "Ms Beatrix Halloran, Consultant Surgeon, performed laparoscopic cholecystectomy on "
            "23 January 2026. There were no intraoperative complications.",
        ),
        ("space", ""),
        ("body", "Countersigned: Ms Beatrix Halloran, Consultant Surgeon"),
    ],
}

FINTECH: dict[str, list[tuple[str, str]]] = {
    "FAC-2026-0210-credit-agreement.pdf": [
        ("head", "HALDANE STREET BANK PLC"),
        ("body", "Corporate Lending, 3 Threadneedle Court, London EC2R 8BB"),
        ("space", ""),
        ("sub", "CREDIT AGREEMENT SUMMARY"),
        ("body", "Facility reference: FAC-2026-0210"),
        ("body", "Date: 6 February 2026"),
        ("space", ""),
        ("sub", "1. THE FACILITY"),
        (
            "body",
            "Facility FAC-2026-0210 lends to Calder Freight Holdings NV. The facility is a "
            "revolving credit facility of EUR 180,000,000 with a final maturity of "
            "6 February 2031.",
        ),
        (
            "body",
            "The facility is governed by the law of England and Wales. Drawings are permitted in "
            "euro and sterling.",
        ),
        ("space", ""),
        ("sub", "2. GUARANTEE"),
        (
            "body",
            "Meridian Bulk Carriers SA guarantees facility FAC-2026-0210. The guarantee is "
            "unlimited in amount and continues until all obligations under the facility are "
            "discharged.",
        ),
        ("space", ""),
        ("sub", "3. COVENANTS"),
        (
            "body",
            "Facility FAC-2026-0210 is subject to a net leverage covenant of 3.5x, tested "
            "quarterly. It is also subject to an information covenant requiring audited accounts "
            "within 120 days of each financial year end.",
        ),
        ("space", ""),
        ("sub", "4. LIMITS"),
        (
            "body",
            "This facility is assessed against the large exposures regime. Aggregate exposure to "
            "a group of connected clients may not exceed 25 per cent of Tier 1 capital.",
        ),
        ("space", ""),
        ("body", "For and on behalf of Haldane Street Bank plc"),
    ],
    "FAC-2026-0210-credit-committee-memo.pdf": [
        ("head", "HALDANE STREET BANK PLC"),
        ("body", "Credit Risk and Compliance"),
        ("space", ""),
        ("sub", "CREDIT COMMITTEE MEMORANDUM"),
        ("body", "Facility: FAC-2026-0210"),
        ("body", "Date: 30 January 2026"),
        ("body", "Prepared by: Mr Emeka Adeyemi, Head of Credit Risk"),
        ("space", ""),
        ("sub", "1. PROPOSED EXPOSURE"),
        (
            "body",
            "Calder Freight Holdings NV has requested a EUR 180,000,000 revolving facility. The "
            "borrower is a logistics group operating across northern Europe.",
        ),
        ("space", ""),
        ("sub", "2. OWNERSHIP"),
        (
            "body",
            "Pallas Maritime Group BV controls Calder Freight Holdings NV, holding 68 per cent of "
            "its issued share capital.",
        ),
        (
            "body",
            "Meridian Bulk Carriers SA controls Pallas Maritime Group BV. Meridian holds 91 per "
            "cent of Pallas and appoints a majority of its board.",
        ),
        ("space", ""),
        ("sub", "3. RELATED PARTIES"),
        (
            "body",
            "Meridian Bulk Carriers SA is a related party of Calder Freight Holdings NV. Two "
            "directors sit on both boards and the treasury function is shared.",
        ),
        ("space", ""),
        ("sub", "4. EXISTING BOOK"),
        (
            "body",
            "The bank already has EUR 240,000,000 of drawn exposure to Meridian Bulk Carriers SA "
            "under facility FAC-2024-0071. That facility is performing and was last reviewed in "
            "November 2025.",
        ),
        ("space", ""),
        ("sub", "5. RECOMMENDATION"),
        (
            "body",
            "Approve, subject to the group limit being recalculated on a connected-client basis "
            "before first drawdown. The individual limits are each within policy; the group "
            "position has not been aggregated.",
        ),
        ("space", ""),
        ("body", "Mr Emeka Adeyemi, Head of Credit Risk"),
    ],
    "FAC-2024-0071-covenant-certificate.pdf": [
        ("head", "MERIDIAN BULK CARRIERS SA"),
        ("body", "Group Treasury, 40 Rue Sainte-Barbe, Luxembourg"),
        ("space", ""),
        ("sub", "COVENANT COMPLIANCE CERTIFICATE"),
        ("body", "Facility: FAC-2024-0071"),
        ("body", "Testing date: 31 December 2025"),
        ("body", "Delivered: 18 February 2026"),
        ("space", ""),
        ("sub", "1. CERTIFICATION"),
        (
            "body",
            "Facility FAC-2024-0071 lends to Meridian Bulk Carriers SA. This certificate is "
            "delivered under the information covenant of that facility.",
        ),
        ("space", ""),
        ("sub", "2. LEVERAGE TEST"),
        (
            "body",
            "Net leverage at the testing date was 3.9x against a covenanted maximum of 3.5x. "
            "Meridian Bulk Carriers SA breached the net leverage covenant at 31 December 2025.",
        ),
        (
            "body",
            "The breach is attributed to a fall in charter rates in the fourth quarter and to the "
            "acquisition of two vessels completed in November 2025.",
        ),
        ("space", ""),
        ("sub", "3. INFORMATION COVENANT"),
        (
            "body",
            "Audited accounts for the year ended 31 December 2025 are due on 30 April 2026. The "
            "information covenant is due on 30 April 2026.",
        ),
        ("space", ""),
        ("body", "For and on behalf of Meridian Bulk Carriers SA, Group Treasury"),
    ],
}

PACKS: dict[str, dict[str, list[tuple[str, str]]]] = {
    "healthcare": HEALTHCARE,
    "fintech": FINTECH,
}


def build() -> list[Path]:
    written = []
    for pack, documents in PACKS.items():
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, lines in documents.items():
                doc = pymupdf.open()
                _page(doc, lines)
                archive.writestr(name, doc.tobytes())
                doc.close()
        out = HERE / f"{pack}-demo.zip"
        out.write_bytes(buffer.getvalue())
        written.append(out)
    return written


if __name__ == "__main__":
    for path in build():
        docs = len(PACKS[path.stem.replace("-demo", "")])
        print(f"wrote {path} ({path.stat().st_size:,} bytes, {docs} documents)")
