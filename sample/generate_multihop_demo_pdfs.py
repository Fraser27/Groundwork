"""Build a harder demo tier: multi-page documents designed to defeat plain vector RAG.

The existing packs (`generate_demo_pdfs.py`, `generate_pack_demo_pdfs.py`) each demonstrate
one rule with the minimum number of documents that can carry it. This script raises the
difficulty on purpose, for demos whose point is to show *why* a graph is needed rather than
merely that one exists:

1. **Longer chains.** Each pack's control/affiliate/ingredient path is walked to the full
   `*1..3` bound the ontology allows, not a single hop. Fintech's control chain is three links
   deep (Halcyon -> Ridgeway -> Corvus -> Solenne) -- one below the ceiling the rule enforces.

2. **More documents per fact, none sufficient alone.** Every rule's premises are split across
   three or more documents. No single passage, and no single document, contains enough of the
   chain to answer the question -- which is what makes it unanswerable by nearest-neighbour
   retrieval over chunks, however good the embedding is.

3. **Decoys that actively mislead a similarity search, not just fail to help it.** Each pack
   includes a name that is lexically near-identical to a real entity but legally distinct
   (`Kestrel Fabrication Group` vs `Kestrel Fabrication Systems Ltd`; `Ridgeway Capital
   Partners NV` vs `...LLC`; `Zenapril` vs `Zentanyl`), so a system matching on text similarity
   is liable to conflate them. The documents disambiguate explicitly, the way a real registry
   extract or formulary note would -- which is exactly the fact a plain-RAG hit would skip past
   in favour of the more prominent, wrong-entity sentence.

4. **A document that states the wrong conclusion.** The fintech credit committee memo says in
   its own text that the existing Halcyon exposure "is not otherwise connected to this
   proposal" -- which is false once the control chain in two *other* documents is followed. A
   system that trusts the most relevant-looking passage will trust that sentence. A system that
   traverses `CONTROLS*1..3` will not.

Each pack still interlocks on real rules from `ontologies/*.yaml`:

  legal      conflict_via_affiliate (AFFILIATE_OF*1..3, two links) + authority_stale
  healthcare contraindication_via_ingredient (three premises, two encounters) +
             a same-chart CONTRAINDICATED_WITH pair that needs a fourth document to assert
  fintech    group_exposure_via_control (CONTROLS*1..3, three links, at the bound) +
             related_party_lending (three premises)
  retail     exception_on_superseded_policy (SUPERSEDES, two-link revision chain) +
             exception_during_investigation, with a status note that states the wrong
             conclusion + related_party_resale (CONTROLS*1..3, three links, at the bound)

    .venv/bin/python sample/generate_multihop_demo_pdfs.py
"""

from __future__ import annotations

import importlib.util
import io
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

# By path, for the same reason generate_pack_demo_pdfs.py imports this way: `sample/` is not a
# package, and importing the layout rather than copying it keeps one definition of what a
# demo page looks like.
_spec = importlib.util.spec_from_file_location("_legal_demo", HERE / "generate_demo_pdfs.py")
assert _spec is not None and _spec.loader is not None
_legal_demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_legal_demo)
_page = _legal_demo._page
pymupdf = _legal_demo.pymupdf

Page = list[tuple[str, str]]
Document = list[Page]

# ─────────────────────────────────────────────────────────────────────────────────────────────
# LEGAL — conflict_via_affiliate walked to two links, plus authority_stale with a decoy case.
#
# Halberd Quinn LLP represents Solenne Materials Ltd against Kestrel Fabrication Group
# (matter SOL-2026-0207). Unrelated matter VIP-2025-0388: the firm represents Vantage
# Industrial Partners, which holds 40% of Solmark Ventures BV -- and Solmark Ventures BV
# holds 65% of Kestrel Fabrication Group. Vantage is an affiliate of Kestrel two links away,
# and each link is asserted in a different document from the one asserting REPRESENTS or
# ADVERSE_TO. `conflict_via_affiliate` fires: (SOL-2026-0207)-[:POTENTIAL_CONFLICT]->(Vantage).
#
# The decoy: Kestrel Fabrication Systems Ltd, an unrelated English company with a name close
# enough to collide in a vector index, explicitly disambiguated in the registry extract.
# ─────────────────────────────────────────────────────────────────────────────────────────────

LEGAL: dict[str, Document] = {
    "SOL-2026-0207-engagement-letter.pdf": [
        [
            ("head", "HALBERD QUINN LLP"),
            ("body", "16 Fetter Lane, London EC4A 1BR"),
            ("space", ""),
            ("body", "14 April 2026"),
            ("space", ""),
            ("body", "Solenne Materials Ltd, Attn: Mr Callum Bright, Group General Counsel"),
            ("space", ""),
            ("sub", "ENGAGEMENT: SOLENNE MATERIALS LTD v KESTREL FABRICATION GROUP"),
            ("body", "Our reference: SOL-2026-0207"),
            ("space", ""),
            (
                "body",
                "Halberd Quinn LLP represents Solenne Materials Ltd in the dispute concerning "
                "the supply agreement dated 2 September 2025 for structural steel roof trusses "
                "supplied to Solenne's Deeside fabrication yard. Solenne Materials Ltd is our "
                "client. Kestrel Fabrication Group is the adverse party.",
            ),
            ("space", ""),
            ("sub", "1. SCOPE"),
            (
                "body",
                "We will advise on and, if instructed, conduct proceedings arising from Kestrel "
                "Fabrication Group's rejection of the final two delivery batches under the "
                "supply agreement, and Solenne's cross-claim for the resulting delay to the "
                "Deeside roofing programme.",
            ),
            (
                "body",
                "We do not advise on the separate insurance claim Solenne has notified to its "
                "contract works policy in respect of the same delay. That exclusion is "
                "deliberate and survives any variation of this engagement.",
            ),
            ("space", ""),
            ("sub", "2. BACKGROUND"),
            (
                "body",
                "Kestrel Fabrication Group rejected batches 14 and 15 on 20 January 2026, citing "
                "non-conforming weld certification. Solenne disputes the rejection and says the "
                "certification supplied met the specification agreed in the order confirmation "
                "of 2 September 2025.",
            ),
        ],
        [
            ("sub", "Matter SOL-2026-0207 — continued"),
            ("sub", "3. THE TEAM"),
            (
                "body",
                "Ms Ines Ferraro, partner, supervises. Mr Tomas Whitlock, senior associate, has "
                "day-to-day conduct.",
            ),
            ("space", ""),
            ("sub", "4. FEES"),
            (
                "body",
                "Partner GBP 610, senior associate GBP 395, associate GBP 285. We estimate GBP "
                "95,000 to the conclusion of any adjudication, excluding counsel's fees. This is "
                "an estimate and not a cap.",
            ),
            ("space", ""),
            ("sub", "5. CONFLICTS"),
            (
                "body",
                "We have checked our current adverse-party register and confirm that the firm "
                "has no existing engagement for or against Kestrel Fabrication Group. No "
                "conflict has been identified on that basis.",
            ),
            ("space", ""),
            ("body", "Ines Ferraro, Partner, for and on behalf of Halberd Quinn LLP"),
        ],
    ],
    "SOL-2026-0207-advice-on-prospects.pdf": [
        [
            ("head", "PRIVILEGED AND CONFIDENTIAL"),
            ("sub", "ADVICE ON PROSPECTS"),
            ("space", ""),
            ("body", "To: Mr Callum Bright, Group General Counsel, Solenne Materials Ltd"),
            ("body", "From: Tomas Whitlock, Senior Associate, Halberd Quinn LLP"),
            ("body", "Date: 6 May 2026"),
            ("body", "Matter: SOL-2026-0207"),
            ("space", ""),
            ("sub", "1. SUMMARY"),
            (
                "body",
                "Solenne has good prospects of establishing that the weld certification "
                "supplied for batches 14 and 15 met the specification in the order confirmation "
                "of 2 September 2025. We assess prospects on liability at 65 to 70 per cent.",
            ),
            ("space", ""),
            ("sub", "2. THE REJECTION"),
            (
                "body",
                "Kestrel Fabrication Group's rejection notice of 20 January 2026 relies on "
                "clause 9.2 of the supply agreement, which permits rejection for certification "
                "that does not conform to EN 1090-2. The certification supplied by Solenne's "
                "approved welder, Bracken NDT Services, records conformity with EN 1090-2 "
                "Execution Class 3.",
            ),
            (
                "body",
                "Kestrel's own quality manager accepted an equivalent certification format on "
                "batches 1 through 13 without objection, which weakens any argument that the "
                "format itself was defective.",
            ),
            ("space", ""),
            ("sub", "3. NOTICE PERIOD"),
            (
                "body",
                "Clause 14.3 requires a rejection notice within ten working days of delivery. "
                "Batch 15 was delivered on 5 January 2026. Ten working days, excluding the day "
                "of delivery, expired on 19 January 2026. The rejection notice is dated 20 "
                "January 2026, one day late on that construction.",
            ),
        ],
        [
            ("sub", "Matter SOL-2026-0207 — continued"),
            ("sub", "4. AUTHORITY"),
            (
                "body",
                "We rely on Harrow Timber Co v Bellamy Frères [2018] EWCA Civ 1140 on the "
                "construction of working-day notice periods, which held that the day of "
                "delivery is excluded from the count.",
            ),
            ("space", ""),
            ("sub", "5. QUANTUM"),
            (
                "body",
                "The direct cost of remedial fabrication is GBP 340,000. Solenne also claims "
                "GBP 210,000 for delay to the Deeside programme, calculated at the daily rate "
                "under the main contract with the ultimate developer.",
            ),
        ],
        [
            ("sub", "Matter SOL-2026-0207 — continued"),
            ("sub", "6. KESTREL'S POSITION"),
            (
                "body",
                "Kestrel Fabrication Group is a mid-sized structural steel fabricator. It is "
                "understood to sit within a wider group structure, though the precise ownership "
                "above Kestrel has not been confirmed for the purposes of this advice and does "
                "not affect the analysis above.",
            ),
            ("space", ""),
            ("sub", "7. RECOMMENDATION"),
            (
                "body",
                "We recommend proceeding to adjudication if the commercial teams cannot resolve "
                "the certification dispute directly.",
            ),
            ("space", ""),
            ("body", "Tomas Whitlock, Senior Associate"),
        ],
    ],
    "VIP-2025-0388-engagement-letter.pdf": [
        [
            ("head", "HALBERD QUINN LLP"),
            ("body", "16 Fetter Lane, London EC4A 1BR"),
            ("space", ""),
            ("body", "3 November 2025"),
            ("space", ""),
            ("body", "Vantage Industrial Partners, Attn: Ms Elodie Marchetti, Chief Financial Officer"),
            ("space", ""),
            ("sub", "ENGAGEMENT: PROJECT HARBOUR — JOINT VENTURE FINANCING"),
            ("body", "Our reference: VIP-2025-0388"),
            ("space", ""),
            (
                "body",
                "Halberd Quinn LLP represents Vantage Industrial Partners in connection with "
                "the financing of a new joint venture vehicle, Solmark Ventures BV, formed to "
                "acquire and operate a bulk-handling terminal. Vantage Industrial Partners is "
                "our client.",
            ),
            ("space", ""),
            ("sub", "1. SCOPE"),
            (
                "body",
                "We will advise on the shareholders' agreement, the senior facility supporting "
                "the acquisition, and related security. Vantage Industrial Partners will hold a "
                "40 per cent interest in Solmark Ventures BV alongside two other investors, "
                "whose interests are not the subject of this engagement.",
            ),
            (
                "body",
                "We do not advise on the operational permits required for the terminal, which "
                "Vantage has instructed local counsel to handle directly.",
            ),
        ],
        [
            ("sub", "Matter VIP-2025-0388 — continued"),
            ("sub", "2. THE TEAM"),
            (
                "body",
                "Mr Dorian Aske, partner, supervises. Ms Freya Lindqvist, associate, has "
                "day-to-day conduct.",
            ),
            ("space", ""),
            ("sub", "3. FEES"),
            (
                "body",
                "Partner GBP 610, associate GBP 285. We estimate GBP 60,000 to financial close, "
                "excluding disbursements.",
            ),
            ("space", ""),
            ("sub", "4. SOLMARK VENTURES BV"),
            (
                "body",
                "Solmark Ventures BV was incorporated in the Netherlands on 12 August 2025 as "
                "the acquisition vehicle for Project Harbour. Its portfolio interests, if any, "
                "beyond the terminal acquisition are a matter for its own board and are not "
                "addressed in this letter.",
            ),
            ("space", ""),
            ("sub", "5. CONFLICTS"),
            (
                "body",
                "We have checked our current client list and identified no conflict in acting "
                "for Vantage Industrial Partners on this engagement.",
            ),
            ("space", ""),
            ("body", "Dorian Aske, Partner, for and on behalf of Halberd Quinn LLP"),
        ],
    ],
    "SOL-2026-0207-solmark-ventures-registry-extract.pdf": [
        [
            ("head", "COMMERCIAL REGISTER EXTRACT"),
            ("body", "Chamber of Commerce, Rotterdam — Trade Register"),
            ("space", ""),
            ("body", "Entity: Solmark Ventures BV"),
            ("body", "Registration number: NL-88214407"),
            ("body", "Extract date: 11 May 2026"),
            (
                "body",
                "Obtained for: Halberd Quinn LLP, Matter SOL-2026-0207 (due diligence on adverse "
                "party structure)",
            ),
            ("space", ""),
            ("sub", "1. REGISTERED SHAREHOLDERS OF SOLMARK VENTURES BV"),
            ("body", "Vantage Industrial Partners — 40 per cent, registered 12 August 2025."),
            ("body", "Thessaly Bulk Investments Ltd — 35 per cent, registered 12 August 2025."),
            ("body", "Corfield Family Trust — 25 per cent, registered 12 August 2025."),
            ("space", ""),
            ("sub", "2. HOLDINGS OF SOLMARK VENTURES BV"),
            (
                "body",
                "Solmark Ventures BV holds 65 per cent of the issued share capital of Kestrel "
                "Fabrication Group, acquired 3 October 2025.",
            ),
            (
                "body",
                "Solmark Ventures BV also holds 12 per cent of Bellwood Structural Ltd, an "
                "unrelated fabrication business acquired as a minority financial investment on "
                "20 October 2025.",
            ),
        ],
        [
            ("sub", "Extract — continued"),
            ("sub", "3. NOTE ON SIMILAR NAMES"),
            (
                "body",
                "Kestrel Fabrication Group (company number NL-77310982) should not be confused "
                "with Kestrel Fabrication Systems Ltd (company number GB-09341207), a separate "
                "and unrelated business registered in England. Solmark Ventures BV holds no "
                "interest in Kestrel Fabrication Systems Ltd.",
            ),
            ("space", ""),
            ("sub", "4. FILING HISTORY"),
            (
                "body",
                "Annual return filed 30 April 2026. No change of shareholder recorded since "
                "incorporation. Registered office: Weena 210, 3012 NM Rotterdam.",
            ),
            ("space", ""),
            ("body", "Extract certified by the Registrar."),
        ],
    ],
    "SOL-2026-0207-authority-note.pdf": [
        [
            ("head", "HALBERD QUINN LLP: KNOW-HOW NOTE"),
            ("sub", "WORKING-DAY NOTICE PERIODS: RECENT AUTHORITY"),
            ("space", ""),
            ("body", "Circulated to: Construction and Engineering Group"),
            ("body", "Author: Priya Sandhu, Professional Support Lawyer"),
            ("body", "Date: 22 May 2026"),
            ("space", ""),
            ("sub", "1. THE NEW DECISION"),
            (
                "body",
                "Renwick v Astor Grain [2025] UKSC 9 overrules Harrow Timber Co v Bellamy "
                "Frères [2018] EWCA Civ 1140 on the construction of working-day notice periods. "
                "The Supreme Court held that the day of delivery is included in the count where "
                "the contract does not say otherwise, reversing the Court of Appeal's approach "
                "in Harrow Timber.",
            ),
            ("space", ""),
            ("sub", "2. PRACTICAL EFFECT"),
            (
                "body",
                "Any advice that excluded the day of delivery when calculating a working-day "
                "notice period under Harrow Timber should be revisited. On the Supreme Court's "
                "approach a notice period that was one day late on the old authority may in "
                "fact be in time, and vice versa.",
            ),
        ],
        [
            ("sub", "Note — continued"),
            ("sub", "3. A SEPARATE POINT — NOT AFFECTED"),
            (
                "body",
                "Practitioners should note that Beaumont Freight Ltd v Cassera Shipping [2021] "
                "EWCA Civ 640, on the separate question of notice by electronic means, is "
                "unaffected by Renwick and remains good law.",
            ),
            ("space", ""),
            ("sub", "4. ACTION"),
            (
                "body",
                "Fee earners should check any live matter that relies on Harrow Timber Co v "
                "Bellamy Frères for a notice-period calculation and consider whether the advice "
                "needs to be revisited in light of Renwick v Astor Grain.",
            ),
            ("space", ""),
            ("body", "Priya Sandhu, Professional Support Lawyer"),
        ],
    ],
}

# ─────────────────────────────────────────────────────────────────────────────────────────────
# HEALTHCARE — contraindication_via_ingredient across two encounters, months apart, plus a
# same-chart CONTRAINDICATED_WITH pair that the chart records but does not itself assert as an
# interaction.
#
# Encounter ENC-2026-0512: Ms Odalys Ferreira is allergic to Zenapril (absolute) and separately
# intolerant of Zentanyl (an unrelated opioid, decoy). Encounter ENC-2026-0567, seven weeks
# later under a different clinician in a different clinic, prescribes Cardaze -- a different
# brand the outpatient chart records the patient did not flag an allergy against, because the
# clinic has no automatic access to inpatient allergy records. Only the formulary note states
# Cardaze and Zenapril share ramipril. `contraindication_via_ingredient` needs all three: the
# prescription, the allergy, and the ingredient link -- each from a different document, two
# from different encounters.
# ─────────────────────────────────────────────────────────────────────────────────────────────

HEALTHCARE: dict[str, Document] = {
    "ENC-2026-0512-admission-note.pdf": [
        [
            ("head", "FENWICK VALE GENERAL HOSPITAL"),
            ("body", "Department of Internal Medicine"),
            ("space", ""),
            ("sub", "ADMISSION NOTE"),
            ("body", "Encounter: ENC-2026-0512"),
            ("body", "Patient: Ms Odalys Ferreira, DOB 9 November 1958, hospital number 52-90144"),
            ("body", "Admitted: 14 April 2026, 06:20"),
            ("body", "Clinician: Dr Marcus Villanueva, Consultant Physician"),
            ("space", ""),
            ("sub", "1. PRESENTING COMPLAINT"),
            (
                "body",
                "Ms Odalys Ferreira presented with severe headache, blurred vision and a blood "
                "pressure of 224/126 on arrival by ambulance. She was diagnosed with a "
                "hypertensive crisis and admitted for urgent blood pressure control.",
            ),
            ("space", ""),
            ("sub", "2. ALLERGY HISTORY"),
            (
                "body",
                "Ms Odalys Ferreira is allergic to Zenapril. The reaction recorded in 2015 was "
                "facial and laryngeal angioedema requiring an overnight admission, and this "
                "allergy is to be treated as absolute.",
            ),
            (
                "body",
                "The patient also reports a past intolerance to Zentanyl, described as nausea "
                "rather than an allergic reaction. This is recorded as an intolerance, not an "
                "allergy, and does not carry the same prescribing restriction.",
            ),
            ("body", "No other drug allergy is reported."),
        ],
        [
            ("sub", "Encounter ENC-2026-0512 — continued"),
            ("sub", "3. RELEVANT HISTORY"),
            (
                "body",
                "Hypertension, diagnosed 2009, previously managed on a calcium channel blocker "
                "withdrawn in 2015 following the angioedema episode. Osteoarthritis of both "
                "knees. Subclinical hypothyroidism, monitored but not treated.",
            ),
            ("space", ""),
            ("sub", "4. MEDICATIONS ON ADMISSION"),
            (
                "body",
                "Amlodipine 10mg once daily, own supply. Levothyroxine was trialled in 2021 and "
                "discontinued after six months with no clear benefit. Paracetamol as required "
                "for joint pain.",
            ),
        ],
        [
            ("sub", "Encounter ENC-2026-0512 — continued"),
            ("sub", "5. EXAMINATION"),
            (
                "body",
                "Pulse 92 regular, respiratory rate 18, oxygen saturation 97 per cent on air. "
                "Fundoscopy shows no papilloedema. Neurological examination is unremarkable.",
            ),
            ("space", ""),
            ("sub", "6. PLAN"),
            (
                "body",
                "Intravenous labetalol to control blood pressure, with a view to establishing an "
                "oral regimen before discharge. Follow-up to be arranged with the cardiology "
                "outpatient clinic once stable.",
            ),
            ("space", ""),
            ("body", "Dr Marcus Villanueva, Consultant Physician"),
        ],
    ],
    "ENC-2026-0512-discharge-summary.pdf": [
        [
            ("head", "FENWICK VALE GENERAL HOSPITAL"),
            ("body", "Department of Internal Medicine"),
            ("space", ""),
            ("sub", "DISCHARGE SUMMARY"),
            ("body", "Encounter: ENC-2026-0512"),
            ("body", "Patient: Ms Odalys Ferreira, hospital number 52-90144"),
            ("body", "Discharged: 18 April 2026"),
            ("body", "Clinician: Dr Marcus Villanueva, Consultant Physician"),
            ("space", ""),
            ("sub", "1. ADMISSION DIAGNOSIS"),
            (
                "body",
                "Hypertensive crisis, resolved with intravenous labetalol followed by oral "
                "therapy. Blood pressure on discharge was 138/84.",
            ),
            ("space", ""),
            ("sub", "2. DRUG ALLERGIES"),
            (
                "body",
                "A drug allergy is recorded on the admission note for this encounter. "
                "Prescribers reviewing this patient should refer to the admission note rather "
                "than relying on this summary alone, as the allergy is not restated here in "
                "full.",
            ),
        ],
        [
            ("sub", "Encounter ENC-2026-0512 — continued"),
            ("sub", "3. DISCHARGE MEDICATION"),
            (
                "body",
                "Amlodipine 10mg once daily, continued. Doxazosin 4mg once daily, newly "
                "started.",
            ),
            ("space", ""),
            ("sub", "4. FOLLOW-UP"),
            (
                "body",
                "Referred to the cardiology outpatient clinic under Dr Priyasha Kohli for "
                "ongoing blood pressure management and review of long-term therapy, given the "
                "limited options following the 2015 withdrawal of calcium channel blocker "
                "therapy.",
            ),
            ("space", ""),
            ("body", "Dr Marcus Villanueva, Consultant Physician"),
        ],
    ],
    "ENC-2026-0567-prescription-chart.pdf": [
        [
            ("head", "FENWICK VALE GENERAL HOSPITAL"),
            ("body", "Cardiology Outpatient Clinic — Prescription Chart"),
            ("space", ""),
            ("body", "Encounter: ENC-2026-0567"),
            ("body", "Patient: Ms Odalys Ferreira, hospital number 52-90144"),
            ("body", "Date: 3 June 2026"),
            ("body", "Clinician: Dr Priyasha Kohli, Consultant Cardiologist"),
            ("space", ""),
            ("sub", "1. REGULAR MEDICATION"),
            (
                "body",
                "Dr Priyasha Kohli prescribed Cardaze 5mg once daily, in place of amlodipine "
                "and doxazosin, to simplify the patient's antihypertensive regimen following "
                "review.",
            ),
            (
                "body",
                "Dr Priyasha Kohli also prescribed Utrenix 200mg twice daily and Fenlorapine "
                "50mg once daily for newly diagnosed paroxysmal atrial fibrillation identified "
                "on ambulatory monitoring.",
            ),
        ],
        [
            ("sub", "Encounter ENC-2026-0567 — continued"),
            ("sub", "2. AS REQUIRED"),
            (
                "body",
                "Paracetamol 1g up to four times daily for joint pain, continued from the "
                "patient's own supply.",
            ),
            ("space", ""),
            ("sub", "3. NOTE"),
            (
                "body",
                "This clinic does not have automatic access to inpatient allergy records from "
                "other departments. Prescribers are asked to confirm allergy status directly "
                "with the patient at each visit. Ms Odalys Ferreira did not raise a drug allergy "
                "at this visit.",
            ),
            ("space", ""),
            ("body", "Dr Priyasha Kohli, Consultant Cardiologist"),
        ],
    ],
    "FORM-2026-cardaze-ramipril-equivalence-note.pdf": [
        [
            ("head", "FENWICK VALE GENERAL HOSPITAL"),
            ("body", "Medicines Information Service"),
            ("space", ""),
            ("sub", "FORMULARY NOTE: ACE-INHIBITOR BRAND EQUIVALENCE AND INTERACTION ALERT"),
            ("body", "Reference: FORM-2026-071"),
            ("body", "Issued: 20 May 2026"),
            ("body", "Author: Mr Idris Whitcombe, Lead Formulary Pharmacist"),
            ("space", ""),
            ("sub", "1. PURPOSE"),
            (
                "body",
                "This note addresses two unrelated prescribing risks raised with Medicines "
                "Information this quarter: brand-name allergy recording for ACE inhibitors, and "
                "a combined QT-prolongation risk between two antiarrhythmic agents.",
            ),
            ("space", ""),
            ("sub", "2. ACE-INHIBITOR EQUIVALENCE"),
            (
                "body",
                "Cardaze has the same active ingredient as Zenapril. Both are ramipril "
                "preparations and differ only in presentation and manufacturer. An allergy "
                "recorded against one brand must be treated as applying to the other.",
            ),
            (
                "body",
                "Zentanyl, despite the similar name, is an opioid analgesic and shares no "
                "active ingredient with either Cardaze or Zenapril. An intolerance to Zentanyl "
                "carries no implication for ramipril-containing products.",
            ),
        ],
        [
            ("sub", "Formulary note — continued"),
            ("sub", "3. ANTIARRHYTHMIC INTERACTION"),
            (
                "body",
                "Utrenix and Fenlorapine are contraindicated in combination. Both prolong the "
                "QT interval, and concurrent use has been associated with torsades de pointes "
                "in post-marketing reports. Prescribers should select one agent, not both.",
            ),
            ("space", ""),
            ("sub", "4. FOR REFERENCE"),
            (
                "body",
                "Broxamol and Xelnorate also share an active ingredient, lisinopril, unrelated "
                "to either issue above. This is included for completeness following an "
                "unrelated query and does not concern this patient's medication.",
            ),
            ("space", ""),
            ("body", "Mr Idris Whitcombe, Lead Formulary Pharmacist"),
        ],
    ],
}

# ─────────────────────────────────────────────────────────────────────────────────────────────
# FINTECH — group_exposure_via_control walked to the full three-link bound, plus
# related_party_lending, plus a memo that states the wrong conclusion in its own text.
#
# Facility FAC-2026-0512 lends to Solenne Freight Logistics BV, guaranteed by Ridgeway Capital
# Partners NV. The control chain is three links: Halcyon Maritime Holdings SA controls
# Ridgeway Capital Partners NV (71%), which controls Corvus Shipping Group Ltd (58%), which
# controls Solenne Freight Logistics BV (83%) -- exactly at the ontology's *1..3 bound, and
# split so that no two links are asserted in the same document as each other or as LENDS_TO.
# The bank already carries EUR 210m of exposure to Halcyon under a separate facility, which is
# what makes the group exposure material. `related_party_lending` fires separately: the
# guarantor is also a disclosed related party of the borrower.
#
# The decoy: Ridgeway Capital Partners LLC, an unrelated Delaware entity, planted in the old
# Halcyon facility's own covenant certificate -- the one place a similarity search is most
# likely to retrieve while looking for "Ridgeway".
# ─────────────────────────────────────────────────────────────────────────────────────────────

FINTECH: dict[str, Document] = {
    "FAC-2026-0512-credit-agreement.pdf": [
        [
            ("head", "GREYFRIARS MERCHANT BANK PLC"),
            ("body", "Corporate & Institutional Banking, 22 Lombard Court, London EC3V 9AA"),
            ("space", ""),
            ("sub", "CREDIT AGREEMENT SUMMARY"),
            ("body", "Facility reference: FAC-2026-0512"),
            ("body", "Date: 2 March 2026"),
            ("space", ""),
            ("sub", "1. THE FACILITY"),
            (
                "body",
                "Facility FAC-2026-0512 lends to Solenne Freight Logistics BV. The facility is "
                "a revolving credit facility of EUR 150,000,000 with a final maturity of 2 "
                "March 2031.",
            ),
            (
                "body",
                "The facility is governed by the law of England and Wales. Drawings are "
                "permitted in euro and sterling.",
            ),
            ("space", ""),
            ("sub", "2. GUARANTEE"),
            (
                "body",
                "Ridgeway Capital Partners NV guarantees facility FAC-2026-0512. The guarantee "
                "is unlimited in amount and continues until all obligations under the facility "
                "are discharged.",
            ),
        ],
        [
            ("sub", "Facility FAC-2026-0512 — continued"),
            ("sub", "3. COVENANTS"),
            (
                "body",
                "Facility FAC-2026-0512 is subject to a net leverage covenant of 3.25x, tested "
                "quarterly. It is also subject to an information covenant requiring audited "
                "accounts within 120 days of each financial year end.",
            ),
            ("space", ""),
            ("sub", "4. LIMITS"),
            (
                "body",
                "This facility is assessed against the large exposures regime. Aggregate "
                "exposure to a group of connected clients may not exceed 25 per cent of Tier 1 "
                "capital.",
            ),
            ("space", ""),
            ("sub", "5. BORROWER"),
            (
                "body",
                "Solenne Freight Logistics BV is a road and rail freight operator "
                "headquartered in Rotterdam. This agreement records the credit terms only; the "
                "borrower's ownership structure is addressed separately by Credit Risk.",
            ),
            ("space", ""),
            ("body", "For and on behalf of Greyfriars Merchant Bank plc"),
        ],
    ],
    "FAC-2026-0512-credit-committee-memo.pdf": [
        [
            ("head", "GREYFRIARS MERCHANT BANK PLC"),
            ("body", "Credit Risk and Compliance"),
            ("space", ""),
            ("sub", "CREDIT COMMITTEE MEMORANDUM"),
            ("body", "Facility: FAC-2026-0512"),
            ("body", "Date: 24 February 2026"),
            ("body", "Prepared by: Ms Naledi Okoro, Head of Credit Risk"),
            ("space", ""),
            ("sub", "1. PROPOSED EXPOSURE"),
            (
                "body",
                "Solenne Freight Logistics BV has requested a EUR 150,000,000 revolving "
                "facility to fund working capital growth. The borrower is a road and rail "
                "freight operator with contracts across the Benelux region.",
            ),
            ("space", ""),
            ("sub", "2. OWNERSHIP"),
            (
                "body",
                "Corvus Shipping Group Ltd controls Solenne Freight Logistics BV, holding 83 "
                "per cent of its issued share capital.",
            ),
            (
                "body",
                "Ridgeway Capital Partners NV controls Corvus Shipping Group Ltd, holding 58 "
                "per cent of its issued share capital and appointing a majority of its board.",
            ),
        ],
        [
            ("sub", "Memorandum — continued"),
            ("sub", "3. EXISTING BOOK"),
            (
                "body",
                "The bank already has EUR 210,000,000 of drawn exposure to Halcyon Maritime "
                "Holdings SA under facility FAC-2023-0144. That facility is performing and was "
                "last reviewed in January 2026.",
            ),
            (
                "body",
                "Halcyon Maritime Holdings SA is not otherwise connected to this proposal for "
                "the purposes of this memorandum, which considers the Solenne facility on its "
                "own terms.",
            ),
        ],
        [
            ("sub", "Memorandum — continued"),
            ("sub", "4. SECTOR CONTEXT"),
            (
                "body",
                "European road and rail freight volumes grew 4 per cent in 2025 despite "
                "continued fuel cost volatility. Solenne's contract mix is weighted toward "
                "retail distribution, which the committee views as lower cyclical risk than "
                "industrial freight.",
            ),
            ("space", ""),
            ("sub", "5. PRICING"),
            (
                "body",
                "Margin of 210 basis points over EURIBOR, arrangement fee of 60 basis points, "
                "commitment fee of 35 basis points on the undrawn amount.",
            ),
            ("space", ""),
            ("sub", "6. RECOMMENDATION"),
            (
                "body",
                "Approve, subject to the group limit being recalculated on a connected-client "
                "basis before first drawdown if further ownership links are identified above "
                "Corvus Shipping Group Ltd. The individual limits are each within policy; the "
                "group position has not been aggregated.",
            ),
            ("space", ""),
            ("body", "Ms Naledi Okoro, Head of Credit Risk"),
        ],
    ],
    "FAC-2026-0512-ridgeway-registry-extract.pdf": [
        [
            ("head", "CORPORATE REGISTRY EXTRACT"),
            ("body", "Dutch Chamber of Commerce — Trade Register"),
            ("space", ""),
            ("body", "Entity: Ridgeway Capital Partners NV"),
            ("body", "Registration number: NL-71209855"),
            ("body", "Extract date: 10 March 2026"),
            ("body", "Obtained for: Greyfriars Merchant Bank plc, KYC file — Facility FAC-2026-0512"),
            ("space", ""),
            ("sub", "1. REGISTERED SHAREHOLDERS OF RIDGEWAY CAPITAL PARTNERS NV"),
            ("body", "Halcyon Maritime Holdings SA — 71 per cent, registered 4 June 2019."),
            ("body", "Osprey Family Office BV — 22 per cent, registered 4 June 2019."),
            ("body", "Individual minority holders — 7 per cent, various dates."),
        ],
        [
            ("sub", "Extract — continued"),
            ("sub", "2. NOTE ON SIMILAR NAMES"),
            (
                "body",
                "Ridgeway Capital Partners NV (Netherlands, registration NL-71209855) should "
                "not be confused with Ridgeway Capital Partners LLC (Delaware, USA), an "
                "unrelated entity with no common ownership, directorship or shared registered "
                "office.",
            ),
            ("space", ""),
            ("sub", "3. FILING HISTORY"),
            (
                "body",
                "Annual return filed 15 January 2026. No change of shareholder recorded since "
                "2019. Registered office: Herengracht 458, 1017 CA Amsterdam.",
            ),
            ("space", ""),
            ("body", "Extract certified by the Registrar."),
        ],
    ],
    "FAC-2026-0512-related-party-disclosure.pdf": [
        [
            ("head", "GREYFRIARS MERCHANT BANK PLC"),
            ("body", "Credit Risk and Compliance"),
            ("space", ""),
            ("sub", "RELATED-PARTY DISCLOSURE REGISTER"),
            ("body", "Facility: FAC-2026-0512"),
            ("body", "Date: 26 February 2026"),
            ("body", "Prepared by: Mr Callan Osei, Compliance Officer"),
            ("space", ""),
            ("sub", "1. DISCLOSED RELATIONSHIP"),
            (
                "body",
                "Ridgeway Capital Partners NV is a related party of Solenne Freight Logistics "
                "BV. Two individuals, Mr Wouter de Groot and Ms Ilse Bakker, sit on the boards "
                "of both entities, and Ms Bakker is the sister of Solenne's chief executive.",
            ),
            ("space", ""),
            ("sub", "2. BASIS FOR DISCLOSURE"),
            (
                "body",
                "The relationship is disclosed under the bank's related-party lending policy, "
                "which requires recording any connection by shared directorship or family "
                "relationship between a guarantor and a borrower on the same facility.",
            ),
        ],
        [
            ("sub", "Register — continued"),
            ("sub", "3. OTHER DISCLOSURES THIS QUARTER"),
            (
                "body",
                "For completeness, the register also records that Fennimore Bulk Traders Ltd "
                "is a related party of Larchwood Components Ltd through common directorship, a "
                "matter unconnected to facility FAC-2026-0512.",
            ),
            ("space", ""),
            ("sub", "4. STATUS"),
            (
                "body",
                "This disclosure is informational. It does not by itself indicate that the "
                "facility breaches any lending limit; that assessment is for Credit Risk.",
            ),
            ("space", ""),
            ("body", "Mr Callan Osei, Compliance Officer"),
        ],
    ],
    "FAC-2023-0144-covenant-certificate.pdf": [
        [
            ("head", "HALCYON MARITIME HOLDINGS SA"),
            ("body", "Group Treasury, 12 Boulevard Royal, Luxembourg"),
            ("space", ""),
            ("sub", "COVENANT COMPLIANCE CERTIFICATE"),
            ("body", "Facility: FAC-2023-0144"),
            ("body", "Testing date: 31 December 2025"),
            ("body", "Delivered: 29 January 2026"),
            ("space", ""),
            ("sub", "1. CERTIFICATION"),
            (
                "body",
                "Facility FAC-2023-0144 lends to Halcyon Maritime Holdings SA. This certificate "
                "is delivered under the information covenant of that facility.",
            ),
            ("space", ""),
            ("sub", "2. LEVERAGE TEST"),
            (
                "body",
                "Net leverage at the testing date was 2.9x against a covenanted maximum of "
                "3.5x. Halcyon Maritime Holdings SA is in compliance with the net leverage "
                "covenant at 31 December 2025.",
            ),
        ],
        [
            ("sub", "Certificate — continued"),
            ("sub", "3. GROUP NOTE"),
            (
                "body",
                "Halcyon Maritime Holdings SA confirms no change in its principal holdings "
                "during the period. For the avoidance of doubt, Halcyon's US joint-venture "
                "partner, Ridgeway Capital Partners LLC (Delaware), is not party to this "
                "facility, holds no security interest under it and has no guarantee obligations "
                "in respect of it.",
            ),
            ("space", ""),
            ("sub", "4. INFORMATION COVENANT"),
            (
                "body",
                "Audited accounts for the year ended 31 December 2025 are due on 30 April 2026. "
                "The information covenant is due on 30 April 2026.",
            ),
            ("space", ""),
            ("body", "For and on behalf of Halcyon Maritime Holdings SA, Group Treasury"),
        ],
    ],
}

# ─────────────────────────────────────────────────────────────────────────────────────────────
# RETAIL — the AnyCorp dataset from `generate_retail_demo_pdfs.py`, taken to full depth.
#
# `exception_on_superseded_policy` walked two links instead of one: the Electronics
# opened-returns provision was amended in 2025 before Policy Bulletin 2026-03 withdrew it
# outright, and the desk's approval cites the original 2024 wording -- one step further back
# than either supersession record names on its own. `SUPERSEDES` chains "the 2026 bulletin
# supersedes the 2025 amendment, which supersedes the 2024 clause" across three documents, none
# of which mentions the approval, and the approval mentions none of them by date.
#
# `related_party_resale` walked three links instead of two: Sam Parker controls Northgate
# Holdings, which was restructured under an intermediate vehicle, Aldergate Ventures, which in
# turn controls PixelPerfect Resale -- exactly at the ontology's *1..3 bound. Each link is
# asserted in a different document than the others and than the seller-approval fact.
#
# The decoy: PixelPerfect Resale (the real, controlled storefront) vs PixelPerfect Returns
# Direct (an unrelated AnyCorp-operated clearance channel, disambiguated only in the seller
# directory extract most likely to be retrieved by a search for "PixelPerfect").
#
# The wrong conclusion: a case status note, filed after the goodwill approval, states outright
# that "no exception has been granted to Sam Parker while case LP-2026-0088 has been open" --
# true of every case file the analyst checked, and wrong about the one approval that a
# different desk, in a different system, granted nine days earlier.
# ─────────────────────────────────────────────────────────────────────────────────────────────

RETAIL: dict[str, Document] = {
    "AC-POL-2024-11-electronics-provision.pdf": [
        [
            ("head", "ANYCORP RETAIL"),
            ("body", "Return Policy Manual 2024 -- Section 2, Return Windows by Category"),
            ("space", ""),
            ("sub", "PROVISION 2.4 -- ELECTRONICS, OPENED"),
            ("body", "Reference: AC-POL-2024-11"),
            ("body", "Issued: 3 June 2024"),
            ("body", "Effective: 1 July 2024"),
            ("body", "Issued by: Priya Kandasamy, Director of Returns Policy"),
            ("space", ""),
            ("sub", "1. THE PROVISION"),
            (
                "body",
                "An opened electronics item may be returned within 30 days of the purchase date "
                "subject to a 10% restocking fee. This is Provision 2.4 of the Return Policy "
                "Manual 2024, the Electronics opened-returns provision.",
            ),
            ("space", ""),
            ("sub", "2. RATIONALE"),
            (
                "body",
                "The 30-day window aligns Electronics with the general merchandise window "
                "elsewhere in Section 2. The 10% fee reflects the depreciation on an opened "
                "electronics item observed in the 2023 markdown data.",
            ),
        ],
        [
            ("sub", "Provision 2.4 -- continued"),
            ("sub", "3. SCOPE"),
            (
                "body",
                "This provision applies to all electronics categories carried by AnyCorp "
                "Retail, in store, online and through the contact centre. Unopened electronics "
                "are governed separately by Provision 2.1 and are not affected here.",
            ),
            ("space", ""),
            ("body", "Priya Kandasamy, Director of Returns Policy, AnyCorp Retail"),
        ],
    ],
    "AC-POL-2025-07-electronics-amendment.pdf": [
        [
            ("head", "ANYCORP RETAIL"),
            ("body", "Returns and Customer Care -- Policy Amendment"),
            ("space", ""),
            ("sub", "AMENDMENT AC-POL-2025-07"),
            ("body", "Reference: AC-POL-2025-07"),
            ("body", "Issued: 2 September 2025"),
            ("body", "Effective: 16 September 2025"),
            ("body", "Issued by: Delia Marchetti, Director of Returns Policy"),
            ("space", ""),
            ("sub", "1. PURPOSE"),
            (
                "body",
                "Amendment AC-POL-2025-07 tightens the Electronics opened-returns provision "
                "following the 2025 mid-year review of return rates by category.",
            ),
            ("space", ""),
            ("sub", "2. PROVISION SUPERSEDED"),
            (
                "body",
                "Amendment AC-POL-2025-07 supersedes Provision 2.4 of the Return Policy Manual "
                "2024, the Electronics opened-returns provision, issued 3 June 2024. That "
                "provision allowed a 30-day window at a 10% restocking fee. It is withdrawn in "
                "full and replaced with the provision below.",
            ),
            ("space", ""),
            ("sub", "3. PROVISION SUBSTITUTED"),
            (
                "body",
                "An opened electronics item may be returned within 14 days of the purchase date "
                "subject to a 15% restocking fee. This becomes the new Provision 2.4 of the "
                "Return Policy Manual 2025.",
            ),
        ],
        [
            ("sub", "Amendment AC-POL-2025-07 -- continued"),
            ("sub", "4. EFFECT ON OPEN RETURNS"),
            (
                "body",
                "A return lodged before 16 September 2025 is assessed under the 2024 wording. A "
                "return lodged on or after that date is assessed under the 14-day, 15% wording "
                "above.",
            ),
            ("space", ""),
            ("body", "Delia Marchetti, Director of Returns Policy, AnyCorp Retail"),
        ],
    ],
    "LP-2026-0088-return-approval.pdf": [
        [
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
                "Return RTN-2026-00912 is approved in favour of Sam Parker. The full purchase "
                "price of 299.99 USD is refunded to the original payment method. The 10% "
                "restocking fee is waived in full as a goodwill gesture.",
            ),
            ("space", ""),
            ("sub", "2. AUTHORITY RELIED ON"),
            (
                "body",
                "This approval is made under Provision 2.4 of the Return Policy Manual 2024, "
                "the Electronics opened-returns provision: a 30-day window from the purchase "
                "date, subject to a 10% restocking fee. Return RTN-2026-00912 was lodged 14 "
                "days after purchase and so falls inside that window.",
            ),
            (
                "body",
                "The 10% restocking fee of 30.00 USD is waived under the Refund Adjustments "
                "note at Section 4, which permits a fee to be waived where a supervisor records "
                "a reason. The reason recorded is customer goodwill.",
            ),
        ],
        [
            ("sub", "Approval RTN-2026-00912 -- continued"),
            ("sub", "3. CUSTOMER STANDING"),
            (
                "body",
                "Sam Parker is described on the account as a long-standing customer at Silver "
                "loyalty tier with a lifetime value of 12,450 USD. The desk treated that "
                "history as the reason for goodwill.",
            ),
            ("space", ""),
            ("sub", "4. CHECKS PERFORMED"),
            (
                "body",
                "Receipt verified. Serial number matched the receipt. Packaging intact. No "
                "manager escalation was raised and no loss prevention check was requested.",
            ),
            ("space", ""),
            ("body", "Curtis Lindgren, Returns Desk Supervisor, Store 118, Omaha NE"),
        ],
    ],
    "LP-2026-0088-investigation-memo.pdf": [
        [
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
                "Case LP-2026-0088 investigates Sam Parker for suspected return abuse. The "
                "ground is the red flag at Section 6 of the Return Policy Manual 2025, Red "
                "Flags for Fraudulent Returns, item 2: \"Pattern of returning high-value "
                "electronics\".",
            ),
            (
                "body",
                "Sam Parker has nineteen recorded purchases and eight returns, a return rate of "
                "42.10%. The account carries a fraud risk score of 85 and its status is under "
                "review. Three suspicious-activity notes are on file.",
            ),
            ("space", ""),
            ("sub", "2. PATTERN"),
            (
                "body",
                "Every one of the eight returns was an electronics item in opened condition. "
                "Seven of the eight were lodged on day 13 or day 14 of the 14-day opened "
                "window. The items were a Samsung 65\" QLED TV on three separate occasions, an "
                "Apple iPhone 15 Pro twice, a Sony PlayStation 5, a MacBook Pro 14\" and a set "
                "of Bose Headphones.",
            ),
        ],
        [
            ("sub", "Case LP-2026-0088 -- continued"),
            ("sub", "3. WHAT THIS MEMORANDUM DOES NOT SAY"),
            (
                "body",
                "No conclusion is drawn. Sam Parker has not been notified and no privilege is "
                "suspended. Case LP-2026-0088 remains open pending review of the resale enquiry "
                "running separately.",
            ),
            ("space", ""),
            ("body", "Ada Okonjo, Loss Prevention Analyst"),
        ],
    ],
    "LP-2026-0088-case-status-note.pdf": [
        [
            ("head", "ANYCORP RETAIL -- LOSS PREVENTION"),
            ("body", "Confidential. Internal distribution only."),
            ("space", ""),
            ("sub", "CASE STATUS NOTE"),
            ("body", "Case: LP-2026-0088"),
            ("body", "Note date: 25 March 2026"),
            ("body", "Prepared by: Ada Okonjo, Loss Prevention Analyst"),
            ("space", ""),
            ("sub", "1. CURRENT STATUS"),
            (
                "body",
                "Case LP-2026-0088 remains open. Ownership records requested from Marketplace "
                "Onboarding on 12 March 2026 have not yet been received.",
            ),
            ("space", ""),
            ("sub", "2. EXCEPTIONS DURING THE OPEN PERIOD"),
            (
                "body",
                "This analyst has reviewed the returns desk exception log for Sam Parker "
                "covering 6 March 2026 to 25 March 2026. No exception has been granted to Sam "
                "Parker while case LP-2026-0088 has been open. The account has had no return "
                "activity since the case opened.",
            ),
        ],
        [
            ("sub", "Case LP-2026-0088 -- continued"),
            ("sub", "3. NOTE ON SCOPE"),
            (
                "body",
                "The returns desk exception log reviewed for this note covers Store 118's own "
                "system. Goodwill approvals recorded directly on an order by a store "
                "supervisor, rather than routed through the exception log, are outside what "
                "this analyst checked.",
            ),
            ("space", ""),
            ("body", "Ada Okonjo, Loss Prevention Analyst"),
        ],
    ],
    "MEM-2026-0231-merchant-onboarding-pixelperfect.pdf": [
        [
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
                "PixelPerfect Resale is approved to trade on AnyCorp Marketplace from 18 "
                "February 2026. PixelPerfect Resale sells refurbished consumer electronics. The "
                "listings submitted at application were a Samsung 65\" QLED TV, an Apple "
                "iPhone 15 Pro and a Sony PlayStation 5, all described as open-box.",
            ),
            ("space", ""),
            ("sub", "2. OWNERSHIP"),
            (
                "body",
                "PixelPerfect Resale is wholly owned by Aldergate Ventures. Aldergate Ventures "
                "controls PixelPerfect Resale and holds no other subsidiary. Aldergate Ventures "
                "is itself a non-trading company; ownership above Aldergate Ventures is a "
                "matter for the group registry and is not addressed further in this "
                "memorandum.",
            ),
        ],
        [
            ("sub", "Approval MEM-2026-0231 -- continued"),
            ("sub", "3. REGISTERED ADDRESS"),
            (
                "body",
                "PixelPerfect Resale's registered address is 4410 Distribution Way, Council "
                "Bluffs, IA 51501. Aldergate Ventures' registered address was not supplied at "
                "application and is not on file with Marketplace Onboarding.",
            ),
            ("space", ""),
            ("sub", "4. CHECKS PERFORMED"),
            (
                "body",
                "Company registration verified. Bank account verified in the name of "
                "PixelPerfect Resale. Tax identifier supplied. Onboarding does not screen a "
                "declared beneficial owner against the customer file, and no such check was "
                "performed here.",
            ),
            ("space", ""),
            ("body", "Naomi Ferreira, Marketplace Onboarding, AnyCorp Marketplace"),
        ],
    ],
    "AC-REG-2026-04-seller-directory-extract.pdf": [
        [
            ("head", "ANYCORP MARKETPLACE"),
            ("body", "Seller Directory -- Corporate Registry Extract"),
            ("space", ""),
            ("body", "Reference: AC-REG-2026-04"),
            ("body", "Extract date: 20 March 2026"),
            ("body", "Obtained for: Loss Prevention, Case LP-2026-0088 (ownership enquiry)"),
            ("body", "Prepared by: Marketplace Legal and Registrations"),
            ("space", ""),
            ("sub", "1. ALDERGATE VENTURES"),
            (
                "body",
                "Aldergate Ventures is a non-trading holding company incorporated on 4 January "
                "2026. Its own trading interests, if any, are not addressed in this extract, "
                "which was obtained to establish who holds Aldergate Ventures rather than what "
                "Aldergate Ventures holds.",
            ),
            (
                "body",
                "Aldergate Ventures is wholly owned by Northgate Holdings, which holds the "
                "entire issued share capital and appoints Aldergate Ventures' sole director. "
                "Northgate Holdings holds no other subsidiary directly.",
            ),
        ],
        [
            ("sub", "Extract AC-REG-2026-04 -- continued"),
            ("sub", "2. NOTE ON SIMILAR NAMES"),
            (
                "body",
                "PixelPerfect Resale, a registered AnyCorp Marketplace seller, should not be "
                "confused with PixelPerfect Returns Direct, an AnyCorp-operated clearance "
                "channel for customer returns that is not a third-party seller and does not "
                "appear on this extract or in any seller approval on file.",
            ),
            ("space", ""),
            ("sub", "3. FILING HISTORY"),
            (
                "body",
                "Northgate Holdings acquired Aldergate Ventures on 4 January 2026, the date of "
                "Aldergate Ventures' incorporation. No change of ownership recorded since.",
            ),
            ("space", ""),
            ("body", "Extract certified by Marketplace Legal and Registrations."),
        ],
    ],
    "LP-2026-0088-northgate-ownership-note.pdf": [
        [
            ("head", "ANYCORP RETAIL -- LOSS PREVENTION"),
            ("body", "Confidential. Internal distribution only."),
            ("space", ""),
            ("sub", "OWNERSHIP FOLLOW-UP NOTE"),
            ("body", "Case: LP-2026-0088"),
            ("body", "Note date: 2 April 2026"),
            ("body", "Prepared by: Ada Okonjo, Loss Prevention Analyst"),
            ("space", ""),
            ("sub", "1. NORTHGATE HOLDINGS"),
            (
                "body",
                "Northgate Holdings is a non-trading company whose sole director is Sam "
                "Parker. Sam Parker controls Northgate Holdings. No further beneficial owner "
                "is declared on Northgate Holdings' own filing.",
            ),
            (
                "body",
                "Northgate Holdings' registered address is 980 Maple St, Omaha, NE 68101, "
                "which matches the address on Sam Parker's customer account.",
            ),
        ],
        [
            ("sub", "Ownership follow-up -- continued"),
            ("sub", "2. STATUS"),
            (
                "body",
                "This note responds to a request from the resale enquiry referenced in the "
                "case opening memorandum, which asked whether Sam Parker holds an interest in "
                "any non-trading company. Northgate Holdings has no dealings with AnyCorp "
                "Marketplace of its own. No conclusion is drawn here; the enquiry remains "
                "open.",
            ),
            ("space", ""),
            ("body", "Ada Okonjo, Loss Prevention Analyst"),
        ],
    ],
}

PACKS: dict[str, dict[str, Document]] = {
    "legal": LEGAL,
    "healthcare": HEALTHCARE,
    "fintech": FINTECH,
    "retail": RETAIL,
}


def build() -> list[tuple[Path, Path]]:
    """Write the loose PDFs and the zip for each pack. Both, for the reason the retail
    generator gives: the zip is what a person uploads through the Documents page in one go,
    and the loose directory is what a diff shows when one document's wording changes.
    """
    written = []
    for pack, documents in PACKS.items():
        loose = HERE / f"{pack}-multihop-demo"
        loose.mkdir(exist_ok=True)

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, pages in documents.items():
                doc = pymupdf.open()
                for page_lines in pages:
                    _page(doc, page_lines)
                payload = doc.tobytes()
                doc.close()
                archive.writestr(name, payload)
                (loose / name).write_bytes(payload)

        out = HERE / f"{pack}-multihop-demo.zip"
        out.write_bytes(buffer.getvalue())
        written.append((out, loose))
    return written


if __name__ == "__main__":
    for zip_path, loose_dir in build():
        pack = zip_path.stem.replace("-multihop-demo", "")
        docs = PACKS[pack]
        pages = sum(len(p) for p in docs.values())
        print(f"wrote {zip_path} ({zip_path.stat().st_size:,} bytes, {len(docs)} documents, {pages} pages)")
        for name in docs:
            print(f"  {loose_dir / name}")
