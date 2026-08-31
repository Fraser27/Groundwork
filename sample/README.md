# Sample data

Demo document packs for the four ontology packs (`legal`, `healthcare`, `fintech`,
`retail`), plus a harder tier per domain built specifically to fail under plain vector RAG.
Load and use them exactly as described in the main [README](../README.md#trying-it-with-the-demo-documents):
download the zip, unzip it, and upload the PDFs through **Documents, then Upload**, or have
an administrator load the zip server-side with **Admin, then Load sample data**.

This file is about the harder tier. For the base packs and the walkthrough of uploading them,
see the main README.

## The multi-hop packs

| Pack | Zip | Documents | Pages | Rule(s) exercised |
|---|---|---|---|---|
| Legal | [`legal-multihop-demo.zip`](legal-multihop-demo.zip) | 5 | 11 | `conflict_via_affiliate` (`AFFILIATE_OF*1..3`, 2 links), `authority_stale` |
| Healthcare | [`healthcare-multihop-demo.zip`](healthcare-multihop-demo.zip) | 4 | 9 | `contraindication_via_ingredient`, across two encounters |
| Fintech | [`fintech-multihop-demo.zip`](fintech-multihop-demo.zip) | 5 | 11 | `group_exposure_via_control` (`CONTROLS*1..3`, 3 links, at the bound), `related_party_lending` |
| Retail | [`retail-multihop-demo.zip`](retail-multihop-demo.zip) | 8 | 16 | `exception_on_superseded_policy` (`SUPERSEDES`, 2-link revision chain), `exception_during_investigation`, `related_party_resale` (`CONTROLS*1..3`, 3 links, at the bound) |

Regenerate after editing the content:

```bash
.venv/bin/python sample/generate_multihop_demo_pdfs.py
```

## Why this tier exists

The base packs (`legal-demo.zip`, `healthcare-demo.zip`, `fintech-demo.zip`) each show one
rule with the minimum number of one-page documents that can carry it. That is enough to prove
a rule *fires*, but it understates the case for a graph: a good enough embedding search can
sometimes get lucky across two documents.

This tier removes the luck. Every rule is walked to the full depth its ontology allows
(`*1..3`), the premises are spread across four or five documents instead of two or three, and
each pack plants a decoy entity — a name lexically close enough to the real one to collide in
a vector index, disambiguated only in a registry extract or formulary note that a similarity
search has no particular reason to prefer over the more prominent, wrong-entity sentence.
Fintech goes one step further: the credit committee memo states the wrong conclusion in its
own words. A retrieval system that trusts its best-matching passage will repeat that sentence.
A system that traverses `CONTROLS*1..3` will not.

## Questions to ask against each pack

Ask these on the **Ask** page once the pack's documents are uploaded and the tenant is on the
matching ontology. Each is unanswerable from any single document or any single embedding-search
hit — the point is to ask them, get a wrong or empty answer from a plain-RAG mental model, then
see the graph answer trace back through its premises.

### Legal (`legal-multihop-demo.zip`)

- **"Does acting for Vantage Industrial Partners on matter VIP-2025-0388 create a conflict
  with matter SOL-2026-0207?"**
  The engagement letter for `SOL-2026-0207` never mentions Vantage. The engagement letter for
  `VIP-2025-0388` never mentions Kestrel Fabrication Group. The two-link affiliate chain —
  Vantage holds 40% of Solmark Ventures BV, Solmark Ventures BV holds 65% of Kestrel
  Fabrication Group — is asserted only in `SOL-2026-0207-solmark-ventures-registry-extract.pdf`,
  a third document neither engagement letter cites. A plain-RAG search for "conflict" or
  "Vantage" will not surface that extract, because it does not use either word in a way a
  similarity match rewards; it surfaces instead the engagement letters' own "no conflict has
  been identified" and "identified no conflict" sentences, both true in isolation and both
  wrong once the chain is followed. `conflict_via_affiliate` walks `AFFILIATE_OF*1..3` and
  concludes `(SOL-2026-0207)-[:POTENTIAL_CONFLICT]->(Vantage Industrial Partners)`.
- **"Is Kestrel Fabrication Systems Ltd connected to any of our matters?"**
  It is not — that is the decoy. `Kestrel Fabrication Group` (the real adverse party, held by
  Solmark Ventures BV) and `Kestrel Fabrication Systems Ltd` (an unrelated English company) are
  distinguished by company number only in the registry extract. A text-similarity match over
  the two names will not tell them apart; the graph will, because the extract's disambiguation
  is itself an assertion.
- **"Which open matters rely on authority that has since been overruled?"**
  `Renwick v Astor Grain [2025] UKSC 9` overrules `Harrow Timber Co v Bellamy Frères [2018]`,
  recorded in `SOL-2026-0207-authority-note.pdf`. The advice on prospects for `SOL-2026-0207`
  cites Harrow Timber and was written before the authority note existed. `authority_stale`
  connects the two.

### Healthcare (`healthcare-multihop-demo.zip`)

- **"Is any prescription for Ms Odalys Ferreira sharing an active ingredient with a recorded
  allergy?"**
  The allergy (Zenapril) is recorded on encounter `ENC-2026-0512`, an inpatient admission in
  April. The prescription (Cardaze) is written seven weeks later on encounter `ENC-2026-0567`,
  an outpatient clinic visit whose own chart says explicitly that it has no access to inpatient
  allergy records and that the patient did not raise an allergy at that visit. Only
  `FORM-2026-cardaze-ramipril-equivalence-note.pdf`, a fourth document tied to neither
  encounter, states that Cardaze and Zenapril share ramipril. A plain-RAG search for
  "Cardaze allergy" returns nothing, because no document contains both words near each other —
  which is the point. `contraindication_via_ingredient` needs the prescription, the allergy,
  and the ingredient link from three separate documents to fire.
- **"Does the patient's intolerance to Zentanyl affect the Cardaze prescription?"**
  It should not, and does not — that is the decoy. Zentanyl is an opioid with a name close
  enough to Zenapril to collide in a similarity search, but the formulary note states plainly
  that it shares no ingredient with either Cardaze or Zenapril. Answering this correctly
  requires reading the disambiguating sentence, not just retrieving the passage that mentions
  Zentanyl.
- **"Are any two of this patient's current medications contraindicated with each other?"**
  Utrenix and Fenlorapine, both prescribed on the outpatient chart, are declared
  contraindicated in the same formulary note — a same-visit pair the prescription chart itself
  never flags as an interaction.

### Fintech (`fintech-multihop-demo.zip`)

- **"Which facilities reach Halcyon Maritime Holdings SA through a controlled group?"**
  The credit agreement for `FAC-2026-0512` lends to Solenne Freight Logistics BV and says
  nothing about ownership above it. The credit committee memo records two of the three control
  links (Corvus Shipping Group Ltd controls Solenne; Ridgeway Capital Partners NV controls
  Corvus) and then states in its own text that "Halcyon Maritime Holdings SA is not otherwise
  connected to this proposal." The third link — Halcyon controls Ridgeway, 71% — is only in
  `FAC-2026-0512-ridgeway-registry-extract.pdf`. A system that trusts the memo's own sentence
  gets this wrong; `group_exposure_via_control` walks `CONTROLS*1..3` across all three links
  and concludes `(FAC-2026-0512)-[:GROUP_EXPOSURE]->(Halcyon Maritime Holdings SA)`, which
  matters because the bank already carries EUR 210,000,000 of exposure to Halcyon directly
  under `FAC-2023-0144`.
- **"Is Ridgeway Capital Partners LLC connected to facility FAC-2026-0512?"**
  It is not — that is the decoy. `Ridgeway Capital Partners NV` (Netherlands, the actual
  guarantor) and `Ridgeway Capital Partners LLC` (Delaware, an unrelated joint-venture partner
  of Halcyon's) are distinguished only in the registry extract and in Halcyon's own covenant
  certificate, planted deliberately in the one document a search for "Ridgeway" is most likely
  to retrieve.
- **"Are we lending to a party related to one of facility FAC-2026-0512's own obligors?"**
  The guarantee (Ridgeway guarantees FAC-2026-0512) is in the credit agreement. The relatedness
  (Ridgeway is a related party of Solenne, the borrower, through shared directors and a family
  relationship) is only in `FAC-2026-0512-related-party-disclosure.pdf`, which itself says the
  disclosure "does not by itself indicate that the facility breaches any lending limit" — true,
  and beside the point the rule exists to surface. `related_party_lending` combines
  `LENDS_TO` + `GUARANTEES` + `RELATED_PARTY_OF` from three documents to conclude
  `(FAC-2026-0512)-[:UNDISCLOSED_RELATED_LENDING]->(Solenne Freight Logistics BV)`.

### Retail (`retail-multihop-demo.zip`)

- **"Does the return approval for Sam Parker rely on a policy provision that has since been
  superseded?"**
  The approval cites Provision 2.4 of the Return Policy Manual 2024 — a 30-day window, 10% fee
  — issued in `AC-POL-2024-11-electronics-provision.pdf`. Neither the approval nor that
  provision mentions what happened next: `AC-POL-2025-07-electronics-amendment.pdf` supersedes
  the 2024 wording with a stricter 14-day, 15% version, over a year before the approval is
  written. A plain-RAG search for "return approval" or "Provision 2.4" surfaces the approval
  and the 2024 provision, which agree with each other and look routine together — the
  amendment shares no obvious keyword overlap with either. `exception_on_superseded_policy`
  walks `SUPERSEDES` and finds the withdrawal a keyword match never would.
- **"Has a policy exception been granted to Sam Parker while case LP-2026-0088 has been
  open?"**
  Yes, and the one document that directly answers "have any exceptions been granted" says
  the opposite. `LP-2026-0088-case-status-note.pdf`, filed 25 March 2026, states plainly that
  "no exception has been granted to Sam Parker while case LP-2026-0088 has been open" — true
  of the exception log the analyst checked, and silent about the goodwill approval a *different*
  desk recorded directly on the order nine days after the case opened, in
  `LP-2026-0088-return-approval.pdf`. A system that trusts the most on-topic-looking sentence
  repeats the status note's denial. `exception_during_investigation` joins the case-opening
  memo and the approval letter directly and does not need the status note at all — which is
  exactly why the status note is worth asking about separately.
- **"Does Sam Parker control a marketplace seller AnyCorp pays out to?"**
  Yes, three links away. `MEM-2026-0231-merchant-onboarding-pixelperfect.pdf` shows only
  PixelPerfect Resale owned by Aldergate Ventures, and stops there. The next link — Aldergate
  Ventures owned by Northgate Holdings — is only in `AC-REG-2026-04-seller-directory-extract.pdf`.
  The last link — Sam Parker controls Northgate Holdings — is only in
  `LP-2026-0088-northgate-ownership-note.pdf`, a document that itself draws no conclusion.
  `related_party_resale` walks `CONTROLS*1..3` across all three documents and concludes
  `(LP-2026-0088)-[:RELATED_PARTY_RESALE]->(PixelPerfect Resale)`.
- **"Is PixelPerfect Returns Direct connected to Sam Parker's case?"**
  It is not — that is the decoy. `PixelPerfect Returns Direct` is AnyCorp's own clearance
  channel, lexically close enough to `PixelPerfect Resale` to collide in a similarity search,
  and disambiguated only in the seller directory extract, the one document a search for
  "PixelPerfect" is most likely to retrieve.

## What a plain-RAG answer looks like on each

If you want to demonstrate the failure mode directly rather than just assert it: ask the same
questions with the tenant's retrieval tier forced to vector-only (`VECTOR_FIRST_TIER`, see
`src/governance.py`), or simply read out the single most relevant passage a search for the
question's keywords returns. In every case above, that passage is either silent (healthcare's
allergy/prescription pair share no words) or actively wrong (fintech's memo, both legal
engagement letters' own "no conflict" lines) — never merely incomplete in a way a bigger top-k
would fix. That is what makes these documents a demonstration of the graph rather than of
retrieval tuning.
