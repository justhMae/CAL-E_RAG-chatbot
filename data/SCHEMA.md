# Document Schema — Citizen's Charter Format

Each service in `data/documents.json` should follow this structure
(this matches the standard Citizen's Charter layout used on the
Calamba City website).

```json
{
  "id": "doc_tax_003",
  "category": "TAX",
  "title": "Issuance of Barangay Jurisdiction",
  "title_filipino": "Pag-iisyu ng sertipikasyon sa nasasakupan ng Barangay",
  "office": "City Assessment Office - Tax Mapping Operations Division",
  "classification": "Simple",
  "transaction_type": "G2C – Government to Citizen, G2B – Government to Business Entity, G2G – Government to Government",
  "who_may_avail": "All Property Owners / Administrators in Calamba City",
  "requirements": [
    { "requirement": "Kopya ng Titulo", "where_to_secure": "Registry of Deeds" },
    { "requirement": "Authorization Letter with Valid ID's", "where_to_secure": "Property Owner" }
  ],
  "process_flow": [
    {
      "client_step": "1. Punan ang request form at isumite ang mga kaukulang dokumento",
      "agency_action": "1.1 Tanggapin at suriin ang mga dokumento at bigyan ng Order of Payment",
      "fee": "Walang Babayaran",
      "processing_time": "5 minuto",
      "person_responsible": "Tax Mapping Aide Draftsman I, Draftsman II"
    },
    {
      "client_step": "",
      "agency_action": "1.2 Iproseso ang dokumento",
      "fee": "",
      "processing_time": "10 minuto",
      "person_responsible": "Taxmapper I, Taxmapper III, Tax Mapper IV"
    },
    {
      "client_step": "2. Magbayad sa Tanggapan ng Ingat Yaman",
      "agency_action": "2. Bigyan ng Opisyal na Resibo",
      "fee": "Php 50.00",
      "processing_time": "5 minuto",
      "person_responsible": "Revenue Collection Clerk III, Treasury Office"
    },
    {
      "client_step": "3. Tanggapin ang dokumento",
      "agency_action": "3. Ibigay ang mga Sertipika dong kopya ng Dokumento",
      "fee": "Walang Babayaran",
      "processing_time": "5 minuto",
      "person_responsible": "Taxmapping personnel"
    }
  ],
  "total_fee": "Php 50.00",
  "total_processing_time": "25 minuto",
  "forms": [
    { "name": "Barangay Jurisdiction Request Form", "filename": "barangay_jurisdiction_request_form.pdf" }
  ]
}
```

## Field notes

- **id**: unique, prefix by category (e.g. `doc_tax_003`).
- **category**: must be one of the 12 keys in `backend/config.py` → `CATEGORIES`.
- **title / title_filipino**: English and Filipino names of the service.
- **office**: the office/division that handles it.
- **classification**: "Simple" / "Complex" / "Highly Technical" per the charter.
- **transaction_type**: G2C / G2B / G2G as listed.
- **who_may_avail**: who can request this service.
- **requirements**: array of `{requirement, where_to_secure}` — one entry per row.
- **process_flow**: array of steps. When a client step spans multiple agency
  actions (like step 1 above having 1.1 and 1.2), repeat the row but leave
  `client_step` and `fee` empty on the continuation rows — that mirrors the
  merged cells in the original table and keeps totals correct.
- **total_fee / total_processing_time**: the TOTAL row at the bottom.
- **forms** *(optional)*: downloadable files for this service. Each entry is
  `{name, filename}`. `filename` must match a real file placed directly in
  `data/forms/` (flat — no subfolders; `/api/forms/<filename>` serves from
  there). When this document is retrieved for a user's question, the chat
  response includes these as download links.

## The forms manifest and "other" forms

`data/forms/manifest.json` is the source of truth for **every** file that
exists in `data/forms/`, whether or not it's attached to a document:
`[{filename, name, office}, ...]`. When adding a new PDF to `data/forms/`,
add a matching entry to `manifest.json` too.

A form doesn't have to be attached to a specific document. If nothing in
`documents.json` matches it closely enough, leave it out of every `forms`
array, `backend/forms_lookup.get_other_forms()` treats any manifest entry
that isn't referenced by some document's `forms` list as an "other"
downloadable file. `rag_pipeline.py` detects general questions like "what
other forms can I download?" and answers with that list (grouped by
office), so orphaned forms are still reachable even without a matching doc.

## Backward compatibility

Older, simpler entries (just `{id, category, title, text}`) still work —
`backend/preprocess.py` checks for `process_flow` and falls back to plain
`text` if it isn't present. You can mix both formats in the same file while
you migrate everything over.
