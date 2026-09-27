import json
from rapidfuzz import fuzz
from backend.config import DOCUMENTS_PATH, FORMS_DIR
from backend.office_directory import parent_office_name, build_office_lookup, build_canonical_names, find_office_doc_id, _fuzzy_score

MANIFEST_PATH = FORMS_DIR / "manifest.json"
OFFICE_FORM_MATCH_THRESHOLD = 80

_forms_by_doc_id = None
_manifest = None
_attached_filenames = None
_office_alias_lookup = None
_office_canonical_names = None


def _load():
    global _forms_by_doc_id, _attached_filenames
    if _forms_by_doc_id is None:
        with open(DOCUMENTS_PATH, "r", encoding="utf-8") as f:
            documents = json.load(f)
        _forms_by_doc_id = {
            doc["id"]: doc.get("forms", [])
            for doc in documents
        }
        _attached_filenames = {
            form["filename"]
            for forms in _forms_by_doc_id.values()
            for form in forms
        }
    return _forms_by_doc_id


def _load_office_resolvers():
    global _office_alias_lookup, _office_canonical_names
    if _office_alias_lookup is None:
        with open(DOCUMENTS_PATH, "r", encoding="utf-8") as f:
            documents = json.load(f)
        _office_alias_lookup = build_office_lookup(documents)
        _office_canonical_names = build_canonical_names(documents)
    return _office_alias_lookup, _office_canonical_names


def _load_manifest():
    global _manifest
    if _manifest is None:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            _manifest = json.load(f)
    return _manifest


def _to_form_dict(name: str, filename: str) -> dict:
    return {
        "name": name,
        "filename": filename,
        "download_url": f"/api/forms/{filename}",
    }


def get_forms_for_doc_ids(doc_ids: list[str]) -> list[dict]:
    lookup = _load()
    seen = set()
    forms = []
    for doc_id in doc_ids:
        for form in lookup.get(doc_id, []):
            key = form["filename"]
            if key in seen:
                continue
            seen.add(key)
            forms.append(_to_form_dict(form["name"], form["filename"]))
    return forms


def get_forms_for_office(office_field: str | None) -> list[dict]:
    if not office_field:
        return []
    manifest = _load_manifest()
    manifest_offices = {entry["office"].lower(): entry["office"] for entry in manifest}

    parent = parent_office_name(office_field).lower()
    candidates = [parent]

    # using aliases for abbreviations of office names
    alias_lookup, canonical_names = _load_office_resolvers()
    doc_id = find_office_doc_id(office_field, alias_lookup)
    if doc_id and doc_id in canonical_names:
        candidates.append(canonical_names[doc_id].lower())

    matched_office = None
    for cand in candidates:
        if cand in manifest_offices:
            matched_office = manifest_offices[cand]
            break

    if matched_office is None:
        best_score, best_office = 0, None
        for cand in candidates:
            for office_lower, office_orig in manifest_offices.items():
                score = _fuzzy_score(cand, office_lower)
                if score > best_score:
                    best_score, best_office = score, office_orig
        if best_score >= OFFICE_FORM_MATCH_THRESHOLD:
            matched_office = best_office

    if matched_office is None:
        return []

    return [
        _to_form_dict(entry["name"], entry["filename"])
        for entry in manifest
        if entry["office"] == matched_office
    ]


def get_forms_for_offices(office_fields: list[str]) -> list[dict]:
    seen = set()
    forms = []
    for office_field in office_fields:
        for form in get_forms_for_office(office_field):
            if form["filename"] in seen:
                continue
            seen.add(form["filename"])
            forms.append(form)
    return forms


def get_other_forms() -> list[dict]:
    _load()  # ensures _attached_filenames is populated
    manifest = _load_manifest()
    others = [
        {**_to_form_dict(entry["name"], entry["filename"]), "office": entry["office"]}
        for entry in manifest
        if entry["filename"] not in _attached_filenames
    ]
    others.sort(key=lambda f: (f["office"], f["name"]))
    return others


def get_all_forms() -> list[dict]:
    # every known downloadable form, regardless of whether it's attached to a doc
    manifest = _load_manifest()
    return [
        {**_to_form_dict(entry["name"], entry["filename"]), "office": entry["office"]}
        for entry in manifest
    ]


def reload():
    global _forms_by_doc_id, _manifest, _attached_filenames
    _forms_by_doc_id = None
    _manifest = None
    _attached_filenames = None