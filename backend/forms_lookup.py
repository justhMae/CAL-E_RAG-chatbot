import json
from backend.config import DOCUMENTS_PATH, FORMS_DIR

MANIFEST_PATH = FORMS_DIR / "manifest.json"

_forms_by_doc_id = None
_manifest = None
_attached_filenames = None


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
    """De-duplicated list of forms for these doc_ids, each with a download_url."""
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


def get_other_forms() -> list[dict]:
    """
    Forms that exist in data/forms/ but aren't attached to any specific
    document -- e.g. because no Citizen's Charter entry matched them closely
    enough. Each entry also carries "office" so the caller can group them
    in the reply text.
    """
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
    """Every known downloadable form, regardless of whether it's attached to a doc."""
    manifest = _load_manifest()
    return [
        {**_to_form_dict(entry["name"], entry["filename"]), "office": entry["office"]}
        for entry in manifest
    ]


def reload():
    """Call after editing documents.json/manifest.json in the same process (e.g. tests)."""
    global _forms_by_doc_id, _manifest, _attached_filenames
    _forms_by_doc_id = None
    _manifest = None
    _attached_filenames = None
