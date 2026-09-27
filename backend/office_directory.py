# matches a service document's "office" field against the office_directory contact documents in documents.json
# so rag_pipeline.py can deterministically attach the correct department's real phone/email/location whenever an answer names an office to contact

import re
from rapidfuzz import fuzz

FUZZY_MATCH_THRESHOLD = 80  # 0-100; token_set_ratio score to accept, after stopword stripping

# generic words common across LGU office names that inflate a naive fuzzy match score between unrelated offices just because they share this generic phrasing
_GENERIC_OFFICE_WORDS = {
    "city", "office", "services", "service", "department", "division",
    "and", "the", "of", "management", "development", "administrative",
}


def strip_generic_words(name: str) -> str:
    words = re.findall(r"[a-z]+", name.lower())
    kept = [w for w in words if w not in _GENERIC_OFFICE_WORDS]
    # if stripping leaves nothing, fall back to the original words rather than matching on an empty string
    return " ".join(kept) if kept else " ".join(words)


def _fuzzy_score(a: str, b: str) -> float:
    # strips generic office vocabulary first, then compares remaining unique words regardless of order (token_set_ratio)
    return fuzz.token_set_ratio(strip_generic_words(a), strip_generic_words(b))


def parent_office_name(office_field: str) -> str:
    return re.split(r"\s+[-–]\s+", office_field, maxsplit=1)[0].strip()


def build_office_lookup(documents: list[dict]) -> dict:
    lookup = {}
    for doc in documents:
        details = doc.get("office_directory_details")
        if not details:
            continue
        doc_id = doc["id"]
        # match against the document's own title (minus the " - Contact Information" suffix, if present)
        name = re.sub(r"\s*-\s*Contact Information\s*$", "", doc["title"], flags=re.IGNORECASE)
        lookup[name.strip().lower()] = doc_id
        for alias in details.get("aliases", []):
            lookup[alias.strip().lower()] = doc_id
    return lookup


def build_canonical_names(documents: list[dict]) -> dict:
    names = {}
    for doc in documents:
        if "office_directory_details" not in doc:
            continue
        name = re.sub(r"\s*-\s*Contact Information\s*$", "", doc["title"], flags=re.IGNORECASE)
        name = re.sub(r"\s*\([^)]*\)\s*$", "", name)  # strip trailing "(BPTFO)"
        names[doc["id"]] = name.strip()
    return names


def find_office_doc_id(office_field: str | None, lookup: dict) -> str | None:
    if not office_field:
        return None

    parent = parent_office_name(office_field).lower()

    if parent in lookup:
        return lookup[parent]

    # fuzzy fallback for close-but-not-exact full names (ex. divisions phrased slightly differently, or minor spelling variance)
    best_score = 0
    best_doc_id = None
    for name, doc_id in lookup.items():
        score = _fuzzy_score(parent, name)
        if score > best_score:
            best_score = score
            best_doc_id = doc_id

    if best_score >= FUZZY_MATCH_THRESHOLD:
        return best_doc_id
    return None