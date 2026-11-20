import re
import difflib
from pathlib import Path
from spellchecker import SpellChecker
from backend.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    STRUCTURED_DOC_MAX_CHARS,
    UNSTRUCTURED_DOC_WHOLE_CHUNK_MAX_CHARS,
)

_spell = SpellChecker(language="en")


def build_domain_vocabulary(chunks: list[dict]) -> set[str]:
    # chunks = the metadata records loaded from data/chunks_meta.json
    vocab = set()
    for meta in chunks:
        for field in (
            meta.get("title") or "",
            meta.get("category") or "",
            meta.get("title_filipino") or "",
        ):
            vocab.update(w.lower() for w in field.split() if len(w) > 3)
    return vocab


TAGALOG_WORDLIST_PATH: Path | None = None

TAGALOG_PROTECTED_WORDS = {
    "kamusta", "kumusta", "magandang", "umaga", "tanghali", "hapon", "gabi",
    "paano", "ano", "saan", "san", "kailan", "sino", "bakit", "pwede",
    "puwede", "bilang", "botante", "negosyo", "opo", "ito", "yan", "yun",
    "mag", "nang", "ang", "mga", "hindi", "oo", "salamat", "po", "paalam",
    "ngayon", "bukas", "kahapon", "dito", "doon", "ganito", "ganyan",
    "magparehistro",
}

_general_vocab: set[str] | None = None


def build_general_vocabulary() -> set[str]:
    global _general_vocab
    if _general_vocab is not None:
        return _general_vocab

    vocab = set(TAGALOG_PROTECTED_WORDS)
    if TAGALOG_WORDLIST_PATH and TAGALOG_WORDLIST_PATH.is_file():
        with open(TAGALOG_WORDLIST_PATH, "r", encoding="utf-8") as f:
            vocab.update(line.strip().lower() for line in f if line.strip())

    _general_vocab = vocab
    return _general_vocab


_TYPO_CORRECTION_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "and", "or", "but", "if", "so",
    "do", "does", "did", "can", "could", "will", "would", "should",
    "i", "you", "he", "she", "it", "we", "they", "me", "my", "your",
    "what", "when", "where", "who", "how", "why", "which",
    "this", "that", "these", "those", "with", "from", "by", "as",
    "not", "no", "yes", "please", "give", "get", "list", "all", "under",
}

_SHORT_WORD_OVERRIDES = {
    "yu": "you", "u": "you", "ur": "your", "thx": "thanks",
    "dis": "this", "dat": "that", "d": "the", "n": "and",
}

_DOMAIN_TERM_OVERRIDES = {
    "bisness": "business",
}

_CASUAL_GREETING_RE = re.compile(
    r"^(hi+|he+l+o+|he+y+|yo+|su+p+|ok(ay)?|thanks?)$", re.IGNORECASE
)


def correct_typos(text: str, domain_vocabulary: set[str]) -> str:
    tagalog_vocabulary = build_general_vocabulary()
    words = text.split()
    corrected = []
    for w in words:
        stripped = w.strip(".,?!:;\"'")
        lw = stripped.lower()

        if lw in _SHORT_WORD_OVERRIDES:
            corrected.append(_SHORT_WORD_OVERRIDES[lw])
            continue

        if _CASUAL_GREETING_RE.match(lw):
            corrected.append(w)
            continue

        if len(lw) <= 2 or lw in _TYPO_CORRECTION_STOPWORDS:
            corrected.append(w)
            continue

        if lw in domain_vocabulary or lw in tagalog_vocabulary:
            corrected.append(w)
            continue

        if lw in _DOMAIN_TERM_OVERRIDES:
            corrected.append(_DOMAIN_TERM_OVERRIDES[lw])
            continue

        domain_match = difflib.get_close_matches(lw, domain_vocabulary, n=1, cutoff=0.85)
        if domain_match:
            corrected.append(domain_match[0])
            continue

        if _spell.known([lw]):
            corrected.append(w)
            continue

        correction = _spell.correction(lw)
        corrected.append(correction if correction else w)
    return " ".join(corrected)


def clean_text(text) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


_GREETING_PATTERNS = [
    r"(hi+|hello+|he+y+|yo+|su+p+)( there| po| guys| everyone)?",
    r"good (morning|afternoon|evening)( po)?",
    r"(kumusta|kamusta)( po| ka)?",
    r"magandang (umaga|tanghali|hapon|gabi)( po)?",
    r"(hi+|hello+)(,)? how are you",
    r"thank(s| you)( po| so much)?",
    r"(ok|okay|alright)( thanks| po)?",
]
_GREETING_RE = re.compile(
    r"^(" + "|".join(_GREETING_PATTERNS) + r")[\s!.,?]*$", re.IGNORECASE
)


def is_greeting(text: str) -> bool:
    return bool(_GREETING_RE.fullmatch(clean_text(text)))


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = clean_text(text)
    if len(text) <= chunk_size:
        return [text]

    def find_boundary(s: str, target: int) -> int:
        window_start = max(0, target - 60)
        window = s[window_start:target]
        for sep in ("; ", ". ", ", ", " "):
            pos = window.rfind(sep)
            if pos != -1:
                return window_start + pos + len(sep)
        return target

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        tentative_end = start + chunk_size
        if tentative_end >= n:
            end = n
        else:
            end = find_boundary(text, tentative_end)
            if end <= start:
                end = tentative_end
        chunk = text[start:end]
        chunks.append(chunk.strip())
        if end >= n:
            break
        next_start = end - overlap
        start = next_start if next_start > start else end
    return [c for c in chunks if c]


# hard fix for semantic mismatch/cross-lingual mismatch
# these docs are written in Filipino, so the multilingual model fails when user query is English
TERM_ALIASES: dict[str, list[str]] = {
    "doc_tax_001": ["amilyar"],
    "doc_vet_010": ["magpabakuna", "bakuna sa aso", "pet vaccination"],
    "doc_bus_009": ["isara ang negosyo", "cancel business permit", "close down business"],
    "doc_general_001": ["san ang city hall", "oras ng city hall", "city hall address", "city hall hours"],
    "doc_reg_039": ["request a bus for an event", "borrow a coaster", "vehicle loan", "city vehicle for event", "humiram ng sasakyan"],
}


def flatten_structured_document_core(doc: dict) -> str:
    lines = [doc["title"]]
    aliases = TERM_ALIASES.get(doc.get("id"))
    if aliases:
        lines.append("Also asked as: " + ", ".join(aliases))
    if doc.get("title_filipino"):
        lines.append(f"({doc['title_filipino']})")
    if doc.get("office"):
        lines.append(f"Office or Division: {doc['office']}")
    if doc.get("classification"):
        lines.append(f"Classification: {doc['classification']}")
    if doc.get("transaction_type"):
        lines.append(f"Type of Transaction: {doc['transaction_type']}")
    if doc.get("who_may_avail"):
        lines.append(f"Who may avail: {doc['who_may_avail']}")

    if doc.get("requirements"):
        lines.append("Checklist of Requirements:")
        for req in doc["requirements"]:
            lines.append(f"- {req['requirement']} (secure from: {req['where_to_secure']})")

    return "\n".join(lines)


def flatten_structured_document_summary(doc: dict) -> str | None:
    if not (doc.get("total_fee") or doc.get("total_processing_time")):
        return None
    return (
        f"{doc['title']}\n"
        f"Total fee: {doc.get('total_fee', 'N/A')}\n"
        f"Total processing time: {doc.get('total_processing_time', 'N/A')}"
    )


def flatten_structured_document_steps(doc: dict) -> str | None:
    if not doc.get("process_flow"):
        return None

    lines = [doc["title"], "Process Flow (step-by-step):"]
    for i, step in enumerate(doc["process_flow"], start=1):
        parts = []
        if step.get("client_step"):
            parts.append(f"Client step: {step['client_step']}")
        if step.get("agency_action"):
            parts.append(f"Agency action: {step['agency_action']}")
        if step.get("fee"):
            parts.append(f"Fee: {step['fee']}")
        if step.get("processing_time"):
            parts.append(f"Processing time: {step['processing_time']}")
        if step.get("person_responsible"):
            parts.append(f"Person responsible: {step['person_responsible']}")
        lines.append(f"{i}. " + "; ".join(parts))

    return "\n".join(lines)


def flatten_structured_document(doc: dict) -> str:
    parts = [flatten_structured_document_core(doc)]
    summary = flatten_structured_document_summary(doc)
    if summary:
        parts.append(summary)
    steps = flatten_structured_document_steps(doc)
    if steps:
        parts.append(steps)
    return "\n".join(parts)


def flatten_office_contact(doc: dict) -> str:
    d = doc.get("office_directory_details", {})
    lines = [doc["title"]]
    if d.get("department_head"):
        head_line = f"Department Head: {d['department_head']}"
        if d.get("designation") and d["designation"] != "Department Head":
            head_line += f" ({d['designation']})"
        lines.append(head_line)
    if d.get("location"):
        lines.append(f"Location: {d['location']}")
    for num in d.get("office_numbers", []):
        lines.append(f"Phone: {num}")
    for label, ext in (d.get("extensions") or {}).items():
        lines.append(f"  - {label.replace('_', ' ').title()}: {ext}")
    if d.get("email_address"):
        lines.append(f"Email: {d['email_address']}")
    if d.get("facebook_page"):
        lines.append(f"Facebook: {d['facebook_page']}")
    if d.get("functional_statement"):
        lines.append(f"About this office: {d['functional_statement']}")
    return "\n".join(lines)


def preprocess_document(doc: dict) -> list[dict]:
    is_office_directory = "office_directory_details" in doc
    is_structured = "process_flow" in doc or "requirements" in doc
    office = doc.get("office")

    if is_office_directory:
        return [{
            "chunk_id": f"{doc['id']}_chunk0",
            "doc_id": doc["id"],
            "category": doc["category"],
            "title": doc["title"],
            "office": None,
            "text": clean_text(flatten_office_contact(doc)),
        }]

    if is_structured:
        out = []

        def _add_chunks(text: str, suffix: str) -> None:
            text = clean_text(text)
            pieces = [text] if len(text) <= STRUCTURED_DOC_MAX_CHARS else chunk_text(text)
            for i, piece in enumerate(pieces):
                out.append({
                    "chunk_id": f"{doc['id']}{suffix}{i}",
                    "doc_id": doc["id"],
                    "category": doc["category"],
                    "title": doc["title"],
                    "office": office,
                    "text": piece,
                })

        _add_chunks(flatten_structured_document_core(doc), "_chunk")

        summary_text = flatten_structured_document_summary(doc)
        if summary_text:
            out.append({
                "chunk_id": f"{doc['id']}_summary_chunk0",
                "doc_id": doc["id"],
                "category": doc["category"],
                "title": doc["title"],
                "office": office,
                "text": clean_text(summary_text),
            })

        steps_text = flatten_structured_document_steps(doc)
        if steps_text:
            _add_chunks(steps_text, "_steps_chunk")

        return out

    # use to simulate the title of the document; there are some documents that have title in the category that are not the title in the document
    # but the title (category) is mentioned inside the document
    full_text = f"{doc['title']}\n{doc['text']}"
    if len(clean_text(full_text)) <= UNSTRUCTURED_DOC_WHOLE_CHUNK_MAX_CHARS:
        chunks = [clean_text(full_text)]
    else:
        chunks = chunk_text(full_text)

    out = []
    for i, chunk in enumerate(chunks):
        out.append({
            "chunk_id": f"{doc['id']}_chunk{i}",
            "doc_id": doc["id"],
            "category": doc["category"],
            "title": doc["title"],
            "office": office,
            "text": chunk,
        })
    return out


def basic_tokenize(text: str) -> list[str]:
    text = clean_text(text).lower()
    return re.findall(r"[a-zA-Z']+", text)
