import re
import difflib
from pathlib import Path
from spellchecker import SpellChecker
from backend.config import CHUNK_SIZE, CHUNK_OVERLAP, STRUCTURED_DOC_MAX_CHARS

_spell = SpellChecker(language="en")


def build_domain_vocabulary(collection) -> set[str]:
    all_data = collection.get()
    vocab = set()
    for meta in all_data.get("metadatas", []):
        for field in (
            meta.get("title", ""),
            meta.get("category", ""),
            meta.get("title_filipino", ""),
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

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        if end >= len(text):
            break
        start = end - overlap
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


def flatten_structured_document(doc: dict) -> str:
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

    if doc.get("process_flow"):
        lines.append("Process Flow:")
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

    if doc.get("total_fee") or doc.get("total_processing_time"):
        lines.append(
            f"Total fee: {doc.get('total_fee', 'N/A')}; "
            f"Total processing time: {doc.get('total_processing_time', 'N/A')}"
        )

    return "\n".join(lines)


def preprocess_document(doc: dict) -> list[dict]:
    is_structured = "process_flow" in doc or "requirements" in doc
    full_text = flatten_structured_document(doc) if is_structured else doc["text"]

    if is_structured:
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
            "text": chunk,
        })
    return out


def basic_tokenize(text: str) -> list[str]:
    text = clean_text(text).lower()
    return re.findall(r"[a-zA-Z']+", text)