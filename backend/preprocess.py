import re
import difflib
from backend.config import CHUNK_SIZE, CHUNK_OVERLAP, STRUCTURED_DOC_MAX_CHARS


def build_domain_vocabulary(collection) -> set[str]:
    all_data = collection.get()
    vocab = set()
    for meta in all_data.get("metadatas", []):
        for field in (meta.get("title", ""), meta.get("category", "")):
            vocab.update(w.lower() for w in field.split() if len(w) > 3)
    return vocab

_TYPO_CORRECTION_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "and", "or", "but", "if", "so",
    "do", "does", "did", "can", "could", "will", "would", "should",
    "i", "you", "he", "she", "it", "we", "they", "me", "my", "your",
    "what", "when", "where", "who", "how", "why", "which",
    "this", "that", "these", "those", "with", "from", "by", "as",
    "not", "no", "yes", "please", "give", "get", "list", "all", "under",
}


def correct_typos(text: str, vocabulary: set[str]) -> str:
    words = text.split()
    corrected = []
    for w in words:
        stripped = w.strip(".,?!:;\"'")
        lw = stripped.lower()
        if len(lw) <= 3 or lw in _TYPO_CORRECTION_STOPWORDS or lw in vocabulary:
            corrected.append(w)
            continue
        match = difflib.get_close_matches(lw, vocabulary, n=1, cutoff=0.85)
        corrected.append(match[0] if match else w)
    return " ".join(corrected)


def clean_text(text: str) -> str:
    """Basic cleanup: collapse whitespace, strip stray control characters."""
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


_GREETING_PATTERNS = [
    r"(hi|hello|hey|yo|sup)( there| po| guys| everyone)?",
    r"good (morning|afternoon|evening)( po)?",
    r"(kumusta|kamusta)( po| ka)?",
    r"magandang (umaga|tanghali|hapon|gabi)( po)?",
    r"(hi|hello)(,)? how are you",
    r"thank(s| you)( po| so much)?",
    r"(ok|okay|alright)( thanks| po)?",
]
_GREETING_RE = re.compile(
    r"^(" + "|".join(_GREETING_PATTERNS) + r")[\s!.,?]*$", re.IGNORECASE
)


def is_greeting(text: str) -> bool:
    """True if the message is only a greeting/pleasantry, nothing else."""
    return bool(_GREETING_RE.fullmatch(clean_text(text)))


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character-based chunks."""
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


TERM_ALIASES: dict[str, list[str]] = {
    "doc_tax_001": ["amilyar"],
    "doc_vet_010": ["magpabakuna", "bakuna sa aso", "pet vaccination"],
    "doc_bus_009": ["isara ang negosyo", "cancel business permit", "close down business"],
    "doc_general_001": ["san ang city hall", "oras ng city hall", "city hall address", "city hall hours"],
}


def flatten_structured_document(doc: dict) -> str:
    """Render a structured Citizen's Charter entry into one readable text blob."""
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
    """Turn one document dict into a list of chunk dicts ready for embedding."""
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
    """Lightweight tokenizer used for quick inspection / debugging."""
    text = clean_text(text).lower()
    return re.findall(r"[a-zA-Z']+", text)


def clean_text(text) -> str:
    """Basic cleanup: collapse whitespace, strip stray control characters."""
    if not isinstance(text, str):
        return ""
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()