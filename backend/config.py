# central configuration = all file paths and model names live here.
import os
from pathlib import Path
from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent
BASE_DIR = BACKEND_DIR.parent

# loads backend/.env for local runs; on Vercel the variables come from the project settings instead
load_dotenv(BACKEND_DIR / ".env")
load_dotenv(BASE_DIR / ".env")

# paths (all resolved relative to this file so they work inside Vercel's serverless runtime)
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_PATH = DATA_DIR / "documents.json"
INTENT_DATA_PATH = DATA_DIR / "intent_training_data.csv"
FORMS_DIR = DATA_DIR / "forms"
FRONTEND_DIR = BASE_DIR / "frontend"
INTENT_MODEL_PATH = BACKEND_DIR / "intent_model.joblib"
EVAL_OUTPUT_DIR = BASE_DIR / "evaluation"

# pre-built static vector store (written by `python -m backend.build_index`, read-only at runtime)
VECTOR_STORE_PATH = DATA_DIR / "vector_store.npz"
CHUNKS_META_PATH = DATA_DIR / "chunks_meta.json"

# embedding model (Google Gemini Embedding API, free tier)
EMBEDDING_MODEL_NAME = "gemini-embedding-001"
EMBEDDING_DIM = 768

def _parse_api_keys(single_var: str, plural_var: str) -> list[str]:
    """Read a single key and/or comma-separated keys from env, preserving order without duplicates."""
    raw = []
    single = os.getenv(single_var, "").strip()
    if single:
        raw.append(single)
    plural = os.getenv(plural_var, "")
    for part in plural.split(","):
        k = part.strip()
        if k:
            raw.append(k)
    seen = set()
    keys = []
    for k in raw:
        if k not in seen:
            seen.add(k)
            keys.append(k)
    return keys


# main generation model then the fallback if the limit is reached
GROQ_API_KEYS = _parse_api_keys("GROQ_API_KEY", "GROQ_API_KEYS")
GROQ_API_KEY = GROQ_API_KEYS[0] if GROQ_API_KEYS else None
GROQ_MODEL = "openai/gpt-oss-120b"

GROQ_FALLBACK_MODELS = [
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]

# different provider fallback if Groq is rate-limited or down account-wide
GEMINI_API_KEYS = _parse_api_keys("GEMINI_API_KEY", "GEMINI_API_KEYS")
GEMINI_API_KEY = GEMINI_API_KEYS[0] if GEMINI_API_KEYS else None
GEMINI_MODEL = "gemini-2.5-flash"

# lightweight model used only for rewriting follow-up questions into standalone ones before retrieval
CONTEXTUALIZE_MODEL = os.getenv("CONTEXTUALIZE_MODEL", "openai/gpt-oss-20b")

# chunking specs
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80 
STRUCTURED_DOC_MAX_CHARS = 2200
UNSTRUCTURED_DOC_WHOLE_CHUNK_MAX_CHARS = 4000 # in order to prevent cutting of enumerated lists; the long answers

# retrieval specs
TOP_K = 4 # number of chunks to retrieve per query
MIN_SIMILARITY = 0.28  # below this cosine similarity, a chunk is treated as unrelated and dropped
FORMS_MIN_SIMILARITY = 0.45 # the threshold before a chunk gets attached with forms

# 18 categories
CATEGORIES = {
    "CERTIFICATES_PERMITS_IDS": "Certificates, Permits, and IDs",
    "BUSINESS_TRADE": "Business and Trade",
    "DISASTER_WEATHER": "Disaster and Weather",
    "REGISTRATIONS_APPLICATIONS": "Registrations, Applications, Ordinances, and Others",
    "EDUCATION_TRAINING": "Education, Training, Seminars, and Competencies",
    "EMPLOYMENT": "Employment",
    "HEALTH": "Health",
    "HOUSING": "Housing",
    "VETERINARY": "Veterinary Services",
    "SOCIAL_SERVICES": "Social Services",
    "TAX": "Tax",
    "AGRICULTURAL": "Agricultural",
    "GENERAL_INFORMATION": "General Information",
    "SECURITY_SAFETY": "Security and Safety",
    "HOUSING_SETTLEMENTS": "Housing and Settlements",
    "GOVERNMENT_LEGAL": "Government and Legal",
    "FINANCE_LOANS": "Finance and Loans",
    "FACILITIES_EQUIPMENT": "Facilities and Equipment",
}