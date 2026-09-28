# central configuration = all file paths and model names live here.
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_PATH = DATA_DIR / "documents.json"
INTENT_DATA_PATH = DATA_DIR / "intent_training_data.csv"
FORMS_DIR = DATA_DIR / "forms"
CHROMA_PERSIST_DIR = str(BASE_DIR / "chroma_store")
INTENT_MODEL_PATH = BASE_DIR / "backend" / "intent_model.joblib"
EVAL_OUTPUT_DIR = BASE_DIR / "evaluation"

# embedding model (sentence-transformers)
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# main generation model then the fallback if the limit is reached
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-120b"

GROQ_FALLBACK_MODELS = [
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]

# different provider fallback if Groq is rate-limited or down account-wide
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

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

# collection name
CHROMA_COLLECTION_NAME = "calamba_city_docs"

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