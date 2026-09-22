# Central configuration = all file paths and model names live here.
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_PATH = DATA_DIR / "documents.json"
INTENT_DATA_PATH = DATA_DIR / "intent_training_data.csv"
FORMS_DIR = DATA_DIR / "forms"
CHROMA_PERSIST_DIR = str(BASE_DIR / "chroma_store")
INTENT_MODEL_PATH = BASE_DIR / "backend" / "intent_model.joblib"
EVAL_OUTPUT_DIR = BASE_DIR / "evaluation"

# embedding model (local, via sentence-transformers)
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# generation model (local, via Ollama)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = "qwen2.5:7b-instruct"

# chunking
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80 
STRUCTURED_DOC_MAX_CHARS = 2200
UNSTRUCTURED_DOC_WHOLE_CHUNK_MAX_CHARS = 4000 # in order to prevent cutting of enumerated lists

# Retrieval
TOP_K = 6 # number of chunks to retrieve per query
MIN_SIMILARITY = 0.28  # below this cosine similarity, a chunk is treated as unrelated and dropped
FORMS_MIN_SIMILARITY = 0.45  # a chunk must clear THIS higher bar before its attached forms get shown alongside an answer

# collection name
CHROMA_COLLECTION_NAME = "calamba_city_docs"

# 18 categories: the original 13 from the project brief, plus 5 added new
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