from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Put all knowledge files here:
# TXT / MD / PDF / HTML
DATA_DIR = BASE_DIR / "data" / "knowledge"


# ============================================================
# QDRANT
# ============================================================

QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "edubot_knowledge"


# ============================================================
# EMBEDDINGS
# ============================================================

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


# ============================================================
# CHUNKING
# ============================================================

CHUNK_WORDS = 300
CHUNK_OVERLAP = 50


# ============================================================
# RETRIEVAL
# ============================================================

TOP_K = 5
MIN_SCORE = 0.25


# ============================================================
# LLM
# ============================================================

GROQ_MODEL = "openai/gpt-oss-120b"

MAX_HISTORY_TURNS = 5


# ============================================================
# FALLBACK
# ============================================================

FALLBACK_REPLY = (
    "I couldn't find enough relevant information in my knowledge base "
    "to answer that confidently."
)