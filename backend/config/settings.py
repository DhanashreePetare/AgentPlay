import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")

# --- Redis ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_TTL_TRANSCRIPT = 60 * 60 * 24       # 24 hours
REDIS_TTL_SUMMARY = 60 * 60 * 24 * 7      # 7 days
REDIS_TTL_NOTES = 60 * 60 * 24 * 7        # 7 days

# --- Qdrant ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = "agentplay_transcripts"

# --- LLM Models ---
GEMINI_CHAT_MODEL = "gemini-2.5-flash"
GEMINI_LLM_MODEL = "gemini/gemini-2.5-flash"
GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"
MISTRAL_MODEL = "mistral-large-latest"

# --- Rate Limiting ---
RATE_LIMIT_DELAY = 3
MAX_RETRIES = 3
MAX_CONCURRENT_REQUESTS = 1

# --- RAG ---
SIMILARITY_THRESHOLD = 0.3
CHUNK_SIZE_WORDS = 50
TRANSCRIPT_CHUNK_SIZE_CHARS = 8000

# --- Audio ---
AUDIO_DATA_DIR = "data"
AUDIO_SEGMENT_TIMEOUT = 30
AUDIO_TOTAL_TIMEOUT = 300

# --- App ---
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", 5000))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"