import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


# ==================================================
# GROQ (LLM CONFIG)
# ==================================================

GROQ_CONFIG = {
    "api_key": os.getenv("GROQ_API_KEY"),
    "base_url": "https://api.groq.com/openai/v1",
    "default_model": "llama-3.1-8b-instant",
    "temperature": 0.1,
}


# ==================================================
# EMBEDDINGS (LOCAL / FREE)
# ==================================================

EMBEDDING_CONFIG = {
    "model_name": "all-MiniLM-L6-v2",
    "dimension": 384
}


# ==================================================
# TAVILY (SEARCH)
# ==================================================

TAVILY_CONFIG = {
    "api_key": os.getenv("TAVILY_API_KEY")
}


# ==================================================
# GEMINI
# ==================================================

GEMINI_CONFIG = {
    "api_key": os.getenv("GEMINI_API_KEY")
}


# ==================================================
# ASSEMBLY AI (SPEECH)
# ==================================================

ASSEMBLYAI_CONFIG = {
    "api_key": os.getenv("ASSEMBLYAI_API_KEY")
}


# ==================================================
# REDIS
# ==================================================

REDIS_CONFIG = {
    "redis_host": os.getenv("REDIS_HOST", "localhost"),
    "redis_port": int(os.getenv("REDIS_PORT", 6379)),
    "redis_db": int(os.getenv("REDIS_DB", 0))
}


# ==================================================
# CHROMA (VECTOR DB)
# ==================================================

CHROMA_CONFIG = {
    "chroma_host": os.getenv("CHROMA_HOST", "localhost"),
    "chroma_port": int(os.getenv("CHROMA_PORT", 8000)),
    "chroma_persist_directory": os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
}


# ==================================================
# MCP SERVERS
# ==================================================

MCP_CONFIG = {
    "servers": {
        "web_research": {
            "url": os.getenv("MCP_WEB_RESEARCH_URL")
        },
        "arxiv_research": {
            "url": os.getenv("MCP_ARXIV_RESEARCH_URL")
        },
        "multimodal_analysis": {
            "url": os.getenv("MCP_MULTIMODAL_ANALYSIS_URL")
        }
    },
    "default_server": "web_research",
    "connection_timeout": 30,
    "retry_attempts": 3
}


# ==================================================
# SUPPORTED FILE TYPES
# ==================================================

SUPPORTED_EXTENSIONS = [
    # Video
    '.mp4', '.mpeg', '.avi', '.mov', '.wmv', '.x-flv', '.webm', '.mpg', '.3gpp',

    # Audio
    '.mp3', '.wav', '.aiff', '.flac', '.aac', '.ogg',

    # Image
    '.jpeg', '.png', '.heic', '.heif', '.webp',

    # Documents
    '.pdf', '.csv', '.md', '.txt', '.html', '.css', '.xml'
]


# ==================================================
# DATA DIRECTORY
# ==================================================

DATA_DIRECTORY_CONFIG = {
    "path": os.getenv("DATA_DIRECTORY_PATH", "./data")
}
