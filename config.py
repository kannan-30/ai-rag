"""
config.py - Centralised configuration loaded from environment variables.

All tuneable parameters (chunk size, top-k, model names, etc.) live here so
that the rest of the codebase never hard-codes magic strings.
"""

import os
from dotenv import load_dotenv

# Load .env file if present (no-op in production where env vars are injected)
load_dotenv()


def _require(key: str) -> str:
    """Return an env var or raise a clear error if it is missing."""
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            "Copy .env.example → .env and fill in your API keys."
        )
    return value


# ── Pinecone ──────────────────────────────────────────────────────────────────
PINECONE_API_KEY: str = _require("PINECONE_API_KEY")
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-ebook")

# ── Gemini ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = _require("GEMINI_API_KEY")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-3.5-flash")

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K: int = int(os.getenv("TOP_K", "5"))

# ── PDF source ────────────────────────────────────────────────────────────────
PDF_PATH: str = os.getenv("PDF_PATH", "data/Ebook-Agentic-AI.pdf")

# ── FastAPI ───────────────────────────────────────────────────────────────────
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
