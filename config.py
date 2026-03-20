from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional

# BaseSettings automatically reads from .env file
class Settings(BaseSettings):

    # ─── Project Paths ───────────────────────────────────────────
    # Path(".")  means "current directory" (where you run the project from)
    # These tell every other file WHERE to find data and save the vectorstore

    BASE_DIR: Path = Path(".")
    DATA_DIR: Path = Path("data")
    VECTORSTORE_DIR: Path = Path("vectorstore")

    # ─── Embedding Model ─────────────────────────────────────────
    # This is the model that converts text into numbers (vectors)
    # all-MiniLM-L6-v2 runs locally, no API key needed

    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # ─── LLM Settings ────────────────────────────────────────────
    # The local LLM running via Ollama
    # Used for: grading docs, rewriting queries, generating answers

    LLM_MODEL: str = "llama3.2"
    LLM_TEMPERATURE: float = 0.0   # 0.0 = focused/deterministic answers
                                    # 1.0 = creative/random answers
                                    # For RAG we always want 0.0

    # ─── Chunking Settings ───────────────────────────────────────
    # When we load a PDF, we split it into small chunks
    # CHUNK_SIZE    = how many characters per chunk
    # CHUNK_OVERLAP = how many characters overlap between chunks
    #                 (overlap helps so we don't lose context at boundaries)

    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # ─── Retrieval Settings ──────────────────────────────────────
    # How many chunks to retrieve for each question

    RETRIEVAL_TOP_K: int = 2

    # ─── CRAG Settings ───────────────────────────────────────────
    # This is the confidence threshold for the grader
    # If grader scores a doc below this → trigger correction
    # "relevant" or "not relevant" in our case (binary decision)

    RELEVANCE_THRESHOLD: str = "yes"

    # ─── Web Search ──────────────────────────────────────────────
    # Tavily API key — automatically loaded from your .env file
    # The "default=None" means app won't crash if key is missing
    # but web search simply won't work

    TAVILY_API_KEY: Optional[str] = None

    # ─── App Settings ────────────────────────────────────────────
    APP_TITLE: str = "SkillVistaar — AI Career Coach"
    APP_DESCRIPTION: str = (
        "Your personalised AI literacy and career guidance system "
        "aligned with NEP 2020 and India Skills Report 2026."
    )

    class Config:
        # Tell Pydantic to read from .env file automatically
        env_file = ".env"
        env_file_encoding = "utf-8"


# Create ONE instance that the whole project imports
# Every file does: from config import settings
settings = Settings()



if __name__ == "__main__":
    print(settings.LLM_MODEL)
    print(settings.DATA_DIR)
    print(settings.TAVILY_API_KEY)