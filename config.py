from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional

# BaseSettings automatically reads from .env file
class Settings(BaseSettings):

    # ─── Project Paths ───────────────────────────────────────────

    BASE_DIR: Path = Path(".")
    DATA_DIR: Path = Path("data")
    VECTORSTORE_DIR: Path = Path("vectorstore")

    # ─── Embedding Model ─────────────────────────────────────────

    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # ─── LLM Settings ────────────────────────────────────────────

    LLM_MODEL: str = "llama3.2"
    LLM_TEMPERATURE: float = 0.0  

    # ─── Chunking Settings ───────────────────────────────────────

    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # ─── Retrieval Settings ──────────────────────────────────────

    RETRIEVAL_TOP_K: int = 2

    # ─── CRAG Settings ───────────────────────────────────────────

    RELEVANCE_THRESHOLD: str = "yes"

    # ─── Web Search ──────────────────────────────────────────────

    TAVILY_API_KEY: Optional[str] = None

    # ─── App Settings ────────────────────────────────────────────
    APP_TITLE: str = "SkillVistaar — AI Career Coach"
    APP_DESCRIPTION: str = (
        "Your personalised AI literacy and career guidance system "
        "aligned with NEP 2020 and India Skills Report 2026."
    )

    class Config:
        
        env_file = ".env"
        env_file_encoding = "utf-8"



settings = Settings()



if __name__ == "__main__":
    print(settings.LLM_MODEL)
    print(settings.DATA_DIR)
    print(settings.TAVILY_API_KEY)
