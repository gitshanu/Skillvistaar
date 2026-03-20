import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import logging
from typing import List

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingStore:
    """
    Handles converting document chunks into vectors
    and storing/loading them in a FAISS index.

    Two main jobs:
        1. build()  — embed chunks and save FAISS index to disk
        2. load()   — load existing FAISS index from disk

    Usage:
        store = EmbeddingStore()
        store.build(chunks)       # first time
        db = store.load()         # every time after
    """

    def __init__(self):
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")

        # HuggingFaceEmbeddings runs the model locally
        # model_kwargs device: cpu  → runs on CPU (works on any laptop)
        # encode_kwargs normalize  → normalizes vectors for better similarity
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        # Where to save/load the FAISS index on disk
        self.vectorstore_path = str(settings.VECTORSTORE_DIR)

        # Make sure vectorstore/ folder exists
        settings.VECTORSTORE_DIR.mkdir(exist_ok=True)

    def build(self, chunks: List[Document]) -> FAISS:
        """
        Converts chunks into vectors and saves FAISS index to disk.

        This is slow the first time (embedding 739 chunks takes ~1-2 min).
        After that you never need to run this again unless documents change.

        Args:
            chunks: List of Document chunks from ingestion pipeline

        Returns:
            FAISS vectorstore object
        """
        if not chunks:
            raise ValueError("No chunks provided to embed.")

        logger.info(f"Embedding {len(chunks)} chunks...")
        logger.info("This may take 1-2 minutes on first run...")

        # FAISS.from_documents does two things internally:
        # 1. Calls embedding model on every chunk's text
        # 2. Builds a searchable index from those vectors
        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )

        # Save to disk so we don't re-embed every time we run the app
        # Creates two files in vectorstore/:
        #   index.faiss  → the actual vectors
        #   index.pkl    → the document metadata
        vectorstore.save_local(self.vectorstore_path)
        logger.info(f"FAISS index saved to: {self.vectorstore_path}")

        return vectorstore

    def load(self) -> FAISS:
        """
        Loads existing FAISS index from disk.

        Call this every time AFTER the first build.
        Much faster than rebuilding — loads in seconds.

        Returns:
            FAISS vectorstore object ready for search
        """
        index_file = settings.VECTORSTORE_DIR / "index.faiss"

        if not index_file.exists():
            raise FileNotFoundError(
                "No FAISS index found. Run build() first.\n"
                "Hint: python src/embeddings.py"
            )

        logger.info("Loading FAISS index from disk...")

        # allow_dangerous_deserialization=True is required by newer FAISS
        # it's safe here because WE created this file ourselves
        vectorstore = FAISS.load_local(
            self.vectorstore_path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )

        logger.info("FAISS index loaded successfully")
        return vectorstore

    def build_or_load(self, chunks: List[Document] = None) -> FAISS: # type: ignore
        """
        Smart method — builds if index doesn't exist, loads if it does.

        This is what the main app will call.
        You never have to think about whether to build or load.

        Args:
            chunks: Only needed if building for first time

        Returns:
            FAISS vectorstore object
        """
        index_file = settings.VECTORSTORE_DIR / "index.faiss"

        if index_file.exists():
            logger.info("Existing index found — loading from disk")
            return self.load()
        else:
            logger.info("No index found — building from scratch")
            if not chunks:
                raise ValueError(
                    "No chunks provided and no existing index found. "
                    "Pass chunks to build the index first."
                )
            return self.build(chunks)


# ─── Run directly to test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # Import ingestion to get chunks
    from src.ingestion import DocumentIngestion

    # Step 1: Get chunks
    ingestion = DocumentIngestion()
    chunks = ingestion.run()

    # Step 2: Build vectorstore
    store = EmbeddingStore()
    vectorstore = store.build(chunks)

    # Step 3: Quick test search
    print("\n--- Test Search ---")
    query = "What is NEP 2020?"
    results = vectorstore.similarity_search(query, k=2)

    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Source : {doc.metadata.get('source')}")
        print(f"Page   : {doc.metadata.get('page')}")
        print(f"Content: {doc.page_content[:200]}")