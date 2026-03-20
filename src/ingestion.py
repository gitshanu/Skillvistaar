import sys
from pathlib import Path

# This adds the project root to Python's search path
# So "from config import settings" works from anywhere
sys.path.append(str(Path(__file__).parent.parent))

import logging
import requests
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

from langchain_community.document_loaders import WikipediaLoader

from config import settings

# ─── Logging Setup ───────────────────────────────────────────────────────────
# logging is like print() but professional
# It shows timestamps, severity levels (INFO, WARNING, ERROR)
# Every production Python file should have this

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Document URLs ───────────────────────────────────────────────────────────
# All our source PDFs in one dictionary
# Key   = filename we save as (in data/ folder)
# Value = public URL to download from

DOCUMENT_URLS = {
    "nep_2020.pdf": (
        "https://www.education.gov.in/sites/upload_files/mhrd/files/NEP_Final_English_0.pdf"
    ),
    "ncf_se_2023.pdf": (
        "https://www.ncert.nic.in/pdf/NCFSE-2023-August_2023.pdf"
    ),
    "india_skills_2026.pdf": (
        "https://wheebox.com/assets/pdf/ISR_Report_2026.pdf"
    ),
}


# ─── DocumentIngestion Class ─────────────────────────────────────────────────

class DocumentIngestion:
    """
    Handles everything related to getting documents ready for the pipeline.

    Responsibilities:
        1. Download PDFs from URLs into the data/ folder
        2. Load PDFs into LangChain Document objects
        3. Split documents into chunks for embedding

    Usage:
        ingestion = DocumentIngestion()
        chunks = ingestion.run()
    """

    def __init__(self):
        settings.DATA_DIR.mkdir(exist_ok=True)

        # This is the splitter that will chunk our documents
        # We define it once here and reuse it for all documents
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,      
            chunk_overlap=settings.CHUNK_OVERLAP,
            # first try paragraph breaks, then sentences, then words
            separators=["\n\n", "\n", " ", ""]
        )

    # ─── Step 1: Download ────────────────────────────────────────

    def download_documents(self) -> None:
        """
        Downloads all PDFs defined in DOCUMENT_URLS into the data/ folder.
        Skips files that already exist (so re-running won't re-download).
        """
        logger.info("Starting document downloads...")

        for filename, url in DOCUMENT_URLS.items():
            save_path = settings.DATA_DIR / filename

            # Skip if already downloaded
            if save_path.exists():
                logger.info(f"Already exists, skipping: {filename}")
                continue

            try:
                logger.info(f"Downloading: {filename}")

                # stream=True means we download in chunks
                # important for large files — doesn't load entire PDF into RAM at once
                response = requests.get(url, stream=True, timeout=60)
                response.raise_for_status()  # raises error if download failed

                # Write the file to disk in binary mode
                save_path.write_bytes(response.content)
                logger.info(f"Saved: {filename}")

            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to download {filename}: {e}")
                logger.warning(f"Skipping {filename} — add it manually to data/ folder")

    # ─── Step 2: Load ────────────────────────────────────────────

    def load_documents(self) -> List[Document]:
        """
        Loads all PDFs from the data/ folder into LangChain Document objects.

        A LangChain Document has two things:
            - page_content : the actual text of that page
            - metadata     : info about the document (filename, page number etc.)

        Returns:
            List of Document objects — one per PAGE of each PDF
        """
        all_documents = []

        # Find every PDF in the data/ folder
        pdf_files = list(settings.DATA_DIR.glob("*.pdf"))

        if not pdf_files:
            logger.warning("No PDFs found in data/ folder. Run download first.")
            return []

        for pdf_path in pdf_files:
            try:
                logger.info(f"Loading: {pdf_path.name}")

                # PyMuPDFLoader is fast and handles scanned + text PDFs well
                loader = PyMuPDFLoader(str(pdf_path))
                documents = loader.load()

                # Add source filename to each document's metadata
                # This is important — when we return an answer we can say
                # "this came from nep_2020.pdf page 12"
                for doc in documents:
                    doc.metadata["source"] = pdf_path.name

                all_documents.extend(documents)
                logger.info(f"Loaded {len(documents)} pages from {pdf_path.name}")

            except Exception as e:
                logger.error(f"Failed to load {pdf_path.name}: {e}")

        logger.info(f"Total pages loaded: {len(all_documents)}")
        return all_documents

    # ─── Step 3: Split ───────────────────────────────────────────

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits loaded documents into smaller chunks.

        Why: LLMs and embedding models have token limits.
             Smaller chunks = more precise retrieval.

        Args:
            documents: List of full-page Document objects

        Returns:
            List of chunked Document objects (much larger list)
        """
        if not documents:
            logger.warning("No documents to split.")
            return []

        chunks = self.splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks")
        return chunks

    # ─── Main Runner ─────────────────────────────────────────────

    def run(self) -> List[Document]:
        """
        Runs the full ingestion pipeline:
        download → load → split → return chunks

        This is the only method you call from outside this class.

        Returns:
            List of document chunks ready for embedding
        """
        logger.info("=== Starting Document Ingestion Pipeline ===")

        # Step 1
        self.download_documents()

            

        # Step 2
        documents = self.load_documents()
        wiki_docs = self.load_wikipedia_sources() 
        documents.extend(wiki_docs)
        if not documents:
            logger.error("No documents loaded. Cannot continue.")
            return []

        # Step 3
        chunks = self.split_documents(documents)

        logger.info(f"=== Ingestion Complete: {len(chunks)} chunks ready ===")
        return chunks
    

# def load_wikipedia_sources(self) -> List[Document]:
#     """
#     Loads Wikipedia articles for broader context.
#     Covers topics our PDFs might not fully address.
#     """
#     topics = [
#         "National Education Policy 2020",
#         "Skill India",
#         "National Skill Development Corporation",
#         "Artificial intelligence in education",
#         "India labour law",
#     ]

#     wiki_docs = []
#     for topic in topics:
#         try:
#             logger.info(f"Loading Wikipedia: {topic}")
#             loader = WikipediaLoader(
#                 query=topic,
#                 load_max_docs=1,       # 1 article per topic
#                 doc_content_chars_max=3000  # limit size
#             )
#             docs = loader.load()
#             for doc in docs:
#                 doc.metadata["source"] = f"wikipedia_{topic[:20]}.txt"
#             wiki_docs.extend(docs)
#         except Exception as e:
#             logger.warning(f"Wikipedia load failed for {topic}: {e}")

#     logger.info(f"Loaded {len(wiki_docs)} Wikipedia articles")
#     return wiki_docs


    def load_wikipedia_sources(self) -> List[Document]:
        """
        Loads Wikipedia articles for broader context.
        Covers topics our PDFs might not fully address.
        """
        topics = [
            "National Education Policy 2020",
            "Skill India",
            "National Skill Development Corporation",
            "Artificial intelligence in education",
            "India labour law",
        ]

        wiki_docs = []
        for topic in topics:
            try:
                logger.info(f"Loading Wikipedia: {topic}")
                loader = WikipediaLoader(
                    query=topic,
                    load_max_docs=1,       # 1 article per topic
                    doc_content_chars_max=3000  # limit size
                )
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = f"wikipedia_{topic[:20]}.txt"
                wiki_docs.extend(docs)
            except Exception as e:
                logger.warning(f"Wikipedia load failed for {topic}: {e}")

        logger.info(f"Loaded {len(wiki_docs)} Wikipedia articles")
        return wiki_docs


# ─── Run directly to test ─────────────────────────────────────────────────────
# When you run: python src/ingestion.py
# It runs the full pipeline and prints a sample chunk

if __name__ == "__main__":
    ingestion = DocumentIngestion()
    chunks = ingestion.run()

    # Print first chunk to verify everything worked
    if chunks:
        print("\n--- Sample Chunk ---")
        print(f"Source  : {chunks[0].metadata.get('source')}")
        print(f"Page    : {chunks[0].metadata.get('page')}")
        print(f"Content : {chunks[0].page_content[:300]}")
        print(f"\nTotal chunks: {len(chunks)}")