import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import logging
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Retriever:
    """
    Handles retrieving relevant document chunks for a given query.

    Takes a user question, searches the FAISS vectorstore,
    and returns the top K most relevant chunks.

    Usage:
        retriever = Retriever(vectorstore)
        docs = retriever.retrieve("What is NEP 2020?")
    """

    def __init__(self, vectorstore: FAISS):
        """
        Args:
            vectorstore: Loaded FAISS vectorstore from EmbeddingStore
        """
        self.vectorstore = vectorstore

        # LangChain's built-in retriever interface
        # search_type="similarity" → cosine similarity search
        # k = how many chunks to return per query
        self.retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.RETRIEVAL_TOP_K}
        )

        logger.info(f"Retriever ready — will return top {settings.RETRIEVAL_TOP_K} chunks")

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieves top K relevant chunks for a given query.

        Args:
            query: User's question as a string

        Returns:
            List of Document objects (chunks) most relevant to the query
        """
        if not query.strip():
            logger.warning("Empty query received")
            return []

        logger.info(f"Retrieving docs for: '{query}'")

        # invoke() converts query to vector and searches FAISS
        docs = self.retriever.invoke(query)

        logger.info(f"Retrieved {len(docs)} chunks")
        return docs

    def retrieve_with_scores(self, query: str):
        """
        Same as retrieve() but also returns similarity scores.

        Useful for debugging — you can see HOW relevant each chunk is.
        Score closer to 1.0 = very relevant
        Score closer to 0.0 = not relevant

        Args:
            query: User's question

        Returns:
            List of (Document, score) tuples
        """
        if not query.strip():
            return []

        results = self.vectorstore.similarity_search_with_score(
            query,
            k=settings.RETRIEVAL_TOP_K
        )

        # Log each result's score for visibility
        for i, (doc, score) in enumerate(results):
            logger.info(
                f"Chunk {i+1} | Score: {score:.4f} | "
                f"Source: {doc.metadata.get('source')} | "
                f"Page: {doc.metadata.get('page')}"
            )

        return results

    def format_docs(self, docs: List[Document]) -> str:
        """
        Formats a list of Document chunks into a single string.

        This formatted string is what gets passed to the LLM as context.
        Each chunk is separated clearly so the LLM knows where one ends
        and another begins.

        Args:
            docs: List of Document chunks

        Returns:
            Single formatted string of all chunks
        """
        if not docs:
            return "No relevant documents found."

        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "unknown")

            formatted.append(
                f"[Chunk {i+1} | Source: {source} | Page: {page}]\n"
                f"{doc.page_content}"
            )

        # Join all chunks with a clear separator
        return "\n\n---\n\n".join(formatted)


# ─── Run directly to test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.embeddings import EmbeddingStore

    # Load existing vectorstore
    store = EmbeddingStore()
    vectorstore = store.load()

    # Create retriever
    retriever = Retriever(vectorstore)

    # Test queries
    test_queries = [
        "What is NEP 2020?",
        "What are the career opportunities in AI for students?",
        "What does NCF say about foundational literacy?",
    ]

    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")

        # Test with scores so we can see relevance
        results = retriever.retrieve_with_scores(query)

        for doc, score in results:
            print(f"\nScore  : {score:.4f}")
            print(f"Source : {doc.metadata.get('source')}")
            print(f"Page   : {doc.metadata.get('page')}")
            print(f"Content: {doc.page_content[:200]}...")