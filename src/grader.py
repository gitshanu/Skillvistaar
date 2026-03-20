import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import logging
from typing import List, Tuple

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Structured Output Schema ─────────────────────────────────────────────────
# This Pydantic model forces the LLM to return ONLY "yes" or "no"
# Field(description=...) tells the LLM what this field means

class GradeResult(BaseModel):
    """
    Binary relevance score for a document chunk.
    The LLM must return exactly this structure.
    """
    score: str = Field(
        description="Is the document relevant to the question? Answer 'yes' or 'no'."
    )


# ─── DocumentGrader Class ─────────────────────────────────────────────────────

class DocumentGrader:
    """
    Uses an LLM to evaluate whether retrieved chunks are
    actually relevant to the user's question.

    This is the CRAG core — it decides whether to:
        - Use the retrieved docs (if relevant)
        - Trigger correction (if not relevant)

    Usage:
        grader = DocumentGrader()
        results = grader.grade_documents(question, docs)
    """

    def __init__(self):
        # Initialize the LLM
        # temperature=0 → we want consistent, deterministic grading
        # not creative answers
        llm = ChatOllama(
            model=settings.LLM_MODEL,
            temperature=0
        )

        # .with_structured_output() forces LLM to return GradeResult shape
        # This means we always get { score: "yes" } or { score: "no" }
        # Never a free-form paragraph
        self.structured_llm = llm.with_structured_output(GradeResult)

        # The grading prompt
        # Notice: we give very clear instructions
        # "do not" is important — stops the LLM from overthinking
        self.prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a document relevance grader for an AI education assistant.

Your job is to assess whether a retrieved document chunk can ACTUALLY answer 
the student's question.

Rules:
- If the question asks for CURRENT news, latest updates, recent events, 
  this week/month/year live data → ALWAYS return 'no' for static PDF chunks.
  Static PDFs cannot answer real-time questions no matter their content.
- If the document contains useful information directly related to the question → 'yes'
- If the document is unrelated → 'no'
- Partial relevance counts as 'yes' ONLY for non-time-sensitive questions

Time-sensitive keywords that always mean 'no' for static docs:
current, latest, recent, this week, this month, today, news, 
trending now, 2026 updates, live, real-time, right now"""
    ),
    (
        "human",
        """Question: {question}

Retrieved Document:
{document}

Is this document relevant AND sufficient to answer the question?
Answer 'yes' or 'no' only."""
    )
])

        # Chain: prompt → structured LLM
        self.chain = self.prompt | self.structured_llm
        logger.info("DocumentGrader initialized")

    def grade_single_document(
        self,
        question: str,
        document: Document
    ) -> Tuple[Document, str]:
        """
        Grades a single document chunk against the question.

        Args:
            question : The user's question
            document : A single retrieved chunk

        Returns:
            Tuple of (document, score) where score is "yes" or "no"
        """
        try:
            result = self.chain.invoke({
                "question": question,
                "document": document.page_content
            })

            score = result.score.strip().lower() # type: ignore

            # Safety check — make sure we only get yes/no
            if score not in ["yes", "no"]:
                logger.warning(f"Unexpected score '{score}' — defaulting to 'no'")
                score = "no"

            logger.info(
                f"Graded chunk from {document.metadata.get('source')} "
                f"page {document.metadata.get('page')} → {score.upper()}"
            )
            return (document, score)

        except Exception as e:
            logger.error(f"Grading failed: {e}")
            # On error, default to "no" so correction is triggered
            return (document, "no")

    def grade_documents(
        self,
        question: str,
        documents: List[Document]
    ) -> Tuple[List[Document], str]:
        """
        Grades all retrieved documents and decides the overall pipeline action.

        Logic:
            - All docs relevant    → "generate"   (go straight to answer)
            - Some docs relevant   → "generate"   (use only the relevant ones)
            - No docs relevant     → "rewrite"    (trigger correction)

        Args:
            question  : The user's question
            documents : List of retrieved chunks

        Returns:
            Tuple of:
                - filtered_docs : Only the relevant chunks
                - decision      : "generate" or "rewrite"
        """
        if not documents:
            logger.warning("No documents to grade")
            return [], "rewrite"

        logger.info(f"Grading {len(documents)} chunks...")

        relevant_docs = []
        not_relevant_count = 0

        # Grade each chunk individually
        for doc in documents:
            graded_doc, score = self.grade_single_document(question, doc)

            if score == "yes":
                relevant_docs.append(graded_doc)
            else:
                not_relevant_count += 1

        # ─── Decision Logic ───────────────────────────────────────
        total = len(documents)
        relevant = len(relevant_docs)

        logger.info(f"Grading complete: {relevant}/{total} chunks relevant")

        if relevant == 0:
            # No relevant docs at all → trigger correction
            logger.info("Decision: REWRITE — no relevant chunks found")
            return [], "rewrite"
        else:
            # At least some relevant docs → proceed to generate
            logger.info(f"Decision: GENERATE — using {relevant} relevant chunks")
            return relevant_docs, "generate"


# ─── Run directly to test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.embeddings import EmbeddingStore
    from src.retriever import Retriever

    # Load vectorstore and retriever
    store = EmbeddingStore()
    vectorstore = store.load()
    retriever = Retriever(vectorstore)

    grader = DocumentGrader()

    # Test 1: Relevant question — should return "generate"
    print("\n" + "="*50)
    print("TEST 1: Relevant question")
    print("="*50)
    question1 = "What does NEP 2020 say about vocational education?"
    docs1 = retriever.retrieve(question1)
    relevant_docs1, decision1 = grader.grade_documents(question1, docs1)
    print(f"\nQuestion : {question1}")
    print(f"Decision : {decision1}")
    print(f"Relevant chunks kept: {len(relevant_docs1)}/{len(docs1)}")

    # Test 2: Irrelevant question — should return "rewrite"
    print("\n" + "="*50)
    print("TEST 2: Irrelevant question")
    print("="*50)
    question2 = "What is the recipe for biryani?"
    docs2 = retriever.retrieve(question2)
    relevant_docs2, decision2 = grader.grade_documents(question2, docs2)
    print(f"\nQuestion : {question2}")
    print(f"Decision : {decision2}")
    print(f"Relevant chunks kept: {len(relevant_docs2)}/{len(docs2)}")