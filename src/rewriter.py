import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import logging

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryRewriter:
    """
    Rewrites a user's query to improve retrieval quality.

    Called when DocumentGrader returns "rewrite" decision —
    meaning the original query didn't retrieve relevant chunks.

    The rewriter makes the query more specific and
    aligned with the document vocabulary in our vectorstore.

    Usage:
        rewriter = QueryRewriter()
        better_query = rewriter.rewrite("How to get a job in AI?")
    """

    def __init__(self):
        llm = ChatOllama(
            model=settings.LLM_MODEL,
            temperature=0.3
            # Slightly higher temperature than grader (0.0)
            # because we want some creativity in rephrasing
            # but not too random
        )

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a query rewriting assistant for an AI education system.

Your job is to rewrite a student's question to make it better 
for searching through these documents:
- NEP 2020 (National Education Policy)
- NCF-SE 2023 (National Curriculum Framework)
- India Skills Report 2026

Rules for rewriting:
- Make the query more specific and detailed
- Include relevant keywords from education policy or career domain
- Keep the core intent of the original question
- Return ONLY the rewritten query — no explanation, no prefix like "Rewritten:"
- The rewritten query should be 1-2 sentences maximum"""
            ),
            (
                "human",
                """Original question: {question}

Rewrite this question to improve document retrieval:"""
            )
        ])

        # StrOutputParser extracts just the text string from LLM response
        # No structured output needed here — we just want a plain string
        self.chain = self.prompt | llm | StrOutputParser()
        logger.info("QueryRewriter initialized")

    def rewrite(self, question: str) -> str:
        """
        Rewrites a query for better retrieval.

        Args:
            question: Original user question that failed retrieval

        Returns:
            Rewritten query string
        """
        if not question.strip():
            logger.warning("Empty question received")
            return question

        logger.info(f"Rewriting query: '{question}'")

        try:
            rewritten = self.chain.invoke({"question": question})

            # Clean up any extra whitespace
            rewritten = rewritten.strip()

            logger.info(f"Rewritten query: '{rewritten}'")
            return rewritten

        except Exception as e:
            logger.error(f"Rewriting failed: {e}")
            # If rewriting fails, return original question
            # so the pipeline can still continue
            logger.warning("Returning original query as fallback")
            return question


# ─── Run directly to test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    rewriter = QueryRewriter()

    test_queries = [
        "How to get a job in AI?",
        "What should I study after 12th?",
        "Is coding important?",
        "Tell me about NEP",
    ]

    print("\n--- Query Rewriting Test ---\n")
    for query in test_queries:
        print(f"Original : {query}")
        rewritten = rewriter.rewrite(query)
        print(f"Rewritten: {rewritten}")
        print("-" * 50)