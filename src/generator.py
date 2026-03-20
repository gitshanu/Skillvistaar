import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import logging
from typing import List, Dict

from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnswerGenerator:
    """
    Generates a final answer from relevant document chunks.

    This is the last node in the CRAG pipeline.
    Takes verified relevant chunks + question → produces answer + sources.

    Usage:
        generator = AnswerGenerator()
        result = generator.generate(question, relevant_docs)
    """

    def __init__(self):
        self.llm = ChatOllama(
            model=settings.LLM_MODEL,
            temperature=0.1,
             num_predict=512
            # Low temperature for factual accuracy
            # Slight non-zero allows natural language flow
        )

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are SkillVistaar, an AI career and education coach
for Indian students aligned with NEP 2020.

Your job is to answer student questions using ONLY the provided context.

STRICT FORMATTING RULES — always follow these:
- Start with one clear sentence summarizing the answer
- Use bullet points (•) for lists of 3 or more items
- Use **bold** for important terms or policy names
- Add a new line between each point for readability
- End with one encouraging sentence for the student
- Keep total response under 300 words
- If context is insufficient, say exactly:
  "I don't have enough information on this in my documents.
   Try asking about NEP 2020, career skills, or AI literacy."
- Do NOT use phrases like "I'd be happy to" or "I hope this helps"
- Get straight to the answer immediately"""
            ),
            (
                "human",
                """Context from documents:
{context}

Student's Question: {question}

Please answer the question based on the context above:"""
            )
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()
        logger.info("AnswerGenerator initialized")

    def _format_context(self, docs: List[Document]) -> str:
        """
        Formats document chunks into a clean context string for the LLM.

        Each chunk is clearly labelled with its source so the LLM
        knows where the information is coming from.
        """
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "unknown")
            formatted.append(
                f"[Source {i+1}: {source}, Page {page}]\n{doc.page_content}"
            )
        return "\n\n".join(formatted)

    # def _extract_sources(self, docs: List[Document]) -> List[Dict]:
    #     """
    #     Extracts source information from documents for citation display.

    #     Returns a clean list of sources the user can see in the UI.
    #     """
    #     sources = []
    #     seen = set()  # avoid duplicate sources

    #     for doc in docs:
    #         source = doc.metadata.get("source", "unknown")
    #         page = doc.metadata.get("page", "unknown")

    #         # Create unique key to avoid duplicates
    #         key = f"{source}_p{page}"
    #         if key not in seen:
    #             seen.add(key)
    #             sources.append({
    #                 "file": source,
    #                 "page": page,
    #             })

    #     return sources

    def _extract_sources(self, docs: List[Document]) -> List[Dict]:
     sources = []
     seen = set()

     for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "unknown")
        url    = doc.metadata.get("url", "")

        key = url if url else f"{source}_p{page}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "file": source,
                "page": page,
                "url":  url       # include URL for web results
            })
 
     return sources

    def generate(
        self,
        question: str,
        docs: List[Document]
    ) -> Dict:
        """
        Generates a complete answer with sources.

        Args:
            question : The user's question
            docs     : Verified relevant document chunks

        Returns:
            Dictionary with:
                - answer   : The generated answer string
                - sources  : List of source citations
                - doc_count: Number of chunks used
        """
        if not docs:
            return {
                "answer": (
                    "I couldn't find relevant information in my documents "
                    "to answer this question. Please try rephrasing or "
                    "ask about NEP 2020, career guidance, or AI literacy topics."
                ),
                "sources": [],
                "doc_count": 0
            }

        logger.info(f"Generating answer using {len(docs)} chunks...")

        # Format chunks into context string
        context = self._format_context(docs)

        try:
            # Generate the answer
            answer = self.chain.invoke({
                "question": question,
                "context": context
            })

            # Extract sources for display
            sources = self._extract_sources(docs)

            logger.info("Answer generated successfully")

            return {
                "answer": answer.strip(),
                "sources": sources,
                "doc_count": len(docs)
            }

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {
                "answer": "Sorry, I encountered an error generating the answer. Please try again.",
                "sources": [],
                "doc_count": 0
            }


# ─── Run directly to test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.embeddings import EmbeddingStore
    from src.retriever import Retriever
    from src.grader import DocumentGrader

    # Load everything
    store = EmbeddingStore()
    vectorstore = store.load()
    retriever = Retriever(vectorstore)
    grader = DocumentGrader()
    generator = AnswerGenerator()

    # Test question
    question = "What does NEP 2020 say about vocational education for students?"

    print(f"\n{'='*55}")
    print(f"Question: {question}")
    print(f"{'='*55}")

    # Step 1: Retrieve
    docs = retriever.retrieve(question)
    print(f"\nRetrieved: {len(docs)} chunks")

    # Step 2: Grade
    relevant_docs, decision = grader.grade_documents(question, docs)
    print(f"Decision : {decision}")
    print(f"Relevant : {len(relevant_docs)} chunks")

    # Step 3: Generate
    result = generator.generate(question, relevant_docs)

    print(f"\n--- Answer ---")
    print(result["answer"])

    print(f"\n--- Sources ---")
    for src in result["sources"]:
        print(f"  {src['file']} — Page {src['page']}")