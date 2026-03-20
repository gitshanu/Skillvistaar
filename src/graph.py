import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import logging
from typing import List, Dict, TypedDict

from langchain_core.documents import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import Any
from langgraph.graph import StateGraph, END

from src.retriever import Retriever
from src.grader import DocumentGrader
from src.rewriter import QueryRewriter
from src.generator import AnswerGenerator
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Graph State ──────────────────────────────────────────────────────────────

class CRAGState(TypedDict):
    question:            str
    rewritten_question:  str
    documents:           List[Document]
    decision:            str
    answer:              str
    sources:             List[Dict]
    doc_count:           int
    was_rewritten:       bool
    used_web_search:     bool
    web_search_query:    str


# ─── CRAG Graph Class ─────────────────────────────────────────────────────────

class CRAGGraph:
    """
    Full CRAG pipeline with web search fallback.

    Flow:
        retrieve → grade → relevant?
                              YES → generate → END
                              NO  → rewrite → retrieve → grade → relevant?
                                                                    YES → generate → END
                                                                    NO  → web_search → generate → END

        Special case: time-sensitive questions → skip local docs → web_search → generate → END
    """

    def __init__(self, vectorstore):
        self.retriever  = Retriever(vectorstore)
        self.grader     = DocumentGrader()
        self.rewriter   = QueryRewriter()
        self.generator  = AnswerGenerator()

        # Tavily web search — top 3 results
        self.web_search = TavilySearchResults(
            api_key=settings.TAVILY_API_KEY,
            max_results=3
        )

        self.graph = self._build_graph()
        logger.info("CRAGGraph initialized with web search fallback")

    # ─── Node: Retrieve ───────────────────────────────────────────

    def _node_retrieve(self, state: CRAGState) -> CRAGState:
        logger.info("--- NODE: RETRIEVE ---")
        query = state.get("rewritten_question") or state["question"]
        docs  = self.retriever.retrieve(query)
        return {**state, "documents": docs}

    # ─── Node: Grade ──────────────────────────────────────────────

    def _node_grade(self, state: CRAGState) -> CRAGState:
        logger.info("--- NODE: GRADE ---")
        relevant_docs, decision = self.grader.grade_documents(
            state["question"],
            state["documents"]
        )
        return {**state, "documents": relevant_docs, "decision": decision}

    # ─── Node: Rewrite ────────────────────────────────────────────

    def _node_rewrite(self, state: CRAGState) -> CRAGState:
        logger.info("--- NODE: REWRITE ---")
        rewritten = self.rewriter.rewrite(state["question"])
        return {
            **state,
            "rewritten_question": rewritten,
            "was_rewritten":      True
        }

    # ─── Node: Web Search ─────────────────────────────────────────

    def _node_web_search(self, state: CRAGState) -> CRAGState:
        """
        Fallback node — fires when:
        1. Question is time-sensitive (current news, latest updates etc.)
        2. Local docs failed even after rewrite

        Converts Tavily results into Document objects so the
        generator receives the same format as local chunks.
        """
        logger.info("--- NODE: WEB SEARCH ---")

        search_query = state.get("rewritten_question") or state["question"]
        logger.info(f"Web searching: '{search_query}'")

        try:
            results  = self.web_search.invoke(search_query)
            web_docs = []

            for result in results:
                doc = Document(
                    page_content=result.get("content", ""),
                    metadata={
                        "source": "web_search",
                        "url":    result.get("url", ""),
                        "page":   "web"
                    }
                )
                web_docs.append(doc)

            logger.info(f"Web search returned {len(web_docs)} results")

            return {
                **state,
                "documents":        web_docs,
                "used_web_search":  True,
                "web_search_query": search_query
            }

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return {
                **state,
                "documents":        [],
                "used_web_search":  True,
                "web_search_query": search_query
            }

    # ─── Node: Generate ───────────────────────────────────────────

    def _node_generate(self, state: CRAGState) -> CRAGState:
        logger.info("--- NODE: GENERATE ---")
        result = self.generator.generate(
            state["question"],
            state["documents"]
        )
        return {
            **state,
            "answer":    result["answer"],
            "sources":   result["sources"],
            "doc_count": result["doc_count"]
        }

    # ─── Conditional Edge: After Grade ────────────────────────────

    def _decide_after_grade(self, state: CRAGState) -> str:
        """
        Routes to the correct next node after grading.

        Priority order:
        1. Time-sensitive question → always web search
        2. Grader says generate    → go generate
        3. Already rewritten once  → escalate to web search
        4. First failure           → try rewriting first
        """
        decision      = state.get("decision", "rewrite")
        was_rewritten = state.get("was_rewritten", False)

        # Keywords that signal the user wants live/current data
        time_sensitive_keywords = [
            "current", "latest", "recent", "this week", "this month",
            "today", "news", "trending", "2026 update", "right now",
            "live", "real-time", "newz", "update", "march 2026",
            "right now", "breaking", "just happened", "this year news"
        ]

        question_lower = state["question"].lower()
        is_time_sensitive = any(
            kw in question_lower for kw in time_sensitive_keywords
        )

        if is_time_sensitive:
            logger.info("Routing → WEB SEARCH (time-sensitive question detected)")
            return "web_search"

        if decision == "generate":
            logger.info("Routing → GENERATE")
            return "generate"

        if was_rewritten:
            logger.info("Routing → WEB SEARCH (local docs failed after rewrite)")
            return "web_search"

        logger.info("Routing → REWRITE (first attempt)")
        return "rewrite"

    # ─── Build Graph ──────────────────────────────────────────────

    def _build_graph(self) -> Any:
        workflow = StateGraph(CRAGState)

        # Add all nodes
        workflow.add_node("retrieve",   self._node_retrieve)
        workflow.add_node("grade",      self._node_grade)
        workflow.add_node("rewrite",    self._node_rewrite)
        workflow.add_node("web_search", self._node_web_search)
        workflow.add_node("generate",   self._node_generate)

        # Entry point
        workflow.set_entry_point("retrieve")

        # Fixed edges
        workflow.add_edge("retrieve",   "grade")
        workflow.add_edge("rewrite",    "retrieve")
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("generate",   END)

        # Conditional edge after grading
        workflow.add_conditional_edges(
            "grade",
            self._decide_after_grade,
            {
                "generate":   "generate",
                "rewrite":    "rewrite",
                "web_search": "web_search",
            }
        )

        return workflow.compile()

    # ─── Public Run Method ────────────────────────────────────────

    def run(self, question: str) -> Dict:
        logger.info(f"CRAG pipeline started: '{question}'")

        initial_state: CRAGState = {
            "question":           question,
            "rewritten_question": "",
            "documents":          [],
            "decision":           "",
            "answer":             "",
            "sources":            [],
            "doc_count":          0,
            "was_rewritten":      False,
            "used_web_search":    False,
            "web_search_query":   ""
        }

        final_state = self.graph.invoke(initial_state)

        return {
            "question":           final_state["question"],
            "answer":             final_state["answer"],
            "sources":            final_state["sources"],
            "doc_count":          final_state["doc_count"],
            "was_rewritten":      final_state["was_rewritten"],
            "rewritten_question": final_state.get("rewritten_question", ""),
            "used_web_search":    final_state["used_web_search"],
            "web_search_query":   final_state.get("web_search_query", "")
        }


# ─── Run directly to test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.embeddings import EmbeddingStore

    store      = EmbeddingStore()
    vectorstore = store.load()
    graph      = CRAGGraph(vectorstore)

    tests = [
        "What does NEP 2020 say about AI in education?",
        "What is the latest news in AI jobs this week?",
    ]

    for question in tests:
        print(f"\n{'='*55}")
        print(f"Question: {question}")
        print(f"{'='*55}")
        result = graph.run(question)
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nWeb search used: {result['used_web_search']}")
        print(f"Was rewritten  : {result['was_rewritten']}")
        for src in result["sources"]:
            if src.get("url"):
                print(f"  🌐 {src['url']}")
            else:
                print(f"  📄 {src['file']} — Page {src['page']}")
