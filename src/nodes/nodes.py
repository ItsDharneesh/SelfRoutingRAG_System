"""LangGraph nodes for router-based Agentic RAG"""

import os
from dotenv import load_dotenv

load_dotenv()

# required by Wikipedia / Tavily
os.environ["USER_AGENT"] = "agentic-rag-project/1.0"

from tavily import TavilyClient
from langchain_core.messages import HumanMessage

from src.state.rag_state import RAGState


# Tavily client
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


class RAGNodes:
    """All node logic lives here"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    # --------------------------------------------------
    # 1. Retrieve from vector DB
    # --------------------------------------------------
    def retrieve_docs(self, state: RAGState) -> RAGState:
        docs = self.retriever.invoke(state.question)
        state.retrieved_docs = docs
        state.debug_retrieved_count = len(docs)
        return state

    # --------------------------------------------------
    # 2. Judge if docs are sufficient
    # --------------------------------------------------
    def judge_docs(self, state: RAGState) -> RAGState:
        # hard fallback
        if not state.retrieved_docs:
            state.debug_judge_decision = "NO_DOCS"
            state.use_web = True
            return state

        context = "\n".join(
            d.page_content[:500] for d in state.retrieved_docs
        )

        prompt = f"""
You are a strict routing controller.

Decide whether the retrieved document context
contains enough factual information
to fully answer the question.

If the answer is clearly present → YES.
If the answer is missing, unrelated, vague,
or requires outside knowledge → NO.

Question:
{state.question}

Document Context:
{context}

Answer ONLY YES or NO.
"""


        decision = (
            self.llm
            .invoke([HumanMessage(content=prompt)])
            .content.strip().upper()
        )

        state.debug_judge_decision = decision
        state.use_web = not decision.startswith("YES")
        return state

    # --------------------------------------------------
    # 3A. Web fallback (Tavily)
    # --------------------------------------------------
    def web_search(self, state: RAGState) -> RAGState:
        result = tavily.search(
            query=state.question,
            search_depth="advanced",
            max_results=5,
        )

        # raw payload (for debugging)
        state.debug_web_raw = str(result)

        # ⚠️ IMPORTANT: Tavily does NOT always return `answer`
        if "answer" in result and result["answer"]:
            web_context = result["answer"]
        else:
            web_context = "\n\n".join(
                r.get("content", "")
                for r in result.get("results", [])
            )

        state.debug_web_context = web_context

        prompt = f"""
Answer the question using the information below.
If there is no single best answer, explain the trade-offs
and list commonly used options with brief justification.


Web context:
{web_context}

Question:
{state.question}
"""

        response = self.llm.invoke(
            [HumanMessage(content=prompt)]
        )

        state.answer = response.content
        return state

    # --------------------------------------------------
    # 3B. Doc-based answer
    # --------------------------------------------------
    def generate_answer(self, state: RAGState) -> RAGState:
        context = "\n".join(
            d.page_content for d in state.retrieved_docs
        )

        prompt = f"""
Answer the question using the information below.

If the question is subjective or comparative:
- Clearly state that there is no single best answer
- Summarize the commonly accepted options
- Explain when each is preferred

Context:
{context}

Question:
{state.question}
"""

        response = self.llm.invoke(
            [HumanMessage(content=prompt)]
        )

        state.answer = response.content
        return state
