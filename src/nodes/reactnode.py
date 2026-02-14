"""LangGraph nodes for RAG workflow + ReAct Agnet inside generate_content"""

import uuid
from typing import List, Optional
from langchain_core.tools import StructuredTool
from src.state.rag_state import RAGState

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# Wikipedia tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun


class RAGNodes:
    """Contains the node function for RAG workflow"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.agent = None  ## lazy init agent

    def retrieve_docs(self, state: RAGState) -> RAGState:
        """Classic retriever node"""

        docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )

    ## Build Tools
    def _build_tools(self):
        """Build retriever + wikipedia tools"""

        def retriever_tool_fn(query):
            docs = self.retriever.invoke(query)
            if not docs:
                return "No documents found"

            merged = []
            for i, d in enumerate(docs[:8], start=1):
                meta = d.metadata if hasattr(d, "metadata") else {}
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
                merged.append(f"[{i}] {title}\n{d.page_content}")

            return "\n\n".join(merged)

        retriever_tool = StructuredTool.from_function(
            func=retriever_tool_fn,
            name="retriever",
            description="Fetch passages from indexed vectorstore",
            infer_schema=False,
        )

        wiki_api = WikipediaAPIWrapper(top_k_results=3, lang="en")
        wikipedia_tool = StructuredTool.from_function(
            func=WikipediaQueryRun(api_wrapper=wiki_api).run,
            name="wikipedia",
            description="Search Wikipedia for general knowledge",
            infer_schema=False,
        )

        return [retriever_tool, wikipedia_tool]

    ## build agent
    def _build_agent_(self):
        """ReAct agent with tools"""

        tools = self._build_tools()
        system_prompt = (
            "You are a helpful RAG agent. "
            "Prefer 'retriever' for user-provided docs; "
            "use 'wikipedia' for general knowledge. "
            "Return only the final useful answer."
        )

        self.agent = create_react_agent(
            self.llm,
            tools=tools,
            prompt=system_prompt
        )

    def generate_answer(self, state: RAGState) -> RAGState:
        """Generate answer using ReAct agent with retriever + wikipedia"""

        if self.agent is None:
            self._build_agent_()

        result = self.agent.invoke(
            {"messages": [HumanMessage(content=state.question)]}
        )

        messages = result.get("messages", [])
        answer: Optional[str] = None

        if messages:
            answer = getattr(messages[-1], "content", None)

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer or "Could not generate answer"
        )
