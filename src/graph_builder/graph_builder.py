"""LangGraph builder for router-based Agentic RAG"""
from langgraph.graph import StateGraph, END
from src.state.rag_state import RAGState
from src.nodes.nodes import RAGNodes


class GraphBuilder:
    """Builds router-based Agentic RAG graph"""

    def __init__(self, retriever, llm):
        self.nodes = RAGNodes(retriever, llm)

    def build(self):
        graph = StateGraph(RAGState)

        # nodes
        graph.add_node("retrieve", self.nodes.retrieve_docs)
        graph.add_node("judge", self.nodes.judge_docs)
        graph.add_node("doc_answer", self.nodes.generate_answer)
        graph.add_node("web", self.nodes.web_search)

        # entry
        graph.set_entry_point("retrieve")

        # edges
        graph.add_edge("retrieve", "judge")

        # 🔑 CONDITIONAL ROUTING (THIS IS THE KEY)
        graph.add_conditional_edges(
            "judge",
            lambda state: "web" if state.use_web else "doc_answer",
            {
                "web": "web",
                "doc_answer": "doc_answer",
            },
        )

        # terminal edges
        graph.add_edge("web", END)
        graph.add_edge("doc_answer", END)

        return graph.compile()
