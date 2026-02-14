"""RAG state definition for router-based Agentic RAG"""

from typing import List, Optional
from pydantic import BaseModel
from langchain_core.documents import Document


class RAGState(BaseModel):
    # core
    question: str
    retrieved_docs: List[Document] = []
    answer: str = ""
    use_web: bool = False

    # 🔍 debug / observability
    debug_retrieved_count: int = 0
    debug_judge_decision: Optional[str] = None
    debug_web_raw: Optional[str] = None
    debug_web_context: Optional[str] = None
