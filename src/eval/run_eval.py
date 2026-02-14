"""
Evaluation script for Router-Based Agentic RAG System

Computes:
- Mean Reciprocal Rank (MRR)
- nDCG
- Key-Term Coverage
- Routing Accuracy
"""

from src.graph_builder.graph_builder import GraphBuilder
from src.eval.metrics import mean_reciprocal_rank, ndcg, key_term_coverage
from src.config.config import Config
from src.doc_ingestion.doc_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore


# ====================================
# Build RAG System
# ====================================

print("Building RAG system...")

llm = Config.get_llm()

doc_processor = DocumentProcessor()
vector_store = VectorStore()

documents = doc_processor.process_urls(Config.DEFAULT_URLS)
vector_store.create_vectorstore(documents)

graph = GraphBuilder(
    retriever=vector_store.get_retriever(),
    llm=llm
).build()


# ====================================
# Evaluation Dataset
# ====================================

eval_data = [
    {
        "question": "What is an AI agent?",
        "gold_answer": "An AI agent is a system that perceives and acts.",
        "gold_source": "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "relevant_sources": [
            "https://lilianweng.github.io/posts/2023-06-23-agent/"
        ],
        "gold_route": "docs"
    },
    {
        "question": "What are diffusion models?",
        "gold_answer": "Diffusion models are generative models that learn by reversing noise.",
        "gold_source": "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
        "relevant_sources": [
            "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/"
        ],
        "gold_route": "docs"
    },
    {
        "question": "Who is Elon Musk?",
        "gold_answer": "Elon Musk is a technology entrepreneur and CEO of SpaceX and Tesla.",
        "gold_source": None,
        "relevant_sources": [],
        "gold_route": "web"
    }
]


# ====================================
# Metric Containers
# ====================================

mrr_scores = []
ndcg_scores = []
coverage_scores = []

correct_routes = 0
total_questions = len(eval_data)


# ====================================
# Evaluation Loop
# ====================================

for sample in eval_data:

    print("\n=================================")
    print("Question:", sample["question"])

    result = graph.invoke({"question": sample["question"]})

    retrieved_docs = result.get("retrieved_docs", [])
    answer = result.get("answer", "")
    used_web = result.get("use_web", False)

    predicted_route = "web" if used_web else "docs"

    print("Predicted Route:", predicted_route)
    print("Gold Route:", sample["gold_route"])

    # ------------------------------
    # Routing Accuracy
    # ------------------------------

    if predicted_route == sample["gold_route"]:
        correct_routes += 1

    # ------------------------------
    # Retrieval Metrics (DOC ONLY)
    # ------------------------------

    if sample["gold_route"] == "docs":

        print("Retrieved Sources:")

        for d in retrieved_docs:
            print(d.metadata.get("source"))

        print("Gold Source:", sample["gold_source"])
        print("------")

        mrr = mean_reciprocal_rank(
            retrieved_docs,
            sample["relevant_sources"]
        )

        ndcg_score = ndcg(
            retrieved_docs,
            sample["relevant_sources"]
        )

        context = "\n".join(
            d.page_content for d in retrieved_docs
        )

        coverage = key_term_coverage(
            answer,
            context
        )

        mrr_scores.append(mrr)
        ndcg_scores.append(ndcg_score)
        coverage_scores.append(coverage)


# ====================================
# Final Metrics
# ====================================

print("\n=================================")

if mrr_scores:
    print("Mean MRR:", sum(mrr_scores) / len(mrr_scores))
    print("Mean nDCG:", sum(ndcg_scores) / len(ndcg_scores))
    print("Mean Key-Term Coverage:", sum(coverage_scores) / len(coverage_scores))
else:
    print("No document-based samples for retrieval metrics.")

print("Routing Accuracy:", correct_routes / total_questions)

print("=================================")
