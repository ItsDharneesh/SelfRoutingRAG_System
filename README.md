# Adaptive Agentic RAG Engine

> A decision-aware Retrieval-Augmented Generation (RAG) system that prioritizes document-grounded answers and intelligently falls back to web search when required.

---

## Project Overview

This project implements a **Router-based Agentic RAG architecture** using:

- **LangGraph** for orchestration  
- **FAISS** for vector similarity search  
- **OpenAI Embeddings + GPT-4o** for semantic reasoning  
- **Tavily Search API** for live web fallback  
- **Streamlit** for interactive UI  

The system first attempts retrieval from indexed documents.  
If insufficient context is found, a **judge LLM node dynamically routes the query to web search**.

---

## Architecture

```text
User Query
    ↓
Retriever (FAISS)
    ↓
Judge LLM (YES / NO Decision)
    ↓
   ┌───────────────┬────────────────┐
   │ Docs Route    │ Web Route      │
   │ (Context QA)  │ (Tavily API)   │
   └───────────────┴────────────────┘
            ↓
     Final LLM Answer
```

---

## Key Features

- **Hybrid Retrieval Pipeline**
  - URL ingestion via BeautifulSoup
  - PDF ingestion support
  - Semantic chunking

- **Vector Search**
  - FAISS indexing
  - OpenAI embedding generation

- **Decision-Based Routing**
  - LLM-based judge node
  - Dynamic switching between local knowledge and live web search

- **Evaluation Metrics**
  - Mean Reciprocal Rank (**MRR**)
  - Normalized Discounted Cumulative Gain (**nDCG**)
  - Key-Term Coverage
  - Routing Accuracy

- **Interactive UI**
  - Streamlit-based interface
  - Debug logs for retrieval, routing, and web results

---

## Current Evaluation Results

| Metric | Score |
|--------|-------|
| **Mean MRR** | 1.0 |
| **Mean nDCG** | 1.0 |
| **Key-Term Coverage** | 0.43 |
| **Routing Accuracy** | 0.33 |

> Retrieval quality is strong. Routing optimization is ongoing.

---

## Project Structure

```text
RAG_DOC_ENGINE/
│
├── data/
│   ├── attention.pdf
│   └── url.txt
│
├── src/
│   ├── config/
│   ├── doc_ingestion/
│   ├── vectorstore/
│   ├── graph_builder/
│   ├── nodes/
│   └── eval/
│
├── streamlit_app.py
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/your-username/adaptive-agentic-rag.git
cd adaptive-agentic-rag

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
```

---

## Running the App

```bash
streamlit run streamlit_app.py
```

---

## Running Evaluation

```bash
python -m src.eval.run_eval
```

---

## Tech Stack

**Python • LangChain • LangGraph • FAISS • OpenAI • Tavily • BeautifulSoup4 • Streamlit**

---

## Future Improvements

- Improve routing accuracy
- Add answer-grounding metrics (Faithfulness / Exact Match)
- Integrate reranking layer
- Expand evaluation dataset

---

## Author
Built as an advanced Agentic RAG exploration project focusing on dynamic routing, retrieval evaluation, and production-ready orchestration.



