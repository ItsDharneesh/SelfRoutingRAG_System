import math
import re


# -----------------------------
# MRR
# -----------------------------
def mean_reciprocal_rank(retrieved_docs, relevant_sources):
    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc.metadata.get("source") in relevant_sources:
            return 1.0 / rank
    return 0.0


# -----------------------------
# nDCG
# -----------------------------
def dcg(relevances):
    return sum(
        rel / math.log2(i + 2)
        for i, rel in enumerate(relevances)
    )


def ndcg(retrieved_docs, relevant_sources):
    relevances = [
        1 if doc.metadata.get("source") in relevant_sources else 0
        for doc in retrieved_docs
    ]

    ideal = sorted(relevances, reverse=True)
    ideal_score = dcg(ideal)

    if ideal_score == 0:
        return 0.0

    return dcg(relevances) / ideal_score


# -----------------------------
# Key-Term Coverage
# -----------------------------
def extract_keywords(text, top_k=15):
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    freq = {}

    for w in words:
        freq[w] = freq.get(w, 0) + 1

    sorted_words = sorted(freq, key=freq.get, reverse=True)
    return set(sorted_words[:top_k])


def key_term_coverage(answer, context):
    context_terms = extract_keywords(context)
    answer_terms = set(
        re.findall(r"\b[a-zA-Z]{4,}\b", answer.lower())
    )

    if not context_terms:
        return 0.0

    overlap = context_terms & answer_terms
    return len(overlap) / len(context_terms)
