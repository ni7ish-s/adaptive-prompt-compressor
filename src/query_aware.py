"""
Step 6: Query-aware compression.

Scores each line by relevance to a query, then protects only the TOP-N%
most relevant lines (relative ranking) rather than an absolute similarity
threshold. An absolute threshold can accidentally mark almost the entire
document as "relevant enough" on some transcripts (leaving nothing for
later stages to compress) - relative ranking guarantees the pipeline
always keeps room to compress, regardless of document topic/phrasing.
"""

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from stripper import count_tokens

_MODEL_NAME = "all-MiniLM-L6-v2"

_model = None


def _load_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def score_relevance(text: str, query: str) -> list[tuple[str, float]]:
    model = _load_model()
    from segment import split_into_segments
    lines = [line for line in split_into_segments(text) if line.strip() != ""]

    if not lines:
        return []

    query_embedding = model.encode(query, convert_to_tensor=True)
    line_embeddings = model.encode(lines, convert_to_tensor=True)

    scores = []
    for line, emb in zip(lines, line_embeddings):
        relevance = cos_sim(query_embedding, emb).item()
        scores.append((line, relevance))

    return scores


def mark_protected_lines(text: str, query: str, protect_top_pct: float = 0.3) -> set:
    """
    Returns the set of lines in the TOP `protect_top_pct` fraction by
    relevance to the query (e.g. 0.3 = top 30% most relevant lines),
    regardless of their raw similarity score. Guarantees a bounded
    fraction of the document is protected, never all or nearly all of it.
    """
    if not query:
        return set()

    scores = score_relevance(text, query)
    if not scores:
        return set()

    n_protect = max(1, int(len(scores) * protect_top_pct))
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)
    return {line for line, _ in ranked[:n_protect]}

def mark_protected_clauses(text: str, query: str, protect_top_pct: float = 0.3) -> set:
    """
    Splits the document into CLAUSES (finer than whole sentences), scores
    each clause's relevance to the query, and protects only the top-N%
    most relevant clauses - so a sentence can have its relevant clause
    protected while the rest of that same sentence still gets cut.
    """
    from segment import split_into_segments, split_into_clauses

    if not query:
        return set()

    model = _load_model()
    all_clauses = []
    for sentence in split_into_segments(text):
        if sentence.strip():
            all_clauses.extend(split_into_clauses(sentence))

    if not all_clauses:
        return set()

    query_embedding = model.encode(query, convert_to_tensor=True)
    clause_embeddings = model.encode(all_clauses, convert_to_tensor=True)

    scores = [(c, cos_sim(query_embedding, e).item()) for c, e in zip(all_clauses, clause_embeddings)]
    n_protect = max(1, int(len(scores) * protect_top_pct))
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)
    return {clause for clause, _ in ranked[:n_protect]}


if __name__ == "__main__":
    sample = """The customer said their package arrived damaged on Tuesday.
Weather in the region has been unusually rainy this month.
They are requesting a full refund for the damaged item.
Our support team typically responds within 24 hours.
The customer mentioned they've been a member since 2019."""

    query = "What is the customer asking for?"

    protected = mark_protected_lines(sample, query, protect_top_pct=0.3)
    print(f"Query: {query}")
    print(f"Protected lines (top 30%): {len(protected)}")
    for line in protected:
        print(f"  - {line}")