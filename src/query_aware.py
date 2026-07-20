"""
Step 6: Query-aware compression.

So far, every stage compresses text the same way regardless of WHY it's
being sent to an LLM. This stage adds an optional query: if the caller
knows what question the compressed text will be used to answer, we score
each line by its relevance to that query and protect highly relevant
lines from being cut by later stages - even if they'd otherwise look
"prunable" (predictable, duplicate-looking, etc).

Uses the same embedding model as semantic_dedup.py (all-MiniLM-L6-v2),
just comparing each line against a QUERY instead of against other lines.

Runs FIRST in the pipeline (before dedup/perplexity), since relevance
to the query should influence whether later stages are even allowed to
touch a line.
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
    """
    Returns (line, relevance_score) for every non-empty line in `text`,
    where relevance_score is cosine similarity to the query (0-1 range,
    can be slightly negative for very unrelated content).
    """
    model = _load_model()
    lines = [line for line in text.split("\n") if line.strip() != ""]

    if not lines:
        return []

    query_embedding = model.encode(query, convert_to_tensor=True)
    line_embeddings = model.encode(lines, convert_to_tensor=True)

    scores = []
    for line, emb in zip(lines, line_embeddings):
        relevance = cos_sim(query_embedding, emb).item()
        scores.append((line, relevance))

    return scores


def mark_protected_lines(text: str, query: str, relevance_threshold: float = 0.4) -> set:
    """
    Returns a set of lines (exact string match) that are relevant enough
    to the query that later pipeline stages should not remove them,
    regardless of what their own importance/duplicate scoring says.
    """
    if not query:
        return set()

    scores = score_relevance(text, query)
    return {line for line, score in scores if score >= relevance_threshold}


if __name__ == "__main__":
    sample = """The customer said their package arrived damaged on Tuesday.
Weather in the region has been unusually rainy this month.
They are requesting a full refund for the damaged item.
Our support team typically responds within 24 hours.
The customer mentioned they've been a member since 2019."""

    query = "What is the customer asking for?"

    scores = score_relevance(sample, query)
    print(f"Query: {query}\n")
    print("Relevance scores per line:")
    for line, score in scores:
        print(f"  {score:.3f}  {line}")

    protected = mark_protected_lines(sample, query, relevance_threshold=0.4)
    print(f"\nProtected lines (relevance >= 0.4): {len(protected)}")
    for line in protected:
        print(f"  - {line}")