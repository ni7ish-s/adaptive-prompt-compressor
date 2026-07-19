"""
Step 4: Semantic deduplication via embeddings.

Step 2 (dedup.py) only catches lines that share overlapping WORDS.
It misses lines that say the same thing with completely different
vocabulary, e.g.:
    "The report needs to be finished by Friday."
    "We need to complete the document before the end of the week."

This step converts each line into an embedding (a vector representing
its MEANING, not its exact wording) using a small local model, then
compares lines using cosine similarity. Lines above a similarity
threshold are treated as duplicates - same logic as step 2, but
meaning-aware instead of word-aware.

Runs AFTER dedup.py in the pipeline: word-overlap dedup catches the
cheap, obvious cases first; this catches what word overlap missed.
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


def remove_semantic_duplicates(text: str, similarity_threshold: float = 0.48) -> dict:
    """
    Removes lines that are semantically near-identical to an earlier line,
    even if they share little to no literal vocabulary. Keeps the FIRST
    occurrence, drops later semantic duplicates.

    Empty lines are preserved as-is to keep paragraph structure.
    """
    original_tokens = count_tokens(text)
    lines = text.split("\n")

    model = _load_model()

    non_empty_lines = [line for line in lines if line.strip() != ""]
    if not non_empty_lines:
        return {
            "original_text": text,
            "compressed_text": text,
            "original_tokens": original_tokens,
            "compressed_tokens": original_tokens,
            "tokens_saved": 0,
            "tokens_saved_pct": 0.0,
            "lines_removed": 0,
        }

    embeddings = model.encode(non_empty_lines, convert_to_tensor=True)

    kept_lines = []
    kept_embeddings = []

    line_idx = 0
    for line in lines:
        if line.strip() == "":
            kept_lines.append(line)
            continue

        emb = embeddings[line_idx]
        line_idx += 1

        is_dup = False
        for kept_emb in kept_embeddings:
            similarity = cos_sim(emb, kept_emb).item()
            if similarity >= similarity_threshold:
                is_dup = True
                break

        if not is_dup:
            kept_lines.append(line)
            kept_embeddings.append(emb)

    compressed = "\n".join(kept_lines)
    compressed_tokens = count_tokens(compressed)
    saved = original_tokens - compressed_tokens
    saved_pct = (saved / original_tokens * 100) if original_tokens else 0.0

    return {
        "original_text": text,
        "compressed_text": compressed,
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "tokens_saved": saved,
        "tokens_saved_pct": round(saved_pct, 2),
        "lines_removed": len(lines) - len(kept_lines),
    }


if __name__ == "__main__":
    sample = """The report needs to be finished by Friday.
We need to complete the document before the end of the week.
Please make sure the quarterly document is done soon.
The weather today is sunny with a high of 75 degrees.
Can you have that report wrapped up by the end of this week?"""

    result = remove_semantic_duplicates(sample, similarity_threshold=0.48)
    print(f"Original tokens:   {result['original_tokens']}")
    print(f"Compressed tokens: {result['compressed_tokens']}")
    print(f"Saved: {result['tokens_saved']} ({result['tokens_saved_pct']}%)")
    print(f"Lines removed: {result['lines_removed']}")
    print("---")
    print(result["compressed_text"])