"""
Step 2: Exact and near-duplicate line removal (baseline, no AI).

Chat histories and pasted context often repeat lines verbatim or with
tiny wording differences. This pass removes those before any model-based
compression runs. Still purely rule/string based - embeddings-based
semantic dedup (meaning-level, not wording-level) comes in step 4.
"""

import re
from difflib import SequenceMatcher

from stripper import count_tokens


_STOPWORDS = {
    "a", "an", "the", "of", "for", "this", "that", "to", "in", "on", "at",
    "is", "are", "was", "were", "be", "can", "you", "me", "i", "please",
    "give", "get", "with", "and", "or", "it", "do", "does", "did",
}


def _simple_stem(word: str) -> str:
    """
    Crude suffix-stripping heuristic (not a real stemmer) so related
    word forms (quarterly/quarter, summarize/summary) count as matches.
    Good enough for rule-based dedup; real semantic matching comes in
    step 4 (embeddings).
    """
    for suffix in ("izing", "ization", "ize", "ies", "ing", "ed", "ly", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _normalize_line(line: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for comparison only."""
    line = re.sub(r"[^\w\s]", "", line.strip().lower())
    return re.sub(r"\s+", " ", line)


def _content_words(norm_line: str) -> set:
    """Extract stemmed, non-stopword words - the actual meaning-bearing signal."""
    words = norm_line.split()
    return {_simple_stem(w) for w in words if w not in _STOPWORDS}


def _similarity(a: str, b: str) -> float:
    """
    Jaccard similarity over stemmed content words (stopwords removed).
    This focuses comparison on meaning-bearing words instead of grammar
    scaffolding, and lets related word forms (quarterly/quarter) match.
    """
    words_a = _content_words(a)
    words_b = _content_words(b)
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def remove_duplicate_lines(text: str, near_threshold: float = 0.9) -> dict:
    """
    Removes exact duplicate lines, and near-duplicate lines above
    `near_threshold` similarity. Keeps the FIRST occurrence of each line,
    drops subsequent duplicates/near-duplicates.

    Empty lines are preserved as-is (not deduped) to keep paragraph structure.
    """
    original_tokens = count_tokens(text)
    
    from segment import split_into_segments
    lines = split_into_segments(text)

    kept_lines = []
    kept_normalized = []  # normalized form of lines we've kept, for comparison

    for line in lines:
        if line.strip() == "":
            kept_lines.append(line)
            continue

        norm = _normalize_line(line)

        # Exact duplicate check (fast path)
        if norm in kept_normalized:
            continue

        # Near-duplicate check against everything kept so far
        is_near_dup = any(
            _similarity(norm, seen) >= near_threshold
            for seen in kept_normalized
        )
        if is_near_dup:
            continue

        kept_lines.append(line)
        kept_normalized.append(norm)

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
    sample = """Please summarize the quarterly report.
The user asked for a summary of the quarterly report.
Please summarize the quarterly report.
Can you give me a summary of the report for this quarter?
The weather today is sunny with a high of 75 degrees.
Please provide a summary of the quarterly report."""

    result = remove_duplicate_lines(sample, near_threshold=0.5)
    print(f"Original tokens:   {result['original_tokens']}")
    print(f"Compressed tokens: {result['compressed_tokens']}")
    print(f"Saved: {result['tokens_saved']} ({result['tokens_saved_pct']}%)")
    print(f"Lines removed: {result['lines_removed']}")
    print("---")
    print(result["compressed_text"])