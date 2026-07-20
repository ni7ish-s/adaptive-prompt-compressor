"""
v1.1 pipeline: runs compression stages in sequence, with optional
query-aware protection.

Step 0 (optional): query relevance scoring - if a query is provided,
        lines highly relevant to it are protected from later cuts.
Step 1: rule-based whitespace/filler stripping
Step 2: exact/near-duplicate line removal (word-overlap based)
Step 3: semantic deduplication via embeddings
Step 4: perplexity-based token pruning (AI component)
"""

from stripper import rule_based_compress, count_tokens
from dedup import remove_duplicate_lines
from perplexity_prune import prune_by_perplexity
from semantic_dedup import remove_semantic_duplicates
from query_aware import mark_protected_lines


def compress(text: str, query: str = None, aggressive_filler: bool = False,
             dedup_threshold: float = 0.3, keep_ratio: float = 0.75,
             semantic_threshold: float = 0.6, relevance_threshold: float = 0.4) -> dict:

    original_tokens = count_tokens(text)

    protected_lines = mark_protected_lines(text, query, relevance_threshold=relevance_threshold) if query else set()

    step1 = rule_based_compress(text, aggressive=aggressive_filler)
    step2 = remove_duplicate_lines(step1["compressed_text"], near_threshold=dedup_threshold)
    step3 = remove_semantic_duplicates(step2["compressed_text"], similarity_threshold=semantic_threshold)
    step4 = prune_by_perplexity(step3["compressed_text"], keep_ratio=keep_ratio, protected_lines=protected_lines)

    final_text = step4["compressed_text"]
    final_tokens = count_tokens(final_text)
    total_saved = original_tokens - final_tokens
    total_saved_pct = (total_saved / original_tokens * 100) if original_tokens else 0.0

    return {
        "original_text": text,
        "final_text": final_text,
        "original_tokens": original_tokens,
        "final_tokens": final_tokens,
        "total_saved": total_saved,
        "total_saved_pct": round(total_saved_pct, 2),
        "protected_lines": protected_lines,
        "step_breakdown": {
            "step1_stripping": step1,
            "step2_word_dedup": step2,
            "step3_semantic_dedup": step3,
            "step4_perplexity": step4,
        },
    }

if __name__ == "__main__":
    sample = """The customer said their package arrived damaged on Tuesday.
Weather in the region has been unusually rainy this month.
They are requesting a full refund for the damaged item.
Our support team typically responds within 24 hours.
The customer mentioned they've been a member since 2019."""

    query = "What is the customer asking for?"

    result = compress(sample, query=query)

    print("=== FINAL RESULT ===")
    print(f"Query: {query}")
    print(f"Original tokens: {result['original_tokens']}")
    print(f"Final tokens:    {result['final_tokens']}")
    print(f"Total saved:     {result['total_saved']} ({result['total_saved_pct']}%)")
    print(f"Protected lines: {result['protected_lines']}")
    print()
    print("Final compressed text:")
    print(result["final_text"])