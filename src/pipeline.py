"""
v1.0 pipeline: runs all three compression steps in sequence.

Step 1: rule-based whitespace/filler stripping
Step 2: duplicate/near-duplicate line removal
Step 3: perplexity-based token pruning (AI component)
"""

from stripper import rule_based_compress, count_tokens
from dedup import remove_duplicate_lines
from perplexity_prune import prune_by_perplexity


def compress(text: str, aggressive_filler: bool = False,
             dedup_threshold: float = 0.4, keep_ratio: float = 0.8) -> dict:

    original_tokens = count_tokens(text)

    step1 = rule_based_compress(text, aggressive=aggressive_filler)
    step2 = remove_duplicate_lines(step1["compressed_text"], near_threshold=dedup_threshold)
    step3 = prune_by_perplexity(step2["compressed_text"], keep_ratio=keep_ratio)

    final_text = step3["compressed_text"]
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
        "step_breakdown": {
            "step1_stripping": step1,
            "step2_dedup": step2,
            "step3_perplexity": step3,
        },
    }


if __name__ == "__main__":
    sample = """Please summarize the quarterly report.
The user asked for a summary of the quarterly report.
Please summarize the quarterly report.
Can you give me a summary of the report for this quarter?
The weather today is sunny with a high of 75 degrees.
Please provide a summary of the quarterly report."""

    result = compress(sample)

    print("=== FINAL RESULT ===")
    print(f"Original tokens: {result['original_tokens']}")
    print(f"Final tokens:    {result['final_tokens']}")
    print(f"Total saved:     {result['total_saved']} ({result['total_saved_pct']}%)")
    print()
    print("Final compressed text:")
    print(result["final_text"])
    print()
    print("=== STEP BY STEP ===")
    for name, step in result["step_breakdown"].items():
        print(f"{name}: {step['original_tokens']} -> {step['compressed_tokens']} tokens "
              f"({step['tokens_saved_pct']}% saved at this step)")