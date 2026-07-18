"""
Step 1: Rule-based stripping (baseline, no AI).

This is the free, deterministic compression pass that runs before any
model-based steps. It should never change meaning - only remove things
that are unambiguously wasteful:
  - extra whitespace / blank lines
  - trailing whitespace
  - (optionally) filler stopwords, applied conservatively

We measure everything using tiktoken so we have a real "tokens saved %"
number from day one, per the roadmap (benchmark early).
"""

import re
import tiktoken

# Use cl100k_base - the encoding used by GPT-3.5/4 family, a reasonable
# general-purpose proxy for "how many tokens is this".
_ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def strip_whitespace(text: str) -> str:
    """Collapse redundant whitespace without touching meaning."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_FILLER_PATTERNS = [
    r"\bjust\b",
    r"\bactually\b",
    r"\bbasically\b",
    r"\breally\b",
    r"\bvery\b",
    r"\bkind of\b",
    r"\bsort of\b",
    r"\bin order to\b",
]


def strip_filler_words(text: str, aggressive: bool = False) -> str:
    if not aggressive:
        return text

    result = text
    for pattern in _FILLER_PATTERNS:
        if pattern == r"\bin order to\b":
            result = re.sub(pattern, "to", result, flags=re.IGNORECASE)
        else:
            result = re.sub(pattern, "", result, flags=re.IGNORECASE)

    result = re.sub(r"[ \t]{2,}", " ", result)
    result = re.sub(r" \n", "\n", result)
    return result.strip()


def rule_based_compress(text: str, aggressive: bool = False) -> dict:
    original_tokens = count_tokens(text)

    compressed = strip_whitespace(text)
    compressed = strip_filler_words(compressed, aggressive=aggressive)

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
    }


if __name__ == "__main__":
    sample = """
    Hey, I just wanted to actually ask you something.   Can you basically
    help me in order to  understand this really  well?


    I sort of need this  done kind of quickly.
    """

    result = rule_based_compress(sample, aggressive=False)
    print("--- Conservative (whitespace only) ---")
    print(f"Original tokens:   {result['original_tokens']}")
    print(f"Compressed tokens: {result['compressed_tokens']}")
    print(f"Saved: {result['tokens_saved']} ({result['tokens_saved_pct']}%)")
    print(repr(result["compressed_text"]))

    result2 = rule_based_compress(sample, aggressive=True)
    print("\n--- Aggressive (whitespace + filler words) ---")
    print(f"Original tokens:   {result2['original_tokens']}")
    print(f"Compressed tokens: {result2['compressed_tokens']}")
    print(f"Saved: {result2['tokens_saved']} ({result2['tokens_saved_pct']}%)")
    print(repr(result2["compressed_text"]))