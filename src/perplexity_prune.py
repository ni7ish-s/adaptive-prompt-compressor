"""
Step 3: Perplexity-based token pruning (first real AI component).

Uses a small causal language model (distilgpt2) to score how "surprising"
each word is given its preceding context. Low-surprise (predictable) words
carry little information and are pruned first. High-surprise words are kept.

This is context-dependent, unlike steps 1-2, which is why it needs a model
instead of fixed rules.
"""

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

from stripper import count_tokens

_MODEL_NAME = "distilbert-base-uncased"

_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _model = AutoModelForMaskedLM.from_pretrained(_MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


def _compute_word_scores(text: str) -> list[tuple[str, float]]:
    """
    For each word, mask it and measure how surprised the model is by the
    true word, using FULL bidirectional context (both sides). Higher score
    = model didn't expect this word = more informative = keep it.
    """
    tokenizer, model = _load_model()
    words = text.split(" ")

    word_scores = []
    for i, word in enumerate(words):
        masked_words = words.copy()
        masked_words[i] = tokenizer.mask_token
        masked_text = " ".join(masked_words)

        encoding = tokenizer(masked_text, return_tensors="pt")
        mask_positions = (encoding["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]

        if len(mask_positions) == 0:
            word_scores.append((word, float("inf")))  # couldn't align, keep safe
            continue

        with torch.no_grad():
            logits = model(**encoding).logits

        # If word was split into multiple subword masks, average their surprise
        original_ids = tokenizer(word, add_special_tokens=False)["input_ids"]
        surprises = []
        for j, pos in enumerate(mask_positions):
            log_probs = torch.log_softmax(logits[0, pos], dim=-1)
            true_id = original_ids[j] if j < len(original_ids) else original_ids[-1]
            surprises.append(-log_probs[true_id].item())

        word_scores.append((word, sum(surprises) / len(surprises)))

    return word_scores

def prune_by_perplexity(text: str, keep_ratio: float = 0.6) -> dict:
    """
    Keeps the top `keep_ratio` fraction of words by surprise score,
    in original order. Everything below that cutoff is dropped.
    """
    original_tokens = count_tokens(text)

    word_scores = _compute_word_scores(text)
    n_keep = max(1, int(len(word_scores) * keep_ratio))

    # Rank by score descending (most surprising first) to find the cutoff,
    # but keep final output in ORIGINAL order.
    ranked = sorted(word_scores, key=lambda ws: ws[1], reverse=True)
    keep_set_indices = set()
    scored_with_index = list(enumerate(word_scores))
    ranked_with_index = sorted(scored_with_index, key=lambda x: x[1][1], reverse=True)
    for idx, _ in ranked_with_index[:n_keep]:
        keep_set_indices.add(idx)

    kept_words = [w for i, (w, s) in enumerate(word_scores) if i in keep_set_indices]
    compressed = " ".join(kept_words)

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
        "word_scores": word_scores,  # exposed for debugging/benchmarking later
    }


if __name__ == "__main__":
    sample = (
        "The quarterly financial report shows that Anthropic's revenue "
        "increased significantly due to strong enterprise adoption of "
        "Claude across multiple industries."
    )

    result = prune_by_perplexity(sample, keep_ratio=0.8)
    print(f"Original tokens:   {result['original_tokens']}")
    print(f"Compressed tokens: {result['compressed_tokens']}")
    print(f"Saved: {result['tokens_saved']} ({result['tokens_saved_pct']}%)")
    print("---")
    print(result["compressed_text"])
    print("---")
    print("Per-word surprise scores (higher = kept preferentially):")
    for word, score in result["word_scores"]:
        print(f"  {word:20s} {score:.3f}")