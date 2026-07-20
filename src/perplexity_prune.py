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

# Structural/connector words that hold sentence grammar together.
# These are often "low surprise" to the model (very predictable), but
# removing them breaks grammar even though no real content is lost.
# We protect them from pruning regardless of their perplexity score.
_PROTECTED_WORDS = {
    "to", "due", "because", "since", "although", "though", "unless",
    "while", "whereas", "if", "then", "so", "but", "however", "not",
    "no", "never", "against", "without",
}
# Unit/quantity words that are meaningless without their adjacent number
# (e.g. "24 hours" - the model finds "hours" predictable and prunes it,
# leaving an orphaned, meaningless number behind).
_UNIT_WORDS = {
    "hour", "hours", "minute", "minutes", "second", "seconds",
    "day", "days", "week", "weeks", "month", "months", "year", "years",
    "dollar", "dollars", "percent", "%", "times", "x",
}

_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _model = AutoModelForMaskedLM.from_pretrained(_MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


_MAX_WORDS_PER_CHUNK = 300  # conservative - well under 512 sub-word tokens
                             # even for words that split into multiple sub-tokens


def _compute_word_scores(text: str) -> list[tuple[str, float]]:
    """
    Scores every word in `text` for surprise, using masked-LM prediction.
    Internally splits very long lines into smaller word-chunks to stay
    under the model's 512-token limit (distilbert-base-uncased), since
    real documents (e.g. meeting transcripts) can have single "lines"
    far longer than any hand-written test sample we used earlier.
    """
    words = text.split(" ")

    if len(words) <= _MAX_WORDS_PER_CHUNK:
        return _compute_word_scores_chunk(text)

    # Split into word-count chunks, score each independently, concatenate.
    all_scores = []
    for start in range(0, len(words), _MAX_WORDS_PER_CHUNK):
        chunk_words = words[start:start + _MAX_WORDS_PER_CHUNK]
        chunk_text = " ".join(chunk_words)
        all_scores.extend(_compute_word_scores_chunk(chunk_text))

    return all_scores


def _compute_word_scores_chunk(text: str) -> list[tuple[str, float]]:
    """
    The original per-chunk scoring logic (previously named
    _compute_word_scores) - guaranteed to be called only on text short
    enough to fit within the model's token limit.
    """
    tokenizer, model = _load_model()
    words = text.split(" ")

    word_scores = []
    for i, word in enumerate(words):
        masked_words = words.copy()
        masked_words[i] = tokenizer.mask_token
        masked_text = " ".join(masked_words)

        encoding = tokenizer(masked_text, return_tensors="pt", truncation=True, max_length=512)
        mask_positions = (encoding["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]

        if len(mask_positions) == 0:
            word_scores.append((word, float("inf")))
            continue

        with torch.no_grad():
            logits = model(**encoding).logits

        original_ids = tokenizer(word, add_special_tokens=False)["input_ids"]
        surprises = []
        for j, pos in enumerate(mask_positions):
            log_probs = torch.log_softmax(logits[0, pos], dim=-1)
            true_id = original_ids[j] if j < len(original_ids) else original_ids[-1]
            surprises.append(-log_probs[true_id].item())

        avg_surprise = sum(surprises) / len(surprises)

        clean_word = word.strip(".,!?;:").lower()
        is_numeric = any(char.isdigit() for char in clean_word)

        if clean_word in _PROTECTED_WORDS or clean_word in _UNIT_WORDS or is_numeric:
            avg_surprise = float("inf")

        word_scores.append((word, avg_surprise))

    return word_scores

def prune_by_perplexity(text: str, keep_ratio: float = 0.85, protected_lines: set = None) -> dict:
    """
    Keeps the top `keep_ratio` fraction of words by surprise score,
    PER LINE (not across the whole text), so line structure is preserved
    and words from different lines never get merged together.

    Lines in `protected_lines` (exact string match) are kept fully intact -
    no words pruned - since they were already marked as essential by
    query-aware relevance scoring.
    """
    protected_lines = protected_lines or set()
    original_tokens = count_tokens(text)

    lines = text.split("\n")
    output_lines = []
    all_word_scores = []  # kept for debugging/inspection across all lines

    for line in lines:
        if line.strip() == "":
            output_lines.append(line)
            continue

        if line in protected_lines:
            # Fully protected - skip pruning, keep every word
            output_lines.append(line)
            continue

        word_scores = _compute_word_scores(line)
        all_word_scores.extend(word_scores)

        n_keep = max(1, int(len(word_scores) * keep_ratio))
        scored_with_index = list(enumerate(word_scores))
        ranked_with_index = sorted(scored_with_index, key=lambda x: x[1][1], reverse=True)
        keep_set_indices = {idx for idx, _ in ranked_with_index[:n_keep]}

        kept_words = [w for i, (w, s) in enumerate(word_scores) if i in keep_set_indices]
        output_lines.append(" ".join(kept_words))

    compressed = "\n".join(output_lines)
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
        "word_scores": all_word_scores,
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