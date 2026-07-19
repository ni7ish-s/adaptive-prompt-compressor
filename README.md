# LLM Token Compressor

An adaptive compression pipeline that shrinks LLM prompts by routing different content types to the compression strategy that fits them best — instead of applying one fixed method uniformly, the way most existing tools (e.g. LLMLingua) do.

## Current status: v1.0

The pipeline currently runs three compression stages in sequence:

1. **Rule-based stripping** — removes redundant whitespace and (optionally) low-information filler words, without touching meaning.
2. **Duplicate line removal** — detects and removes exact and near-duplicate lines (common in chat logs and repeated context), using word-overlap similarity rather than naive string matching.
3. **Perplexity-based pruning** — the first real AI component. A small local language model (`distilbert-base-uncased`) scores how "surprising" each word is given its context. Predictable, low-information words are pruned; important, high-surprise words are kept. Grammatical connector words (e.g. "due to," "because," "although") are explicitly protected from pruning, since they can be structurally important even when statistically predictable.

Each stage reports its own token savings, and the full pipeline reports a combined before/after result.

The pipeline currently runs three compression stages in sequence:

1. **Rule-based stripping** — removes redundant whitespace and (optionally) low-information filler words, without touching meaning.
2. **Duplicate line removal** — detects and removes exact and near-duplicate lines (common in chat logs and repeated context), using word-overlap similarity rather than naive string matching.
3. **Perplexity-based pruning** — the first real AI component. A small local language model (`distilbert-base-uncased`) scores how "surprising" each word is given its context. Predictable, low-information words are pruned; important, high-surprise words are kept. Grammatical connector words (e.g. "due to," "because," "although") are explicitly protected from pruning, since they can be structurally important even when statistically predictable.

Each stage reports its own token savings, and the full pipeline reports a combined before/after result.

## Benchmarking

A benchmark script evaluates the pipeline on realistic text samples (not toy sentences written to flatter the code) — including a customer support chat log and a set of meeting notes, both of which contain natural repetition and filler the way real prompts do.

Quality retention is measured using BERTScore, which compares the meaning of the compressed text against the original (rather than just checking word overlap). This gives an honest signal on whether compression is saving tokens without silently damaging meaning.

Results are modest but genuine: real-world text compresses less dramatically than short, repetitive examples, but with strong meaning retention. This is treated as a starting baseline, not a final result — every future feature is expected to be measured against it.


## Known limitations (v1.0)

- Perplexity-based pruning can occasionally drop content words that are highly predictable in context (e.g. "report" after "financial ... shows"), even though they carry meaning to a human reader. This is a known limitation of perplexity-based methods in general, not specific to this implementation — it's part of the motivation for eventually moving to a trained classifier (see roadmap).
- Compression is currently applied uniformly regardless of content type (code, structured data, chat, prose). The content-type router is planned but not yet implemented.
- No API-based components are used yet; all current model inference runs locally, at zero cost.

## Tech stack

- Python
- `tiktoken` — token counting
- HuggingFace `transformers` — local perplexity scoring model
- `bert-score` — quality-retention evaluation
