Build roadmap (in order):

Rule-based stripping (stopwords, whitespace) — free baseline, no AI yet
Exact/near-duplicate line removal
Perplexity-based token pruning — first real AI component
Semantic deduplication via embeddings
Benchmark dashboard (tokens saved % vs quality retained, via BERTScore) — build this early so every later feature has before/after proof
Query-aware compression
Adaptive compression ratio (light compression for short prompts, aggressive for long ones)
Content-type router (detect code vs prose vs chat history vs structured data, send to matching strategy) — the key differentiator
Adaptive compression ratio tuning
Classifier-based keep/discard (trainable model)
Extreme compression mode (gist tokens) — stretch goal