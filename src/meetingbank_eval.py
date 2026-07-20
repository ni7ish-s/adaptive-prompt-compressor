"""
Large-scale evaluation on real MeetingBank transcripts - the same
dataset LLMLingua-2 uses for benchmarking. This tells us how the
pipeline actually performs on real, long, naturally redundant text,
not hand-written toy samples.

Uses microsoft/MeetingBank-QA-Summary (862 real test-set transcripts
from U.S. city council meetings, with GPT-4-generated summaries).
"""

import time
from datasets import load_dataset
from bert_score import score as bert_score

from pipeline import compress
from stripper import count_tokens


NUM_TRANSCRIPTS = 5  # how many real meetings to run through the pipeline


def run_large_scale_eval():
    print(f"Loading MeetingBank dataset...")
    dataset = load_dataset("microsoft/MeetingBank-QA-Summary", split="test")

    results = []
    total_start = time.time()

    for i, sample in enumerate(dataset):
        if i >= NUM_TRANSCRIPTS:
            break

        transcript = sample["prompt"]
        original_tokens = count_tokens(transcript)

        start = time.time()
        result = compress(transcript)
        elapsed = time.time() - start

        final = result["final_text"]

        P, R, F1 = bert_score(
            [final], [transcript],
            model_type="distilbert-base-uncased",
            num_layers=6,
            verbose=False,
        )
        f1 = F1.item()

        results.append({
            "idx": i,
            "original_tokens": result["original_tokens"],
            "final_tokens": result["final_tokens"],
            "saved_pct": result["total_saved_pct"],
            "f1": f1,
            "seconds": elapsed,
        })

        print(f"[{i+1}/{NUM_TRANSCRIPTS}] tokens: {result['original_tokens']} -> "
              f"{result['final_tokens']} ({result['total_saved_pct']}% saved) "
              f"| F1: {f1:.4f} | {elapsed:.1f}s")

    total_elapsed = time.time() - total_start

    avg_saved = sum(r["saved_pct"] for r in results) / len(results)
    avg_f1 = sum(r["f1"] for r in results) / len(results)
    avg_seconds = sum(r["seconds"] for r in results) / len(results)
    total_original_tokens = sum(r["original_tokens"] for r in results)
    total_final_tokens = sum(r["final_tokens"] for r in results)

    print("\n=== SUMMARY ACROSS", len(results), "REAL MEETINGBANK TRANSCRIPTS ===")
    print(f"Average tokens saved:     {avg_saved:.2f}%")
    print(f"Average BERTScore F1:     {avg_f1:.4f}")
    print(f"Average time per doc:     {avg_seconds:.1f}s")
    print(f"Total wall-clock time:    {total_elapsed:.1f}s")
    print(f"Total original tokens:    {total_original_tokens}")
    print(f"Total final tokens:       {total_final_tokens}")
    print(f"Overall reduction:        {(1 - total_final_tokens/total_original_tokens) * 100:.2f}%")


if __name__ == "__main__":
    run_large_scale_eval()