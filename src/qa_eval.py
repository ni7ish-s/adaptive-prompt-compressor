"""
Real output-quality evaluation, matching LLMLingua's actual methodology:
compress a transcript, feed the COMPRESSED version + a real question to
an LLM (Groq), and check if the answer still matches the ground truth -
instead of measuring text similarity (BERTScore) as a proxy.

Uses microsoft/MeetingBank-QA-Summary, which includes real GPT-4-generated
QA pairs per transcript - the same dataset LLMLingua-2 was evaluated on.
"""

import os
import time
import tiktoken
from datasets import load_dataset
from dotenv import load_dotenv
from groq import Groq

from pipeline import compress
from stripper import count_tokens

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

NUM_TRANSCRIPTS = 15
MODEL = "llama-3.1-8b-instant"
DELAY_SECONDS = 2.5  # stay safely under free-tier rate limits

_encoder = tiktoken.get_encoding("cl100k_base")
MAX_CONTEXT_TOKENS = 4000  # conservative budget, leaves room for question +
                            # system prompt + response, safely under 6000 TPM


def truncate_to_token_limit(text: str, max_tokens: int) -> str:
    tokens = _encoder.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return _encoder.decode(truncated_tokens)


def ask_llm(context: str, question: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Answer the question using only the provided meeting transcript. Be concise - a short phrase or sentence is enough."},
            {"role": "user", "content": f"Transcript:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0,
        max_tokens=100,
    )
    return response.choices[0].message.content.strip()


def judge_correctness(question: str, ground_truth: str, model_answer: str) -> bool:
    """Uses the LLM itself as a judge: does model_answer convey the same
    information as ground_truth, given the question? This avoids brittle
    exact-string matching, since phrasing will differ."""
    judge_prompt = (
        f"Question: {question}\n"
        f"Reference answer: {ground_truth}\n"
        f"Given answer: {model_answer}\n\n"
        "Does the given answer convey the same key information as the reference answer? "
        "Reply with only YES or NO."
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0,
        max_tokens=5,
    )
    verdict = response.choices[0].message.content.strip().upper()
    return verdict.startswith("YES")


def run_qa_eval():
    print("Loading MeetingBank QA dataset...")
    dataset = load_dataset("microsoft/MeetingBank-QA-Summary", split="test")

    results = []

    for i, sample in enumerate(dataset):
        if i >= NUM_TRANSCRIPTS:
            break

        transcript = sample["prompt"]
        qa_pairs = sample["QA_pairs"]
        if not qa_pairs:
            continue

        qa = qa_pairs[0]  # one question per transcript, keeps this fast
        question = qa["question"]
        ground_truth = qa["answer"]

        original_tokens = count_tokens(transcript)
        result = compress(transcript, query=question)
        compressed_text = result["final_text"]
        compressed_text = truncate_to_token_limit(compressed_text, MAX_CONTEXT_TOKENS)
        final_tokens = count_tokens(compressed_text)

        try:
            time.sleep(DELAY_SECONDS)
            model_answer = ask_llm(compressed_text, question)

            time.sleep(DELAY_SECONDS)
            is_correct = judge_correctness(question, ground_truth, model_answer)
            if not is_correct:
                print(f"    >>> FAILED <<<")
        except Exception as e:
            print(f"[{i+1}/{NUM_TRANSCRIPTS}] SKIPPED due to error: {e}")
            continue

        saved_pct = result["total_saved_pct"]
        results.append({"saved_pct": saved_pct, "correct": is_correct})

        print(f"[{i+1}/{NUM_TRANSCRIPTS}] tokens: {original_tokens} -> {final_tokens} "
              f"({saved_pct}% saved) | correct: {is_correct}")
        print(f"    Q: {question}")
        print(f"    Ground truth: {ground_truth}")
        print(f"    Model answer: {model_answer}")
        print()

    if not results:
        print("No transcripts were successfully evaluated.")
        return

    accuracy = sum(r["correct"] for r in results) / len(results) * 100
    avg_saved = sum(r["saved_pct"] for r in results) / len(results)

    print("=== SUMMARY ===")
    print(f"Transcripts evaluated: {len(results)}")
    print(f"Average tokens saved:  {avg_saved:.2f}%")
    print(f"Answer accuracy:       {accuracy:.2f}%")


if __name__ == "__main__":
    run_qa_eval()