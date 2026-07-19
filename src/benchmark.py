"""
Step 5: Benchmark dashboard.

Runs the full pipeline (steps 1-3) against realistic sample texts and
reports two numbers side by side for each:
  - tokens saved (%) - the cost/latency win
  - BERTScore F1 - how much original meaning was preserved (0-1, higher = better)

This is the proof-of-work step: every future feature (embeddings, query-aware,
router) gets compared against these same baseline numbers.
"""

from bert_score import score as bert_score

from pipeline import compress


# Realistic samples - NOT crafted to flatter our specific functions.
# These represent actual use cases: a support chat log and meeting notes,
# both of which naturally contain repetition, filler, and redundancy
# the way real user-submitted prompts do.

SAMPLES = {
    "customer_support_chat": """Customer: Hi, I'm having trouble logging into my account.
Customer: I tried resetting my password but I never got the email.
Agent: I'm sorry to hear that. Can you confirm the email address on your account?
Customer: Yes it's the same email I use for everything, john.doe@example.com
Agent: Thank you. I can see your account. It looks like the password reset email might have gone to spam.
Customer: I checked spam, nothing there either.
Customer: I really just need to get into my account today if possible.
Agent: Understood, let me manually trigger a new reset link for you right now.
Agent: I've just sent a new password reset link to john.doe@example.com, please check again.
Customer: Got it, thanks, that one came through.
Customer: I was able to log in now, thank you so much for your help.
Agent: You're welcome! Is there anything else I can help you with today?
Customer: No that's all, thanks again.""",

    "meeting_notes": """Meeting notes from the product sync on Tuesday.

We basically discussed the Q3 roadmap and what we actually want to prioritize
for the next release. The engineering team mentioned that the API migration
is still in progress and should really be done by the end of this month.

Sarah mentioned that the API migration is still in progress and should be
done by the end of this month, and that QA will need at least a week after
that to test everything properly.

Marketing raised concerns about the launch date slipping again, since it
has already slipped twice this quarter. We agreed to basically lock the
date once engineering confirms the migration is fully complete.

Action items: Sarah to confirm migration completion date by Friday.
Marketing to prepare launch materials assuming a mid-month release.
QA to block out one week for full regression testing after migration.""",
}


def run_benchmark():
    print(f"{'Sample':<25} {'Orig Tokens':<12} {'Final Tokens':<13} {'Saved %':<10} {'BERTScore F1':<12}")
    print("-" * 75)

    for name, text in SAMPLES.items():
        result = compress(text)
        original = result["original_text"]
        final = result["final_text"]

        # BERTScore compares meaning preservation between original and compressed.
        # Using distilbert-base-uncased since it's already cached locally from step 3.
        P, R, F1 = bert_score(
            [final], [original],
            model_type="distilbert-base-uncased",
            num_layers=6,
            verbose=False,
        )

        print(f"{name:<25} {result['original_tokens']:<12} {result['final_tokens']:<13} "
              f"{result['total_saved_pct']:<10} {F1.item():<12.4f}")

    print("\nBERTScore F1 ranges 0-1. Above ~0.85 generally indicates strong meaning retention.")


if __name__ == "__main__":
    run_benchmark()