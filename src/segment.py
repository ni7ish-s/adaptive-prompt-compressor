"""
Shared text-segmentation utility.

Splitting on newlines only works when input text actually has line
breaks (chat logs, hand-written notes). Real documents - like MeetingBank
transcripts - are often one continuous block of text with no line breaks
at all, which silently breaks every line-based stage (dedup, semantic
dedup, query relevance) by treating the entire document as "one line".

This splits into SENTENCES instead, which works correctly regardless of
whether the input has newlines or not.
"""

import re


def split_into_segments(text: str) -> list[str]:
    """
    Splits text into sentence-level segments. Falls back sensibly on
    text with no clear sentence punctuation (treats the whole thing as
    one segment rather than crashing or returning nothing).
    """
    # First split on existing newlines (preserves paragraph structure
    # where it exists), then further split each chunk into sentences.
    segments = []
    for chunk in text.split("\n"):
        if chunk.strip() == "":
            segments.append(chunk)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", chunk.strip())
        segments.extend(s for s in sentences if s.strip())
    return segments if segments else [text]


def split_into_clauses(sentence: str) -> list[str]:
    """Splits a sentence into clauses on commas/semicolons, keeping the
    punctuation attached to the preceding clause."""
    parts = re.split(r"(?<=[,;])\s+", sentence.strip())
    return [p for p in parts if p.strip()]