# One claim survived a second question generator; one did not

*A short technical note on chunking for Arabic RAG, and on checking a benchmark against its own generator.*

## Setup

300 Arabic Wikipedia articles; questions written by an LLM with a **verbatim quote** as evidence, so a
chunk counts as correct if it contains the middle six words of the quote. This makes the ground truth
independent of any chunking method. Retrieval uses `BAAI/bge-m3`, cosine over all chunks. Methods:
fixed windows (100/150/200 words, 20% overlap), paragraph chunks, and semantic chunks (sentence
embeddings, split at low-similarity breakpoints). Confidence intervals are bootstrap over questions.

## First result (generator: qwen3.8:27b, 442 questions)

Hit@1 rose with chunk size (0.683 → 0.724 → 0.771 for 100/150/200 words); paragraph chunks (median 211
words) scored 0.792; semantic chunks (median 113) scored 0.715 — exactly where the fixed-window trend
predicts for their size. I summarised this as "chunk size drives quality, not strategy".

## The check

One generator is one set of question habits. I regenerated questions for the same articles with a
different model family (`gemini-3.5-flash-lite`; same prompt, same quote filter; 514 accepted,
11 rejected for non-verbatim quotes), and evaluated with identical code.

Absolute scores fell sharply (paragraph Hit@1 0.792 → 0.619), because the two generators write
questions of different difficulty. That alone is a warning against quoting a single Hit@1 as a property
of a chunker. The informative question is whether *comparisons* survive, so I used a paired bootstrap on
the questions both methods were scored on (comparing overlapping unpaired intervals is far less powerful):

| Comparison | qwen | gemini |
|---|---|---|
| fixed-200 − fixed-100 (MRR) | **+0.060** (+0.033, +0.088) | +0.016 (−0.010, +0.044) |
| paragraph − fixed-200 | +0.010 (−0.014, +0.034) | +0.001 (−0.023, +0.026) |
| semantic − fixed-100 (size-matched) | +0.027 (−0.006, +0.060) | +0.002 (−0.027, +0.032) |

## What survives

- **Semantic ≈ size-matched fixed windows.** No measurable gain from semantic splitting, under both generators.
- **Paragraph ≈ best fixed size.** Equivalent quality, no overlap and no extra embedding pass.

## What does not

- **"Bigger chunks are better."** Significant under one generator, indistinguishable from zero under the other.
  My first write-up stated it as a headline finding; the second generator shows that was premature.

## Why it matters beyond this benchmark

1. A benchmark whose questions come from one LLM can encode that LLM's habits in the ranking of methods.
   A second, independent generator is cheap (about 240k tokens here) and is the most direct test.
2. Use paired comparisons when methods are scored on the same items.
3. Report absolute numbers with their provenance; they do not transfer between question sets.

## Limits

Both generators are LLMs reading the same text with the same prompt, so shared biases (for example,
preferring facts stated in single sentences) are not controlled. Questions were not human-reviewed.
Wikipedia only; one embedding model; retrieval only (no answer-generation quality). Two articles for
the second generator produced malformed JSON after three attempts and were skipped.

Data, code and every number above: this repository (`results*.md`, `05_paired.py`).
