# Qualitative Case Study: Financial-Option Valuation

## Question

What are the methods and principles for valuing financial options?

Query ID: q_0c244d4030fd08ec

## Experimental setup

- Retrieval depth: K = 20
- Generation context sizes: N = 3, 5, 10, 20
- Identical retrieval ranking across conditions
- Generator: EuroLLM-22B-Instruct
- Evaluator: GPT-4.1
- One generation and evaluation per condition

## Observed results

| Context N | Normalized judge score |
|---|---:|
| 3 | 0.50 |
| 5 | 0.50 |
| 10 | 0.50 |
| 20 | 1.00 |

## Evidence analysis

The document at retrieval rank 11
(record_id: r_939ac4e1656a98cf)
contains a detailed explanation of the Binomial Option
Pricing Model (BOPM).

It describes:
- Dividing time to expiry into sub-periods
- Upward and downward movements of the underlying price
- Backward induction from expiry to the present
- A worked example of single-period option pricing

This document is excluded when N <= 10 and included
when N = 20.

## Generated-answer analysis

At N = 3, the generator focused on intrinsic value and
general valuation concepts.

At N = 5 and N = 10, it introduced some relevant option
pricing methods but also included unrelated material,
such as company valuation and value investing.

At N = 20, the answer provided more detailed explanations
of binomial pricing, risk-neutral valuation, hedging,
dividends, and volatility.

The automated judge assigned N = 20 a normalized score
of 1.00, compared with 0.50 for the other conditions.

## Interpretation

This case suggests that increasing the generation context
can improve answer completeness when relevant evidence
appears below the original context cutoff.

The rank-11 BOPM document provides a plausible explanation
for the improved N = 20 answer.

However, this observational comparison does not establish
that the rank-11 document alone caused the improvement.
Other added documents and generation variability may
also contribute.

## Implications

A relevance reranking stage could potentially promote
highly relevant documents into a smaller context window,
improving the quality-cost trade-off.

This remains a proposed improvement, not a tested result.

## Limitations

- Single query case study
- One generation per context condition
- Automated judge scores are not human ground truth
- No controlled document-ablation experiment
- No evidence that the same pattern generalizes to other queries
