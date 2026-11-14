# Optimize report: dolly-summarization-qwen2.5-0.5b

Status: final. Label: benchmark.

**Conclusion (test data): not supported.** On test data, 'random@110000' beat the best model-aware arm beyond noise: gradient scoring did not improve the quality/compute tradeoff here.

## 1. Estimated training-value signals

Estimated signals, not measurements. A score estimates how one update from a record aligns with the gradient of the development loss. It is not a probability, not a task metric, and a negative value is an estimated conflict signal rather than proof that a record is wrong or that deleting it will improve training.

Method: LESS-inspired gradient alignment (Adam-aware update direction, countsketch projection to 8192 dims, 2 checkpoint(s), cosine normalization, lr checkpoint weights). Records scored: 5529. Target alignment: min -0.07039, median 0.02673, max 0.1322; 307 negative.
Retention alignment (kept separate): min -0.05743, median 0.03749; 265 negative.

## 2. Measured development results

Measured on the comparison half of the development split. These results chose the candidate, so they are optimistic for it; the held-out test results are the check.

| arm | comparison | seeds | loss mean | across-seed std | min..max | eval 95% CI |
|---|---|---|---|---|---|---|
| base model (no training) | reference | 1 | 1.528 | n/a | 1.528..1.528 | [1.287, 1.778] |
| diversity@110000 | matched_tokens | 3 | 1.347 | 0.003665 | 1.342..1.349 | [1.141, 1.544] |
| diversity@55000 | matched_tokens | 3 | 1.342 | 0.00193 | 1.34..1.344 | [1.139, 1.54] |
| less@110000 | matched_tokens | 3 | 1.346 | 0.005822 | 1.339..1.35 | [1.132, 1.552] |
| less@110000~diversity_lambda=0.2 | matched_tokens | 3 | 1.349 | 0.002078 | 1.347..1.351 | [1.136, 1.554] |
| less@55000 | matched_tokens | 3 | 1.362 | 0.001436 | 1.36..1.363 | [1.148, 1.569] |
| random+overhead | matched_compute | 3 | 1.303 | 0.003838 | 1.299..1.306 | [1.098, 1.501] |
| random@110000 | matched_tokens | 3 | 1.317 | 0.004226 | 1.314..1.321 | [1.112, 1.517] |
| random@55000 | matched_tokens | 3 | 1.318 | 0.007194 | 1.313..1.326 | [1.115, 1.52] |

Candidate: `random@110000`

- Does training on the candidate subset beat not training at all? (`random@110000` vs `base model`): **supported**. Difference 0.2119 (positive favors the candidate); across-seed 95% interval [0.2014, 0.2224]; evaluation 95% interval [0.1177, 0.3383]. The difference exceeds both run-to-run variation and evaluation uncertainty at the 95% level.
- At the same number of training tokens, does the candidate beat the strongest other policy? (`random@110000` vs `less@110000`): **supported**. Difference 0.02898 (positive favors the candidate); across-seed 95% interval [0.01577, 0.0422]; evaluation 95% interval [0.009969, 0.04606]. The difference exceeds both run-to-run variation and evaluation uncertainty at the 95% level.

## 3. Final held-out results

Measured once on the test split after the candidate and recipe were frozen. Nothing was selected or tuned using these numbers.

| arm | comparison | seeds | loss mean | across-seed std | min..max | eval 95% CI |
|---|---|---|---|---|---|---|
| base model (no training) | reference | 1 | 1.427 | n/a | 1.427..1.427 | [1.335, 1.521] |
| diversity@110000 | matched_tokens | 3 | 1.246 | 0.001698 | 1.245..1.248 | [1.16, 1.332] |
| diversity@55000 | matched_tokens | 3 | 1.245 | 0.0006372 | 1.244..1.245 | [1.159, 1.329] |
| less@110000 | matched_tokens | 3 | 1.239 | 0.005039 | 1.235..1.245 | [1.15, 1.326] |
| less@110000~diversity_lambda=0.2 | matched_tokens | 3 | 1.245 | 0.002003 | 1.243..1.247 | [1.157, 1.331] |
| less@55000 | matched_tokens | 3 | 1.254 | 0.002586 | 1.251..1.255 | [1.165, 1.341] |
| random+overhead | matched_compute | 3 | 1.208 | 0.001553 | 1.206..1.209 | [1.121, 1.294] |
| random@110000 | matched_tokens | 3 | 1.216 | 0.0006736 | 1.216..1.217 | [1.131, 1.302] |
| random@55000 | matched_tokens | 3 | 1.216 | 0.009462 | 1.208..1.226 | [1.132, 1.301] |

Candidate: `random@110000`

- Does training on the candidate subset beat not training at all? (`random@110000` vs `base model`): **supported**. Difference 0.211 (positive favors the candidate); across-seed 95% interval [0.2094, 0.2127]; evaluation 95% interval [0.1836, 0.2417]. The difference exceeds both run-to-run variation and evaluation uncertainty at the 95% level.
- At the same number of training tokens, does the candidate beat the strongest other policy? (`random@110000` vs `less@110000`): **supported**. Difference 0.02249 (positive favors the candidate); across-seed 95% interval [0.009865, 0.03512]; evaluation 95% interval [0.01397, 0.03113]. The difference exceeds both run-to-run variation and evaluation uncertainty at the 95% level.

## 4. Compute consumed

Unit: forward-token-equivalents (1 per forward token, +2 per backward token); an accounting model, not measured FLOPs.

Total 44,875,785 of 250,000,000 budgeted (19,834.4 s wall clock). Optimization overhead (warmup, scoring, selection): 5,535,360 (12.3% of the total).

| stage | FTE | of which from cache | wall seconds |
|---|---|---|---|
| warmup | 238,968 | 0 | 113.4 |
| scoring | 5,296,392 | 0 | 1,727.7 |
| selection | 0 | 0 | 5.1 |
| pilot_training | 35,909,100 | 0 | 15,310.7 |
| evaluation | 478,375 | 0 | 435.8 |
| final_evaluation | 2,952,950 | 0 | 2,241.7 |

## 5. Coverage changes

`diversity@110000`: 1577 records, 109988 tokens.
- minimum for category=open_qa: selected 675 records / 39702 tokens (pool 1398 / 141796).
- token share by category (pool -> selected): brainstorming 9.9% -> 18.7%, classification 9.9% -> 35.6%, closed_qa 19.6% -> 1.1%, creative_writing 6.5% -> 1.2%, general_qa 16.3% -> 6.2%, information_extraction 16.4% -> 0.6%, open_qa 16.4% -> 36.1%, summarization 5.1% -> 0.4%
`diversity@55000`: 897 records, 54999 tokens.
- minimum for category=open_qa: selected 467 records / 25295 tokens (pool 1398 / 141796).
- token share by category (pool -> selected): brainstorming 9.9% -> 18.0%, classification 9.9% -> 29.6%, closed_qa 19.6% -> 0.8%, creative_writing 6.5% -> 1.2%, general_qa 16.3% -> 4.5%, information_extraction 16.4% -> 0.0%, open_qa 16.4% -> 46.0%, summarization 5.1% -> 0.0%
`less@110000`: 569 records, 109998 tokens.
- minimum for category=open_qa: selected 100 records / 7316 tokens (pool 1398 / 141796).
- token share by category (pool -> selected): brainstorming 9.9% -> 1.6%, classification 9.9% -> 2.3%, closed_qa 19.6% -> 46.0%, creative_writing 6.5% -> 1.0%, general_qa 16.3% -> 1.6%, information_extraction 16.4% -> 28.4%, open_qa 16.4% -> 6.7%, summarization 5.1% -> 12.4%
`less@110000~diversity_lambda=0.2`: 723 records, 109978 tokens.
- minimum for category=open_qa: selected 167 records / 9705 tokens (pool 1398 / 141796).
- token share by category (pool -> selected): brainstorming 9.9% -> 3.8%, classification 9.9% -> 6.5%, closed_qa 19.6% -> 41.0%, creative_writing 6.5% -> 0.4%, general_qa 16.3% -> 2.5%, information_extraction 16.4% -> 31.0%, open_qa 16.4% -> 8.8%, summarization 5.1% -> 6.0%
`less@55000`: 313 records, 54987 tokens.
- minimum for category=open_qa: selected 100 records / 7316 tokens (pool 1398 / 141796).
- token share by category (pool -> selected): brainstorming 9.9% -> 0.3%, classification 9.9% -> 0.6%, closed_qa 19.6% -> 48.2%, creative_writing 6.5% -> 0.0%, general_qa 16.3% -> 0.1%, information_extraction 16.4% -> 27.7%, open_qa 16.4% -> 13.3%, summarization 5.1% -> 9.8%
`random+overhead`: 4639 records, 725026 tokens.
- minimum for category=open_qa: selected 1179 records / 118437 tokens (pool 1398 / 141796).
- token share by category (pool -> selected): brainstorming 9.9% -> 10.1%, classification 9.9% -> 9.8%, closed_qa 19.6% -> 19.3%, creative_writing 6.5% -> 6.6%, general_qa 16.3% -> 16.0%, information_extraction 16.4% -> 16.9%, open_qa 16.4% -> 16.3%, summarization 5.1% -> 5.0%
`random@110000`: 704 records, 109996 tokens.
- minimum for category=open_qa: selected 171 records / 16514 tokens (pool 1398 / 141796).
- token share by category (pool -> selected): brainstorming 9.9% -> 10.7%, classification 9.9% -> 10.7%, closed_qa 19.6% -> 20.3%, creative_writing 6.5% -> 5.4%, general_qa 16.3% -> 16.4%, information_extraction 16.4% -> 16.2%, open_qa 16.4% -> 15.0%, summarization 5.1% -> 5.4%
`random@55000`: 375 records, 54984 tokens.
- minimum for category=open_qa: selected 100 records / 9945 tokens (pool 1398 / 141796).
- token share by category (pool -> selected): brainstorming 9.9% -> 12.2%, classification 9.9% -> 11.4%, closed_qa 19.6% -> 20.4%, creative_writing 6.5% -> 4.3%, general_qa 16.3% -> 16.0%, information_extraction 16.4% -> 13.7%, open_qa 16.4% -> 18.1%, summarization 5.1% -> 3.9%

Split overlap: 0 exact and 4 uncertain finding(s); 0 candidate(s) excluded.

## 6. Limitations and experiments not performed

- Gradient-alignment scores are first-order estimates about a differentiable loss. They are not calibrated probabilities and are not task metrics; a negative score is an estimated conflict signal, not proof that a record is wrong or that removing it helps.
- One training path is implemented: LoRA instruction tuning of a causal LM on plain chat records.
- Near-duplicate leakage detection is lexical; paraphrased overlap between splits is not detected.
- Compute is an accounting model (forward-token-equivalents), not measured FLOPs or energy.
