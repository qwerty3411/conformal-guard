# Conformal Guard results (false = contradicted, label rule local, main alpha 0.05)

Answers: train 246, calibrate 328, test 245. Claims: 4840.

**Raw Muse Spark answers (test): 4.5% contain at least one false claim** (0.8% of claims are false).

## Guarantee on held-out test answers

| alpha | target 1-alpha | test coverage | 95% CI | claims kept | true claims kept | non-empty answers |
|---|---|---|---|---|---|---|
| 0.01 | 99.0% | 99.6% | 97.7%–100.0% | 10.4% | 10.4% | 50.6% |
| 0.02 | 98.0% | 99.6% | 97.7%–100.0% | 17.1% | 17.2% | 72.7% |
| 0.03 | 97.0% | 99.6% | 97.7%–100.0% | 28.0% | 28.1% | 95.5% |
| 0.04 | 96.0% | 99.6% | 97.7%–100.0% | 41.6% | 41.9% | 100.0% |
| 0.05 | 95.0% | 98.4% | 95.9%–99.6% | 63.2% | 63.5% | 100.0% |
| 0.075 | 92.5% | 95.5% (no filtering needed) | 92.1%–97.7% | 100.0% | 100.0% | 100.0% |
| 0.1 | 90.0% | 95.5% (no filtering needed) | 92.1%–97.7% | 100.0% | 100.0% | 100.0% |
| 0.15 | 85.0% | 95.5% (no filtering needed) | 92.1%–97.7% | 100.0% | 100.0% | 100.0% |

## 1000 random re-splits (alpha = 0.05)

Mean test coverage 95.1%; conformal theory predicts 95.1%. Individual re-splits range 92.2%–97.6% (middle 90%) because each test set has only 245 answers; the guarantee is about the average, and the histogram matches the predicted Beta-binomial curve.

Share of claims kept, averaged over the same re-splits (each score calibrated separately): combined 83.2% (66.0%–100.0%), agreement 81.6% (57.8%–100.0%), evidence 91.6% (80.4%–100.0%), random 80.9% (60.9%–100.0%).

Every alpha, averaged over the same 1000 re-splits (unfiltered answers: 94.0% clean):

| alpha | target | mean coverage | middle 90% | sentences kept | answers with a sentence kept |
|---|---|---|---|---|---|
| 0.01 | 99.0% | 99.1% | 97.6%–100.0% | 13.1% | 59.4% |
| 0.02 | 98.0% | 98.1% | 95.9%–99.6% | 28.9% | 91.3% |
| 0.03 | 97.0% | 97.2% | 94.7%–99.2% | 47.1% | 99.5% |
| 0.04 | 96.0% | 96.0% | 93.1%–98.4% | 69.5% | 100.0% |
| 0.05 | 95.0% | 95.1% | 92.2%–97.6% | 83.2% | 100.0% |
| 0.075 | 92.5% | 94.0% | 92.2%–95.9% | 99.8% | 100.0% |
| 0.1 | 90.0% | 94.0% | 92.2%–95.9% | 100.0% | 100.0% |
| 0.15 | 85.0% | 94.0% | 92.2%–95.9% | 100.0% | 100.0% |

## Baselines

- Keep everything: coverage 95.5%
- Fixed cutoff (combined score > 0.5), no calibration: coverage 95.5%, claims kept 100.0%
- Random score + conformal: coverage 96.7% (still valid), claims kept 66.4% (useless)

## Which scores matter (alpha = 0.05)

| score | AUROC (true vs false claims) | test coverage | claims kept |
|---|---|---|---|
| combined | 0.704 | 98.4% | 63.2% |
| agreement | 0.552 | 97.1% | 70.5% |
| evidence | 0.674 | 97.6% | 81.0% |
| random | 0.527 | 96.7% | 66.4% |

## Per group (alpha = 0.05)

| group | calibration answers | test answers | coverage (shared threshold) | coverage (group threshold) | claims kept (group threshold) |
|---|---|---|---|---|---|
| clinical | 114 | 86 | 98.8% | 100.0% | 27.5% |
| consumer | 90 | 66 | 97.0% | 95.5% | 83.4% |
| medication | 124 | 93 | 98.9% | 100.0% | 34.2% |

## Other guarantee modes

- Learn-Then-Test (with 95.0% confidence, error <= 0.05): test coverage 99.6%, claims kept 28.6%
- Lenient (expected share of false claims per answer <= 0.05): test value 0.8%, claims kept 100.0%
- Ship / review / block: ship 68.6%, review 31.4%, block 0.0%; P(shipped and wrong) = 1.6% (bound 0.05), P(wrong | shipped) = 2.4%

## Decision matrix (245 test answers, alpha = 0.05)

| decision | no contradicted sentence | has a contradicted sentence |
|---|---|---|
| shipped unchanged | 164 (66.9%) | 4 (1.6%) |
| filtered, sent for review | 70 (28.6%) | 7 (2.9%) |
| blocked | 0 (0.0%) | 0 (0.0%) |

The guarantee bounds one cell: shipped unchanged with a contradicted sentence <= 0.05 of all answers.

## Sentence filter matrix (1449 test sentences, alpha = 0.05; applied to every answer)

| filter | not contradicted | contradicted |
|---|---|---|
| kept | 912 (63.5% of these) | 4 (33.3% of these) |
| dropped | 525 (36.5% of these) | 8 (66.7% of these) |

## Combiner (logistic regression, fit on train only)

| feature | weight (standardised) |
|---|---|
| agree_mean | +0.344 |
| agree_frac | +0.171 |
| evidence_support | -0.148 |
| retrieval_sim | +0.048 |
| log_words | +0.083 |
| position | +1.069 |
