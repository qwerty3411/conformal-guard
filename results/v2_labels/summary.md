# Conformal Guard results (false = contradicted, label rule v2, main alpha 0.1)

Answers: train 246, calibrate 328, test 245. Claims: 4840.

**Raw Muse Spark answers (test): 18.4% contain at least one false claim** (3.9% of claims are false).

## Guarantee on held-out test answers

| alpha | target 1-alpha | test coverage | 95% CI | claims kept | true claims kept | non-empty answers |
|---|---|---|---|---|---|---|
| 0.02 | 98.0% | 98.8% | 96.5%–99.7% | 12.9% | 13.2% | 56.7% |
| 0.04 | 96.0% | 97.6% | 94.7%–99.1% | 24.2% | 24.8% | 85.3% |
| 0.06 | 94.0% | 95.1% | 91.6%–97.4% | 38.6% | 39.3% | 100.0% |
| 0.08 | 92.0% | 94.3% | 90.6%–96.8% | 46.8% | 47.6% | 100.0% |
| 0.1 | 90.0% | 93.5% | 89.6%–96.2% | 50.9% | 51.7% | 100.0% |
| 0.15 | 85.0% | 89.4% | 84.8%–92.9% | 69.4% | 70.0% | 100.0% |
| 0.2 | 80.0% | 84.1% | 78.9%–88.4% | 88.3% | 88.6% | 100.0% |
| 0.3 | 70.0% | 81.6% (no filtering needed) | 76.2%–86.3% | 100.0% | 100.0% | 100.0% |

## 1000 random re-splits (alpha = 0.1)

Mean test coverage 90.2%; conformal theory predicts 90.3%. Individual re-splits range 85.3%–93.9% (middle 90%) because each test set has only 245 answers; the guarantee is about the average, and the histogram matches the predicted Beta-binomial curve.

Share of claims kept, averaged over the same re-splits (each score calibrated separately): combined 58.8% (52.6%–66.4%), agreement 34.3% (26.9%–41.1%), evidence 47.5% (39.4%–55.5%), random 37.1% (29.5%–43.2%).

Every alpha, averaged over the same 1000 re-splits (unfiltered answers: 78.9% clean):

| alpha | target | mean coverage | middle 90% | sentences kept | answers with a sentence kept |
|---|---|---|---|---|---|
| 0.02 | 98.0% | 98.1% | 95.9%–99.6% | 15.8% | 65.6% |
| 0.04 | 96.0% | 96.0% | 93.1%–98.4% | 30.0% | 91.8% |
| 0.06 | 94.0% | 94.2% | 90.6%–97.1% | 42.1% | 99.0% |
| 0.08 | 92.0% | 92.0% | 87.8%–95.5% | 52.2% | 99.9% |
| 0.1 | 90.0% | 90.2% | 85.3%–93.9% | 58.8% | 100.0% |
| 0.15 | 85.0% | 85.1% | 80.0%–89.8% | 78.2% | 100.0% |
| 0.2 | 80.0% | 80.4% | 75.5%–85.3% | 93.9% | 100.0% |
| 0.3 | 70.0% | 78.9% | 75.5%–82.0% | 100.0% | 100.0% |

## Baselines

- Keep everything: coverage 81.6%
- Fixed cutoff (combined score > 0.5), no calibration: coverage 81.6%, claims kept 100.0%
- Random score + conformal: coverage 91.8% (still valid), claims kept 35.4% (useless)

## Which scores matter (alpha = 0.1)

| score | AUROC (true vs false claims) | test coverage | claims kept |
|---|---|---|---|
| combined | 0.619 | 93.5% | 50.9% |
| agreement | 0.501 | 93.1% | 30.2% |
| evidence | 0.587 | 94.3% | 38.0% |
| random | 0.474 | 91.8% | 35.4% |

## Per group (alpha = 0.1)

| group | calibration answers | test answers | coverage (shared threshold) | coverage (group threshold) | claims kept (group threshold) |
|---|---|---|---|---|---|
| clinical | 114 | 86 | 94.2% | 95.3% | 45.9% |
| consumer | 90 | 66 | 93.9% | 90.9% | 64.8% |
| medication | 124 | 93 | 92.5% | 92.5% | 42.3% |

## Other guarantee modes

- Learn-Then-Test (with 95.0% confidence, error <= 0.1): test coverage 94.3%, claims kept 44.4%
- Lenient (expected share of false claims per answer <= 0.1): test value 3.9%, claims kept 100.0%
- Ship / review / block: ship 41.2%, review 58.8%, block 0.0%; P(shipped and wrong) = 6.9% (bound 0.1), P(wrong | shipped) = 16.8%

## Decision matrix (245 test answers, alpha = 0.1)

| decision | no contradicted sentence | has a contradicted sentence |
|---|---|---|
| shipped unchanged | 84 (34.3%) | 17 (6.9%) |
| filtered, sent for review | 116 (47.3%) | 28 (11.4%) |
| blocked | 0 (0.0%) | 0 (0.0%) |

The guarantee bounds one cell: shipped unchanged with a contradicted sentence <= 0.1 of all answers.

## Sentence filter matrix (1449 test sentences, alpha = 0.1; applied to every answer)

| filter | not contradicted | contradicted |
|---|---|---|
| kept | 719 (51.7% of these) | 18 (31.6% of these) |
| dropped | 673 (48.3% of these) | 39 (68.4% of these) |

## Combiner (logistic regression, fit on train only)

| feature | weight (standardised) |
|---|---|
| agree_mean | +0.452 |
| agree_frac | -0.484 |
| evidence_support | +0.223 |
| retrieval_sim | +0.123 |
| log_words | +0.075 |
| position | +0.328 |
