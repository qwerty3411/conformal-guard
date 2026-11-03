# Triage check: does the review queue hold up on more answers?

Robustness check on the frozen data (alpha = 0.05); nothing re-tuned, headline results unchanged. The combiner is fitted on the train split only, so calibration and test answers are both unseen by it.

## 1. Both directions of the frozen split

| Judged answers | Answers | Contradicted | Sent to review | Contradicted answers in the review queue (95% CI) | Blocked & contradicted | Shipped & contradicted | Relative risk | Fisher p |
|---|---|---|---|---|---|---|---|---|
| Test (calibrated on calibration answers; the headline) | 245 | 11 | 31% | 7 of 11 = 64% (31%–89%) | 0 | 4 = 1.6% | 3.8 | 0.039 |
| Calibration (calibrated on test answers; swapped) | 328 | 23 | 0% | 0 of 23 = 0% (0%–15%) | 0 | 23 = 7.0% | 0.0 | 1 |
| **Pooled: every answer judged once** | 573 | 34 | 13% | 7 of 34 = 21% (9%–38%) | 0 | 27 = 4.7% | 1.7 | 0.2 |

Ship threshold 0.97099 forward, 0.93199 swapped.

## 2. 1,000 random re-splits (328 calibrate / 245 judged)

| | Mean | Median | Middle 95% of re-splits |
|---|---|---|---|
| Contradicted answers in the review queue | 21.4% | 18.8% | 0.0%–57.2% |
| Answers sent to review | 17.2% | 19.6% | 0.0%–32.7% |
| Shipped with a contradiction (bounded by 0.05 on average) | 4.7% | 4.5% | 2.0%–8.2% |
| Contradicted answers per judged part | 14.5 | 15 | 9–20 |

In 65% of re-splits the queue held a larger share of the contradicted answers than of all answers (random review would hold the same share).

## 3. Review budgets: send the riskiest share by weakest-sentence score

| Reviewed | All answers: contradicted caught (95% CI) | Test split only |
|---|---|---|
| 10% | 5 of 34 = 15% (5%–31%) | 3 of 11 = 27% |
| 20% | 8 of 34 = 24% (11%–41%) | 4 of 11 = 36% |
| 30% | 13 of 34 = 38% (22%–56%) | 6 of 11 = 55% |
| 50% | 19 of 34 = 56% (38%–73%) | 7 of 11 = 64% |

Answer-level AUROC of the weakest-sentence score: all answers 0.54 (95% bootstrap CI 0.44–0.64); test split 0.64 (0.45–0.81).
