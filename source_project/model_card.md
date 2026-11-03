# Model card: Probability of Default (PD) model

**Purpose.** Predict the probability that an unsecured personal loan (1,000 to 40,000 USD, 36 or 60 months) defaults, at application time, to support auto-approve, manual review and decline decisions. Not validated for applicants the platform declined.

**Version.** Contract 1.0.0. The model metadata (`models/metadata.json`) records git commit, training data hash, training date and thresholds.

## Data window
| Period | Loans | Default rate | Use |
|---|---|---|---|
| Issued 2013-01 to 2014-12 | 14,365 | 17.3% | training; the contract transform is fitted on these only |
| Issued 2015 | 15,623 | 20.1% | validation, never used to fit or tune |
| Issued 2016 | 5,000, no labels | n/a | scored; drift monitoring |

## Model
- **Main:** LightGBM (200 trees, 8 leaves), class weights (4.8 per default), monotonic constraints on term, `dti`, revolving utilization, loan-to-income and recent account openings (risk rises) and on credit score and income (risk falls). Each direction was checked against the training data before training. Probabilities are calibrated by Platt scaling fitted on out-of-fold predictions.
- **Baseline:** logistic regression on the Data Scientist's WoE bins (18 variables), fully explainable.
- **Inputs:** application-time fields only. Not used: anything from the servicing file, `bureau_recovery_flag` (an outcome flag: 100% default when 1), `grade`, `sub_grade`, `int_rate`, `installment` (these reproduce the platform's own risk model), `zip_code` and `addr_state`.

## Performance on 2015 (15,623 loans)
| Model | AUC | Gini | KS |
|---|---|---|---|
| Grade only | 0.7026 | 0.405 | 0.299 |
| Scorecard | 0.7111 | 0.422 | 0.308 |
| **LightGBM (main)** | **0.7154** | **0.431** | **0.322** |
| LightGBM with grade and rate (comparison only) | 0.7307 | 0.461 | 0.336 |

By quarter the main model scores 0.697, 0.725, 0.718 and 0.719. Within a term the gain over grade is small: 36-month 0.666 against 0.667, 60-month 0.655 against 0.635. Calibration: mean PD 19.1% against 20.1% actual; 36-month 15.0% against 14.8%; **60-month 31.5% against 36.4%**.

## Decision policy (economics: a default loses 70% of the loan, a repaid loan earns 10%)
Approve below **12.5%** (break-even PD), decline at **25%** and above, review in between. Set from the economics, not tuned on 2015.

| 2015 | Approved | Review | Declined | Default rate approved | Default rate declined | Profit on approved loans |
|---|---|---|---|---|---|---|
| Approve everyone | 100% | none | none | 20.1% | n/a | -$17.0M |
| **Proposed** | **35.1%** | 36.2% | 28.7% | **8.6%** | 36.8% | **+$2.29M** |

Profit counts approved loans only; review outcomes are not assumed. Regions show similar outcomes (approval 33% to 36%, default among approved 8.0% to 9.0%). Approval rises with income (15.5% below 40k, 49.2% from 90k) and follows real default differences; income is a model input, so the credit team should review this.

## Which Data Scientist outputs helped
Cross-validated AUC inside 2014: raw fields 0.697, **factor scores alone 0.630**, raw plus factors 0.698, raw plus the platform's grade and rate 0.712.
- The cleaning (about 10% of rows repaired), the leakage findings and the WoE bins helped most: the scorecard on the WoE bins is as good as the boosted model.
- `loan_to_income` was the strongest new feature (IV 0.10).
- The factor scores summarize the bureau data but add nothing on top of the raw fields (+0.001, within noise). The principal components ship in the contract but neither model uses them.

## Known limits
- **Accepted loans only.** Declined applicants and accepted applicants who never took the loan are missing, so risk is truncated (credit scores start at about 660) and the model inherits the old policy. Use it first as a second opinion, and use reject inference or a controlled test before relying on it for declined populations.
- **In-progress bias.** 60-month loans issued from April 2014 were unfinished at the March 2019 snapshot, so their observed default rate is overstated (roughly 8 points, an estimate). Report metrics by term; issue date is never a feature.
- **Compressed ranking.** Excluding the platform's pricing costs about 0.015 AUC; the model over-predicts the safest grade and under-predicts grades D and E (by 6 and 9.5 points) and applicants with a blank employment section (by 6 points).
- **Few 60-month approvals.** About 2% of the approved book is 60-month.
- **Out-of-domain inputs.** The model has no information for FICO below 660, joint applications (none in training, 1.9% of 2016) or `dti` above 100. These are scored but always sent to review. Missing or damaged inputs are rejected by validation, because the model scores them silently otherwise.
- **Location** is excluded from the model because it can stand in for protected traits; state is used only to monitor outcomes. Other fields can carry geography indirectly, which was not tested.
- **Reason codes** list up to 4 plain reasons, only when the applicant is on the unfavorable side of the training median. The wording is mine and should be reviewed by a credit officer before use in notices.

## Monitoring (alert levels)
PSI of the PD or any top-10 input: investigate above 0.10, alert above 0.25 (all 2015 to 2016 values are below 0.03 after the `revol_util` unit fix). A first PSI above 0.25 should be checked for a unit or format change before it is read as drift. Joint applications: investigate above 3%, alert above 5%. Calibration by term once outcomes mature: investigate at a 3-point gap, alert at 5. The last three levels are my assumptions, not measured values.

## Latency
300 single-record predictions on a laptop through the full path: p50 61 ms, p95 74 ms, p99 119 ms, maximum 129 ms against a 100 ms requirement (`reports/latency_benchmark.txt`).