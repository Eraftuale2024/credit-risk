# Findings: Credit Risk Data Science Report

Fit cutoff: loans issued up to 2014-12-31 (14,365 loans). Every cap, bin, scaler and factor is fitted on these loans only.

## 1. Summary for the credit manager

*What i did. i turned three messy sources (applications, credit bureau, loan servicing) into one reliable dataset of 29,988 loans and found the signals that separate good borrowers from bad ones.

Five things you should know

1. Data quality needed real work. About 1 in 10 rows had a planted or genuine problem: duplicate loans (600), income typed in thousands (899), loan amounts inflated 10 times (300), swapped credit scores (150), impossible credit-line dates (300), inconsistent states (3,000) and mixed date formats (1,200). Each was fixed only when we could prove the fix, and 12 loans with irreconcilable pricing were dropped.
2. **One column was a hidden leak.** `bureau_recovery_flag` was set to 1 on 4,207 loans and every one of them defaulted. It records an outcome, not an application fact, so it is excluded. All ten loan-servicing columns are excluded for the same reason (recorded after the loan was issued).
3. The portfolio loses money without a model.** 18.7% of loans default. With the stated economics (a default loses 70% of the loan, a repaid loan earns 10%), the break-even default probability is 12.5%, so approving everyone loses money. Loan term matters most: 60-month loans default 32.4%, 36-month loans 14.1%.
4. Recent 60-month default rates are overstated.** Loans still in progress were removed, and mostly the good ones, so 60-month loans issued from April 2014 look worse than they will end up (roughly 8 points, a rough estimate). 36-month loans are all finished and give a reliable trend (+1.9 points across cohorts).
5. **What predicts default.** Strongest independent signals: loan term, credit score, loan size relative to income, and recent account activity (borrowers who opened many accounts recently default 25% against 11% for the quietest). Individually most fields are weak, which is normal in credit data. Six hidden factors summarize the bureau data; they add only a modest amount on top of the platform's grade (AUC 0.674 to 0.693 in a screening test).

**Fairness and limits.** We recommend not using zip code or state as model inputs: they add almost no predictive power and can stand in for protected characteristics. Keep state for monitoring only. All findings come from accepted loans, so they may not hold for applicants the platform declined.

**Recommendation.** Use the cleaned contract as the model input. Do not use the platform's grade or interest rate in the main model, since they are the old model's own output. Monitor debt-to-income: 2015 applicants carry more debt and 2% exceed the earlier cap.

## 2. Technical appendix

### 2.1 Data audit and cleaning
Issue table in `reports/issue_log.csv` (issue, rows, decision, reason). Highlights:
- applications.csv had 30,600 rows for 30,000 ids: 600 near-duplicates (`emp_title` case, `dti` +0.01); label, amounts and dates identical, so the first row was kept.
- `issue_d` and `earliest_cr_line` used three formats (Nov-2015, 2015-11-01, 11/2015) in 1,200 rows each; parsed with explicit formats, 0 failures.
- `annual_inc` below 1,000 (899 rows): multiplied by 1,000; after the fix the group's median is 63.5k against 65k for the rest.
- `loan_amnt` ×10 (300 rows, 20 inside the valid range): the installment formula recovers the true amount, always a multiple of 25.
- `fico_range_low` above high (150 rows): exact swap (gap of 4 in all normal rows).
- `earliest_cr_line` after `issue_d` (300 rows): set to missing with flag `cr_line_invalid`; shifted dates that do not cross `issue_d` cannot be detected, so some probably remain.
- `revol_util` and `bc_util` negative (about 1% each): absolute value; for `revol_util` the recomputed balance/limit matches the absolute value as closely as clean rows; for `bc_util` no limit column exists, so this is an assumption.
- `addr_state`: 3,000 rows with wrong case, spaces or full names (652); mapped to 49 two-letter codes.
- 12 loans dropped: installment contradicts amount, rate and term, and the rate contradicts the sub-grade.
- Label against servicing: 1 contradiction among 30,000.

### 2.2 Target analysis
- Default rate 18.73% (5,617 of 29,988). Fit set 17.3%, 2015 20.1%.
- By term: 36 months 14.05%, 60 months 32.37%.
- Servicing snapshot is March 2019. Every 36-month loan had matured; 60-month loans from 2014Q2 on had not.
- 36-month rate: 12.6% (cohorts to 2014Q1) to 14.5% (later). 60-month: 24.7% to 34.9%. If real drift was the same for both terms, roughly 8 points of the 60-month rise is censoring bias (an assumption, not a measurement).
- Consequences: validate and report calibration by term; never use issue year or month as a feature.

### 2.3 Financial values and outliers
- Profile (mean, median, std, IQR, skew, p1 to p99): `annual_inc` skew 63, maximum 7.5M; balances and limits skew 3 to 13.
- Mean ± 3 std flags 0.4% of incomes while the IQR rule flags 4.3%: the extreme value inflates the std and widens its own fence, and the lower fence is negative (−156,597) for incomes.
- On a log scale the share flagged for balances and limits drops from 3% to 7% to under 0.3%: most raw "outliers" were normal skew.
- Errors fixed (unit errors, sign flips). Genuine extremes kept: 500k to 1.25M incomes have matching job titles and low `dti`. The 7.5M income cannot be verified, so it is capped, not deleted.
- Method: cap at the fit p99.5, then `log1p`, for `annual_inc`, `revol_bal`, `tot_cur_bal`, `avg_cur_bal`, `total_rev_hi_lim`, `tot_hi_cred_lim`. Skew falls to between −0.6 and 0.1 (except `revol_bal` −2.3 because of 40 real zero balances). Default by decile is identical before and after: the treatment keeps the ranking.
- Loans above the caps do not default more (4% to 12% against 17%), so capping loses no risk signal.
- `loan_amnt`, `installment`, `dti`: bounded, left raw.
- `dti` is deliberately not capped: 2015 loans above the earlier p99.5 (36.94) defaulted at 34.5% against 19.8%. Valid range 0 to 100 in the spec.
- Zero is a real value (no card debt, 40 loans); a zero limit makes utilization undefined; missing `bc_util` (151) gets the fit median plus a flag; missing was not shown to relate to default.
- All caps and fills: `reports/caps.json`, merged into `data_spec.json`.

### 2.4 Reliability and leakage
- Income by verification status: unverified applicants earn least and default least (12.7% against 19% to 19.3%), with the best grades and smallest loans: verification reflects the platform's decision, not a borrower trait. Round incomes are 45% to 50% in every group.
- Installment-to-income never exceeds 22.3%; income correlates about 0.55 with bureau limits in every verification group.
- Missingness: `mths_since_last_delinq` (46%) means never delinquent; `emp_length` and `emp_title` are blank together; none relates significantly to default (p = 0.09 to 0.24).
- Leakage scan: servicing columns and `bureau_recovery_flag` score AUC 0.72 to 0.91 alone; legitimate fields 0.50 to 0.68. Columns absent from the holdout are the ten servicing columns, `defaulted` and our own flag.
- Features: credit history months, loan-to-income (AUC 0.591, above both parents), installment-to-income, mid FICO, `never_delinquent`, `bc_util_missing`.
- Reliability ratings: loan terms, bureau amounts, FICO and utilization high (after repair); `annual_inc`, `dti`, `bc_util`, `earliest_cr_line`, `verification_status` medium; `emp_length` low to medium; `emp_title` low; `grade`, `sub_grade`, `int_rate` accurate but not independent of the platform's model.

### 2.5 Categorical variables, WoE and IV
- `emp_title` (9,093 titles) grouped into 11 job families by ordered keyword rules; minimum group size 300 loans; hospitality merged into other. `emp_length` mapped to <1y, 1 to 9y, 10+y (10+ years defaults 15.5%). Nine rare purposes and 31 states merged.
- IV on fit data: `bureau_recovery_flag` 9.37 (leak); `sub_grade` 0.50, `grade` 0.46, `int_rate` 0.45 (the platform's own model); `term` 0.20; `fico_mid` 0.11; `loan_to_income` 0.10; the rest below 0.09; job group 0.018, purpose 0.014, state 0.013.
- 20 variables with IV of at least 0.02 are scorecard candidates; the smallest bin has 779 loans. IV is in-sample and slightly optimistic.
- Bins and WoE values: `reports/woe_spec.json`.

### 2.6 Correlation, PCA, factor analysis
- Spearman: 7 pairs above 0.7 (`tot_cur_bal` and `tot_hi_cred_lim` 0.98). VIF: `tot_cur_bal` 92.5, `avg_cur_bal` 63.0, `tot_hi_cred_lim` 24.8, `open_acc` 8.8. `avg_cur_bal` is approximately `tot_cur_bal` divided by open accounts (rank correlation 0.97 of the log difference with log open accounts).
- PCA on 21 standardized, treated fields: 7 components above eigenvalue 1 and above the parallel-analysis threshold, 74.8% of variance. Treatment barely changes component directions (cosine 0.93 to 0.99) but protects them from outliers.
- Factor analysis: 19 fields (removed `delinq_2yrs`, KMO 0.44, and derived `avg_cur_bal`); overall KMO 0.710; Bartlett p about 0; 6 factors by parallel analysis; minimum-residual extraction; varimax.
- Factors: credit footprint, revolving capacity, utilization stress, recent activity, public records, debt burden (loadings in `reports/factor_spec.json`). They explain 61.4% of the variance; communality is low for inquiries (0.10) and credit history (0.15), so those stay separate inputs. Debt burden rests on two fields and is the weakest.
- Rotation: oblimin was rejected because it reproduced the observed correlations 5 times worse than varimax (mean error 0.100 against 0.019) and implied the wrong sign for one pair. Worst varimax pair: `open_acc` and `total_acc` (observed 0.68, implied 0.51).
- Factor analysis is preferred here because the fields are redundant measurements of a few behaviours and factors model shared variance and give nameable factors; PCA is for pure compression.
- Scores computed as standardized inputs times stored weights; they equal the library's scores to 1e-13.

### 2.7 Stability and risk insights
- PSI 2013 against 2014: all fields and factors below 0.1 (largest `int_rate` 0.070); pooling both years for the fit is justified. Default 16.0% then 18.0%.
- Default by factor decile, 95% Wilson intervals, Mann-Whitney and trend test: recent activity 11.1% to 24.7%, utilization stress 10.9% to 22.6%, credit footprint 20.0% to 12.0%, debt burden 13.6% to 21.4%; all six p-values below 0.01.
- The public-records factor runs against expectation; the raw public-record field has no signal (17.9% against 17.1%), so the factor carries no risk signal of its own and stays out of reason codes.
- Against the platform's grade: four of six factors correlate below 0.15 with grade; cross-validated AUC grade 0.674, factors alone 0.632, both 0.693. Within a grade band, recent activity still raises default by 4 to 8 points; utilization stress mostly restates the grade.
- Mortgage holders default 15.8% against 19.6%.

### 2.8 Location (zip_code, addr_state)
Location can act as a proxy for protected traits; this dataset has no race or age, so a proxy cannot be measured, only the cost of excluding it. State IV is 0.013 and the chi-square across 18 states gives p = 0.03 with overlapping intervals. 803 zip3 areas with a median of 10 loans. Recommendation: exclude both from the model, keep state for monitoring only.

### 2.9 Limits: accepted-only bias
The dataset contains only loans the platform accepted and funded. Applicants it declined, and applicants who were accepted but never took the loan, are absent, so we never observe what would have happened to them. A model trained on accepted loans learns the risk of applicants who already passed the platform's existing risk model, and it is not guaranteed to hold for the full applicant population. The effects are likely to be: (1) the observed range of risk is truncated, since the weakest applicants are mostly missing, which can shrink or distort relationships the model estimates, for instance the credit score distribution starts at about 660; (2) the model inherits the old policy's choices, including any bias in who was accepted; (3) if the model is used to approve applicants the old model would have declined, performance on that group is unverified. In production the model should be used first as a second opinion on already-accepted applicants, with monitoring of outcomes, and reject inference or a controlled test of approvals just below the old cutoff would be needed before relying on it for declined populations.
<!-- section 2.10 -->
5,000 loans, unique ids, one date format, none of the planted corruptions of the training files. Differences from training: `term` has a leading space, 2 loans with `dti` above 100, 96 joint applications (1.9%; none in training, 16 in 2015), slightly more missing `emp_length` (6.7%) and `emp_title` (6.8%). `bureau_recovery_flag` is present but always 0 and ignored.
The holdout stores `revol_util` as a fraction of 1 (median 0.49) instead of a percentage (about 55 in 2014 and 2015); balance / limit reproduces it (49.4 against 49). The first PSI scan flagged it (utilization_stress PSI 5.7) and the first holdout scores were understated. The transform now rescales a batch of at least 100 rows whose 99th percentile is at most 1.5; it never triggers on training, 2015 or the reference rows, and a single record is never rescaled. A row-level rule was rejected because it would have changed 4 training rows. After the fix every PSI is below 0.03 and the PD distribution PSI is 0.012.
### 2.11 Contract
Files in `contract/`: `analytical_dataset.parquet` (29,988 cleaned loans, 2013 to 2015, with cleaned fields, features, job groups, six factor scores, seven principal components, WoE-encoded scorecard variables and the label), `transform.joblib` (fitted on the 14,365 loans up to 2014-12-31, stores only plain numbers: caps, fills, means, standard deviations, PCA components and factor weights, so scoring needs no factor-analysis library), `data_spec.json` (37 input fields with type, valid range, cleaning rule, missing strategy and reliability rating; factor definitions; WoE bins and groupings; excluded columns), `reference_transforms.csv` (50 raw rows chosen to trigger each cleaning rule, with expected outputs) and this file.
Version 1.0.0. Run `python -m src.verify_contract` from the project root: it reloads the transform, scores the 50 raw rows from text and compares every output with tolerance 1e-6.  Result: PARITY OK | rows: 50 | columns checked: 82 | max abs diff: 8.88e-16 (tolerance 1e-06)..
Ingestion steps outside the transform: removal of 600 repeated ids and 12 inconsistent loans. Cleaning rules inside it: date formats, term parsing, income unit fix, loan amount fix, FICO swap, utilization sign, state mapping, impossible credit lines.
Known limits: the planted corruptions exist only in the training files, so the cleaning rules are tested on the 50 reference rows and not on the 2016 holdout; the PCA components are shipped but the factor scores are the main reduced inputs; `bc_util` sign repair is an assumption.
PARITY OK | rows: 50 | columns checked: 82 | max abs diff: 8.88e-16 (tolerance 1e-06)