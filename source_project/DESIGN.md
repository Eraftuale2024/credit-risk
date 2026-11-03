# DESIGN: Probability of Default (PD) Model

## 1. Problem Definition and Business Objective

The objective is to estimate the probability of default (PD) at loan application time for unsecured personal loans ranging from USD 1,000 to USD 40,000, with repayment terms of 36 or 60 months.

The model is designed to support three lending decisions:

- **Approve:** Automatically approve applications with sufficiently low predicted risk.
- **Manual Review:** Refer intermediate-risk applications for additional credit assessment.
- **Decline:** Identify applications with sufficiently high predicted risk.

The target variable is `defaulted`:

- `1`: Charged Off or Default.
- `0`: Fully Paid.

The overall observed portfolio default rate is approximately 18.7%.

### 1.1 Lending economics

The assignment specifies the following economic assumptions:

- A repaid loan generates a gain of 10% of the loan amount.
- A defaulted loan results in a loss of 70% of the loan amount.

The expected return from approving a loan with default probability \(p\) is:

\[
EV(p)=0.10(1-p)-0.70p
\]

At the break-even point:

\[
0.10(1-p)-0.70p=0
\]

\[
0.10=0.80p
\]

\[
p=\frac{0.10}{0.80}=0.125
\]

Therefore, the economic break-even PD is **12.5%**.

Applications with predicted PD below 12.5% have positive expected return under the assignment's assumptions. This is an economic decision boundary, not a guarantee that an individual loan will be profitable.

The 12.5% approval threshold is derived directly from the assignment's lending economics. A separate 25% decline threshold is established as a predefined risk-policy boundary. Neither threshold is optimized against the 2015 validation outcomes. Applications between the two thresholds are referred for manual review.

---

## 2. Data and Temporal Design

The model uses the Data Scientist's preprocessing contract, version `1.0.0`.

The fitted transformation artifact, `contract/transform.joblib`, is reused without refitting during model development and prediction.

### 2.1 Dataset split

| Period | Loans | Purpose |
|---|---:|---|
| Through 2014-12-31 | 14,365 | Model training |
| 2015 | 15,623 | Out-of-time validation |
| 2016 | 5,000, without labels | Holdout scoring and drift monitoring |

### 2.2 Temporal controls

The project uses chronological separation to reduce temporal leakage and better represent real lending deployment conditions.

- The preprocessing contract uses a fitting cutoff of 2014-12-31.
- Model training uses loans issued through 2014-12-31.
- Model experiments use time-aware cross-validation within the training period.
- Earlier quarters are used to train each fold, and subsequent quarters are used to evaluate it.
- Random train-validation splitting is not used.
- Probability calibration is fitted using out-of-fold predictions generated within the training period.
- The 2015 dataset is reserved for out-of-time model evaluation and is not used to fit the preprocessing contract, train the estimator, or fit the calibration model.
- The 2016 holdout is reserved for scoring and drift analysis. It has no labels available for supervised performance evaluation.

### 2.3 Outcome maturity limitation

The loan-servicing snapshot is dated March 2019.

Some 60-month loans issued from April 2014 were still in progress at the snapshot date. Consequently, their observed default rates may be overstated by approximately eight percentage points. This is an estimate, not a directly measured correction.

This maturity limitation affects the interpretation of observed outcomes, particularly for 60-month loans.

For that reason:

- Performance is examined separately by loan term.
- Calibration is evaluated by loan term.
- Observed default rates are interpreted with regard to outcome maturity.
- The model does not treat incomplete repayment histories as fully mature outcomes.

Raw issue year and issue month are not included directly as predictive features. However, the application date is used with the earliest credit-line date to derive `credit_history_months`.

---

## 3. Data Contract and Transformation Architecture

The Data Scientist's preprocessing contract is the interface between data preparation and model development.

The contract defines the accepted raw inputs, data-quality rules, feature transformations, missing-value handling, engineered variables, and output representation.

### 3.1 Contract artifacts

| Artifact | Purpose |
|---|---|
| `contract/data_spec.json` | Documents input fields, data types, validation rules, exclusions, and transformation specifications |
| `contract/transform.joblib` | Fitted transformation pipeline |
| `contract/findings.md` | Documents data-quality findings and transformation decisions |
| `contract/analytical_dataset.parquet` | Transformed analytical dataset used for model development |
| `contract/reference_transforms.csv` | Reference records used for transformation-parity verification |

The analytical dataset and reference records are private working artifacts and must not be published because they contain borrower-level information.

### 3.2 Transformation responsibilities

The transformation pipeline addresses:

- Date-format inconsistencies.
- Numeric formatting and scale inconsistencies.
- Invalid or inconsistent financial values.
- FICO range inconsistencies.
- Missing credit-utilization information.
- Income and balance outliers.
- Credit-history derivation.
- Financial ratios and engineered features.
- Factor-score generation.
- Principal component analysis.
- Weight of Evidence encoding.

### 3.3 Important transformation decisions

**Income**

Annual income values are normalized, including values represented in thousands. Extreme income values are capped at the training-derived 99.5th percentile of USD 300,000 and log-transformed where specified by the contract.

**Debt-to-income ratio**

DTI is validated against the range 0–100. It is not capped at the training 99.5th percentile because the validation analysis found elevated default risk among loans above the previous percentile boundary.

The upper WoE treatment preserves this high-risk information.

**Credit utilization**

`revol_util` is normalized to percentage representation. The training-derived 99.5th percentile cap is 99.2.

The 2016 holdout contains a utilization-unit inconsistency: values are represented as fractions rather than percentages. Batch-level scale detection and rescaling are incorporated into the transformation logic.

**Missing values**

- `bc_util` is filled using the training-derived median of 69.4 and accompanied by `bc_util_missing`.
- `revol_util` is filled using the training-derived median of 57.0.
- Missing credit-history information uses the training-derived median of 179 months.

Transformation parameters are not re-estimated from the 2015 validation or 2016 holdout datasets.

### 3.4 Contract integrity

The model checks compatibility between:

- The configuration version.
- `data_spec.json`.
- The fitted transformation artifact.

A contract-version mismatch causes training or prediction to be refused rather than silently accepting incompatible transformations.

Transformation parity is checked against reference outputs to detect unexpected changes in preprocessing behavior.

---

## 4. Input Features and Leakage Controls

The primary model uses application-time information represented by the preprocessing contract.

The selected feature set combines raw application and credit variables with six engineered factor scores.

### 4.1 Selected predictive information

The raw feature groups cover:

- Loan amount and repayment term.
- Applicant income and debt burden.
- FICO credit score.
- Credit-history length.
- Delinquencies and public records.
- Revolving credit utilization.
- Credit balances and available credit.
- Recent account-opening activity.
- Contract-derived categorical WoE variables.

The six factor scores represent:

1. Credit footprint.
2. Revolving capacity.
3. Utilization stress.
4. Recent activity.
5. Public records.
6. Debt burden.

The primary LightGBM model uses the `raw+factors` feature set.

### 4.2 Excluded information

The following information is excluded from the primary model:

**Outcome-derived information**

`bureau_recovery_flag` is excluded because it is directly associated with the outcome. Its presence is associated with 100% default in the development data, and it achieves standalone AUC of approximately 0.874. It is therefore treated as leakage, not legitimate application-time predictive information.

**Post-origination servicing information**

All ten loan-servicing columns are excluded from model features because they contain information generated after loan origination.

**Existing platform risk-model outputs**

`grade`, `sub_grade`, and `int_rate` are excluded from the primary model.

These fields reflect the existing platform's risk assessment and pricing process. Including them would make the comparison against the platform's existing risk model circular.

A separate comparison model includes platform grade and interest rate. This variant is reported as a benchmark, not selected as the primary model.

**Geographic information**

`zip_code` and `addr_state` are excluded from the selected model's predictive feature set because geographic information may act as a proxy for protected characteristics.

State information is retained for regional monitoring and outcome analysis only.

### 4.3 Out-of-domain applications

Joint applications were absent from the training population but appeared in later datasets:

- 16 joint applications in 2015.
- 96 joint applications in 2016.

They are treated as out-of-domain applications and flagged for manual review rather than being treated as ordinary in-domain applications.

---

## 5. Model Architecture and Experimentation

Two principal models are evaluated: an interpretable WoE scorecard and a LightGBM gradient-boosting model.

### 5.1 WoE scorecard

The scorecard uses logistic regression trained on contract-generated Weight of Evidence variables.

Its purpose is to provide an interpretable statistical baseline and a comparison against the more flexible tree-based model.

The scorecard configuration excludes `installment` and `installment_to_income` from the candidate WoE variables.

The scorecard's regularization parameter is selected through training-period experiments.

The selected regularization parameter is:

- `C = 0.1`

The scorecard provides an interpretable benchmark, although its validation performance is slightly below that of the selected LightGBM model.

### 5.2 LightGBM primary model

LightGBM is selected as the primary predictive model.

Final configuration:

| Parameter | Value |
|---|---:|
| Estimators | 200 |
| Learning rate | 0.03 |
| Number of leaves | 8 |
| Minimum child samples | 100 |
| Subsample | 0.8 |
| Subsample frequency | 1 |
| Column sampling | 0.8 |
| L2 regularization | 5 |
| Random seed | 42 |
| Class weighting | Dynamic positive-class weighting, approximately 4.8 |

The positive-class weight is calculated from the training target distribution:

\[
w_{positive}=\frac{N_{negative}}{N_{positive}}
\]

This addresses class imbalance without applying synthetic oversampling or undersampling.

### 5.3 Monotonic constraints

Monotonic constraints are applied to selected features where the expected direction of credit risk is clear.

The model is constrained so that:

- Higher DTI does not reduce predicted risk.
- Higher revolving utilization does not reduce predicted risk.
- Higher loan-to-income does not reduce predicted risk.
- More recent account openings do not reduce predicted risk.
- Higher FICO scores do not increase predicted risk.
- Higher annual income does not increase predicted risk.

No monotonic constraint is imposed on total credit balances because their observed relationship with default is not consistently monotonic.

These constraints help align the model's behavior with established credit-risk relationships.

### 5.4 Feature ablation

The experiments compare four feature configurations:

1. Raw features only.
2. Factor scores only.
3. Raw features combined with factor scores.
4. Raw features combined with platform grade and interest rate, for comparison only.

Training-period cross-validation AUC results:

| Feature configuration | CV AUC |
|---|---:|
| Raw features | 0.6970 |
| Factor scores only | 0.6297 |
| Raw + factors | 0.6980 |
| Raw + platform variables | 0.7124 |

The selected primary model uses raw features combined with factor scores.

The small improvement over raw features indicates that factor scores add limited incremental ranking performance in this experiment. They are retained as part of the contract and evaluated transparently rather than being presented as a major performance improvement.

The platform-variable comparison achieves a higher CV AUC, but is not selected because its inputs reflect the existing platform's risk and pricing decisions.

### 5.5 Probability calibration

Class weighting can change the distribution of predicted probabilities. Since lending decisions depend on probability values rather than ranking alone, calibration is an important part of the pipeline.

The selected LightGBM model generates out-of-fold predictions within the training period.

A logistic regression model is fitted using the log-odds of those out-of-fold predictions and the corresponding training outcomes.

This is Platt scaling.

The calibration transformation is:

\[
PD_{calibrated}=
\frac{1}
{1+\exp[-(a\cdot logit(PD_{raw})+b)]}
\]

The fitted calibration coefficients are stored with the model artifact.

Neither 2015 validation labels nor 2016 holdout data are used to fit the calibration model.

---

## 6. Evaluation Strategy and Results

The final model is evaluated on the 2015 out-of-time validation dataset.

Evaluation considers ranking performance, probability quality, lending economics, segment stability, and operational behavior.

### 6.1 Ranking performance

The primary ranking metrics are:

- ROC-AUC.
- Gini coefficient.
- Kolmogorov-Smirnov statistic.

Gini is calculated as:

\[
Gini=2\times AUC-1
\]

Results:

| Model | AUC | Gini | KS |
|---|---:|---:|---:|
| WoE scorecard | 0.7111 | 0.4223 | 0.3079 |
| LightGBM primary | 0.7154 | 0.4308 | 0.3219 |
| Platform comparison | 0.7307 | 0.4613 | 0.3363 |
| Grade only | 0.7026 | 0.4053 | 0.2990 |

The primary LightGBM model outperforms the WoE scorecard and grade-only benchmark on the reported validation metrics.

The platform comparison model achieves higher AUC, but it is not selected as the main model because it includes the platform's existing risk and pricing outputs.

The results are also examined by quarter and loan term to identify temporal and maturity-related differences.

### 6.2 Probability quality

Probability quality is evaluated using:

- Brier score.
- Mean predicted PD.
- Observed default rate.
- Calibration plots.
- Calibration by loan term and relevant borrower segments.

For the 2015 validation dataset:

| Model | Brier score | Mean predicted PD | Observed default rate |
|---|---:|---:|---:|
| WoE scorecard | 0.1459 | 17.32% | 20.09% |
| LightGBM primary | 0.1447 | 19.07% | 20.09% |

LightGBM has a slightly better Brier score than the scorecard.

Its mean predicted PD is approximately 19.07%, compared with an observed default rate of approximately 20.09%. This indicates an overall calibration gap that requires continued monitoring.

Calibration differences are also examined by loan term because incomplete 60-month outcomes can distort observed default rates.

### 6.3 Business evaluation

The business evaluation considers:

- Approval rate.
- Manual-review rate.
- Decline rate.
- Default rate among approved loans.
- Default rate among declined loans.
- Expected profit under the assignment's economic assumptions.

For an approved loan, the modeled expected profit is:

\[
Profit=LoanAmount\times[0.10(1-PD)-0.70(PD)]
\]

Realized portfolio profit is calculated using observed outcomes in the validation dataset, not by assigning assumed outcomes to manual-review applications.

### 6.4 Primary decision-policy results

Using the LightGBM model with the 12.5% approval threshold and 25% decline threshold:

| Measure | Result |
|---|---:|
| Approval rate | 35.13% |
| Manual-review rate | 36.21% |
| Decline rate | 28.66% |
| Default rate among approved loans | 8.56% |
| Default rate among declined loans | 36.82% |
| Estimated profit on approved loans | USD 2,294.29 thousand |
| Profit per approved dollar | 2.96% |

The policy reduces the default rate in the approved portfolio compared with approving every application.

For comparison, approving everyone results in a validation default rate of approximately 20.09% and negative estimated profit of approximately USD 17,037.77 thousand.

### 6.5 Sensitivity analysis

Alternative thresholds are evaluated to understand the trade-off between approval volume, portfolio risk, and expected return.

For example:

- Lowering the approval threshold to 10% reduces the approval rate to approximately 26.59% and increases estimated profit to approximately USD 2,353.90 thousand.
- Raising the approval threshold to 15% increases the approval rate to approximately 43.44%, but reduces estimated profit to approximately USD 1,947.48 thousand.

These are policy sensitivity results, not evidence that the final thresholds were optimized on validation data.

The primary policy remains the predefined 12.5% and 25% decision framework.

### 6.6 Segment analysis

Validation performance is also examined by region and income band.

Examples from the 2015 regional analysis:

| Region | Approval rate | Default rate among approved |
|---|---:|---:|
| Midwest | 33.13% | 8.03% |
| Northeast | 36.24% | 8.14% |
| South | 35.01% | 9.05% |
| West | 35.80% | 8.57% |

Examples from the income analysis:

| Income band | Approval rate | Default rate among approved |
|---|---:|---:|
| Below USD 40,000 | 15.47% | 11.11% |
| USD 40,000–60,000 | 29.60% | 10.03% |
| USD 60,000–90,000 | 38.13% | 8.25% |
| Above USD 90,000 | 49.16% | 7.49% |

These comparisons support monitoring and investigation. They do not establish that the model is fair or free from indirect discrimination.

### 6.7 Stability analysis

Population Stability Index (PSI) is used to monitor changes in:

- Predicted PD distributions.
- Important input variables.
- The population scored over time.

The recommended monitoring interpretation is:

- PSI above 0.10: investigate the change.
- PSI above 0.25: raise an alert and investigate more urgently.

The 2016 holdout is used for feature and prediction drift monitoring, not supervised performance measurement, because its target labels are unavailable.

---

## 7. Decision Policy and Operational Safeguards

The decision policy uses two predefined PD boundaries.

| Predicted PD | Decision |
|---|---|
| Below 12.5% | Approve |
| 12.5% to below 25% | Manual Review |
| 25% or higher | Decline |

The 12.5% boundary is the economic break-even point under the assignment's 10% gain and 70% loss assumptions.

The 25% boundary is a separate predefined risk-policy threshold.

Applications flagged as out-of-domain or requiring manual intervention must not receive an automatic decision solely from their predicted PD.

### 7.1 Out-of-domain and input-risk flags

The prediction pipeline checks for conditions including:

- Joint applications not represented in training.
- FICO values below the training-supported range.
- DTI values above the normal valid range.
- Unusual revolving-utilization representation.
- Issue dates outside the supported training period.

Certain conditions result in a review requirement rather than an ordinary automated decision.

### 7.2 Input validation

Incoming application records are validated against `contract/data_spec.json`.

Validation includes:

- Required input fields.
- Numeric and categorical data types.
- Allowed ranges and values.
- Missing-value rules.
- Financial consistency checks.
- Loan installment consistency with loan amount, interest rate, and repayment term.

Invalid records are rejected rather than silently accepted.

The installment consistency check is a validation control. `installment` and `int_rate` are not direct predictive inputs to the selected primary LightGBM model.

### 7.3 Explainability and reason codes

The prediction pipeline uses LightGBM prediction contributions to identify factors associated with increased predicted risk.

The system:

- Uses a minimum contribution impact threshold of 0.02.
- Returns up to four relevant reason codes.
- Uses plain-language descriptions.
- Reports adverse risk factors relative to reference values.
- Generates reasons reproducibly for the same input and model.

Reason codes are intended to support credit-officer review. They should be reviewed by qualified credit professionals before being used in formal lending notices.

---

## 8. Reproducibility, Testing, and Operational Performance

### 8.1 Reproducibility

The project uses:

- A centralized configuration file.
- Fixed random seed of 42.
- A versioned transformation contract.
- A saved model artifact.
- A training experiment log.
- Model metadata.

The metadata records information including:

- Git commit.
- Training-data hash.
- Training date.
- Contract version.
- Fit cutoff.
- Training population size.
- Training default rate.
- Model configuration.
- Feature-set selection.
- Decision thresholds.

These controls support traceability and reproducibility of the model-development process.

### 8.2 Automated tests

The test suite covers:

- Contract-version compatibility.
- Demonstration examples.
- Input validation.
- Transformation and prediction parity.
- Reason-code behavior.
- Decision thresholds.

Tests are intended to detect contract incompatibility, invalid inputs, inconsistent prediction behavior, and changes to decision logic.

### 8.3 Prediction latency

The complete single-record prediction path is benchmarked using 300 records.

| Metric | Result |
|---|---:|
| Startup time | 0.03 seconds |
| p50 | 61.2 ms |
| p95 | 74.3 ms |
| p99 | 119.1 ms |
| Mean | 63.7 ms |
| Maximum | 129.1 ms |
| Target | Below 100 ms |

The p50 and p95 results are below the target. However, p99 and maximum latency exceed 100 ms.

Therefore, the current implementation does not meet a strict under-100-ms latency requirement for all or nearly all requests. Further profiling and optimization are required before making a production latency guarantee.

---

## 9. Production Architecture and Monitoring

### 9.1 Serving architecture

The prediction system is designed around a lightweight application or API interface.

The intended prediction sequence is:

1. Receive an application record.
2. Validate the input against the data contract.
3. Check contract compatibility.
4. Apply the fitted transformation pipeline.
5. Generate the LightGBM raw probability.
6. Apply the saved calibration parameters.
7. Evaluate out-of-domain and operational flags.
8. Assign an approval, review, or decline category where permitted.
9. Return the predicted PD, decision, and relevant reason codes.

The fitted transformation and trained model are loaded once at application startup rather than repeatedly for every prediction.

### 9.2 Production monitoring

Monitoring should include:

- PSI for predicted PD and important input variables.
- Missing and invalid input rates.
- Out-of-range input rates.
- Share of joint and other out-of-domain applications.
- Approval, review, and decline distributions.
- Prediction latency at p50, p95, and p99.
- Calibration by loan term when sufficiently mature outcomes become available.
- Changes in regional and income-segment outcomes.

PSI above 0.10 should trigger investigation, while PSI above 0.25 should trigger a more urgent alert.

### 9.3 Retraining strategy

Retraining should be considered quarterly or when monitoring identifies material changes in population characteristics, data quality, or model performance.

A new model release should:

1. Define a new historical training cutoff.
2. Fit a new transformation contract using the permitted training data.
3. Assign a new contract version.
4. Rebuild the analytical dataset.
5. Train and evaluate the model.
6. Fit probability calibration using training-period out-of-fold predictions.
7. Perform independent out-of-time validation.
8. Review decision policy and operational performance.
9. Update model metadata and documentation.
10. Complete deployment approval before replacing the current model.

Recent 60-month loan outcomes require particular caution because repayment outcomes may not have fully matured.

---

## 10. Results, Design Changes, and Limitations

### 10.1 Changes informed by experimentation

The final implementation preserves the original time-based data split, fitted preprocessing contract, leakage controls, and economic decision framework.

The following findings informed the final design:

1. **LightGBM was selected as the primary model.** It achieved validation AUC of 0.7154, compared with 0.7111 for the WoE scorecard and 0.7026 for grade alone.

2. **Platform pricing variables remain excluded from the primary model.** The comparison model using platform grade and interest rate achieved validation AUC of 0.7307. However, these variables reflect the existing platform's risk and pricing process, so the model is retained only as a comparison.

3. **Factor scores were retained in the preprocessing contract.** Their addition to raw features produced only a marginal improvement in training cross-validation AUC, from 0.6970 to 0.6980. They are not presented as a major source of incremental predictive performance.

4. **Principal components remain contract outputs but are not selected as primary model inputs.** Their availability supports analytical exploration without requiring their use in the final estimator.

5. **Probability calibration remains important.** The primary model achieved a validation Brier score of 0.1447 and mean predicted PD of approximately 19.07%, compared with an observed default rate of 20.09%. Calibration differences, particularly by term, remain a monitoring priority.

6. **The decision thresholds were retained from the lending economics and policy design.** The 12.5% approval boundary is derived from the 10% gain and 70% loss assumptions. The 25% decline boundary is a predefined policy choice. Neither was optimized against 2015 validation outcomes.

7. **The holdout utilization-unit inconsistency was addressed in the transformation logic.** The 2016 holdout represents `revol_util` as a fraction rather than a percentage. Batch-level scale detection and rescaling were incorporated to handle this difference.

8. **Operational limitations remain documented.** The p99 prediction latency exceeds the 100 ms target, the model has not been validated for rejected applicants, and out-of-domain populations require additional assessment.

These findings describe the implemented solution and its measured results. The documentation does not imply that the model was retrained or that its parameters were changed as a result of writing this design document.

### 10.2 Limitations

The model is developed using accepted and funded loans. It has not been validated for applicants rejected by the original lending platform or for applicants who were approved but did not take the loan.

Additional limitations include:

- Selection bias inherited from the original lending policy.
- Incomplete maturity of some 60-month loan outcomes.
- Calibration differences across borrower segments and loan terms.
- Limited representation of joint applications.
- Potential indirect geographic effects through non-location variables.
- A p99 prediction latency above the stated target.
- No demonstrated guarantee of fairness or regulatory compliance.
- No labeled outcomes available for supervised performance evaluation on the 2016 holdout.

### 10.3 Responsible use

The model should initially support human credit assessment as a decision-support system.

It should not be treated as an independently validated replacement for credit-risk governance or qualified lending judgment.

Before production lending decisions rely on it, the system requires:

- Independent model validation.
- Fairness and disparate-impact assessment.
- Review of reason codes and adverse-action requirements.
- Data privacy and security review.
- Model governance approval.
- Operational monitoring and incident procedures.
- Additional latency optimization.

The model's predicted PD is an estimate based on historical data and assumptions. It does not guarantee the outcome of an individual loan.

---

## 11. Project Artifacts

The project separates data-contract development, model development, evaluation, and application serving.

| Location | Responsibility |
|---|---|
| `configs/config.yaml` | Model and pipeline configuration |
| `contract/` | Data specification and fitted transformation artifacts |
| `src/contract_transform.py` | Transformation contract implementation |
| `src/treat.py` | Data cleaning and treatment |
| `src/features.py` | Engineered feature creation |
| `src/train.py` | Model training and calibration |
| `src/evaluate.py` | Validation metrics and business-policy evaluation |
| `src/predict.py` | Input validation, prediction, calibration, decisions, and reason codes |
| `src/verify_contract.py` | Transformation parity verification |
| `tests/` | Automated contract, validation, parity, and decision tests |
| `reports/` | Experiment, evaluation, monitoring, and benchmark outputs |
| `notebooks/` | Exploratory analysis and development evidence |
| `models/` | Trained model and metadata |
| `app.py` | Interactive prediction application |
| `DESIGN.md` | Technical and business design |
| `model_card.md` | Model purpose, performance, limitations, and responsible-use documentation |

### Data privacy

Original borrower datasets, holdout records, analytical datasets, and borrower-level reference fixtures must remain private.

Only approved code, documentation, configuration, non-sensitive aggregate results, and permitted model artifacts should be considered for publication.

The public repository must not expose personal borrower information or assignment-provided raw data.

---

**Final design position:** The project delivers a reproducible application-time credit-risk pipeline built around a versioned preprocessing contract, a LightGBM primary model, out-of-fold probability calibration, an economically grounded decision policy, automated validation controls, and documented operational limitations. Its measured performance supports further controlled evaluation, but independent validation and production governance remain necessary before autonomous lending deployment.