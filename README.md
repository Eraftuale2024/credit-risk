# Credit Risk Prediction — Probability of Default (PD)

An end-to-end credit-risk machine learning project that estimates the Probability of Default (PD) for unsecured personal loan applicants and supports three decision categories: **Approve, Manual Review, and Decline**.

The project combines data quality engineering, a reproducible preprocessing contract, feature engineering, machine learning, probability calibration, model evaluation, explainability, and an interactive prediction application.

## 1. Business Objective

Lending institutions need to assess applicant default risk before issuing loans.

This project aims to:

- Estimate the probability of default using application-time information.
- Support consistent lending decisions.
- Identify lower-risk applications for approval.
- Route intermediate-risk applications for manual assessment.
- Identify high-risk applications for potential decline.
- Provide interpretable explanations and input warnings.

The system is designed as a decision-support tool, not a replacement for human credit officers.

## 2. Project Components

The repository contains two connected components:

**A. Prediction application**
- Interactive applicant input.
- Input validation.
- Preprocessing and model inference.
- Calibrated PD estimates.
- Approve, Manual Review, and Decline decisions.
- Risk reason codes and review warnings.

**B. Analytical and engineering project**
- Data auditing and quality assessment.
- Reproducible data transformation contract.
- Feature engineering and feature analysis.
- Model training and experiment tracking.
- Out-of-time validation and error analysis.
- Drift and latency evaluation.
- Automated tests and technical documentation.

## 3. Dataset and Temporal Design

The analysis uses the four files supplied for the assignment:

- `applications.csv`
- `credit_bureau.csv`
- `loan_servicing.csv`
- `holdout_applications.csv`

| Period | Loans | Default rate | Purpose |
|---|---:|---:|---|
| 2013–2014 | 14,365 | 17.3% | Model development and training |
| 2015 | 15,623 | 20.1% | Out-of-time validation |
| 2016 | 5,000 | Unavailable | Holdout scoring and drift monitoring |

The preprocessing contract is fitted using training-period data only, with a cutoff of 31 December 2014.

**Leakage controls**

- `bureau_recovery_flag` was identified as an outcome-derived variable and excluded from model features.
- Loan-servicing variables were excluded from application-time prediction because they contain post-origination information.
- The 2016 holdout was kept separate from model development and validation.

Original datasets, borrower-level records, and holdout predictions are intentionally excluded from this repository.

## 4. Data Engineering and Feature Development

The data preparation pipeline addresses:

- Duplicate application records.
- Inconsistent date formats.
- Financial-value scaling and unit inconsistencies.
- Missing values and missingness indicators.
- Invalid numerical values and out-of-range observations.
- FICO score inconsistencies.
- Credit utilization transformations.
- Derived financial and credit-history features.

The preprocessing contract includes fitted transformations, feature definitions, and a versioned data specification.

Feature engineering evaluates raw application variables, financial ratios, credit-history measures, WoE transformations, factor representations, and principal component analysis.

The selected model uses raw and factor-based features. PCA and alternative feature sets were evaluated as experiments rather than silently incorporated into the final model.

## 5. Machine Learning Methodology

### Primary model: LightGBM

The selected application model is a LightGBM classifier.

| Parameter | Value |
|---|---:|
| Estimators | 200 |
| Learning rate | 0.03 |
| Number of leaves | 8 |
| Minimum child samples | 100 |
| Class weight | Approximately 4.8:1 |
| Random seed | 42 |
| Probability calibration | Platt scaling |

The model uses selected monotonic constraints to enforce expected risk relationships.

Probability calibration is fitted using out-of-fold predictions generated within the training period.

### Baseline: WoE scorecard

A logistic regression scorecard using Weight of Evidence transformations provides an interpretable baseline for comparison.

Model experiments use chronological validation within the training period. The final evaluation is performed on the separate 2015 out-of-time dataset.

## 6. Model Performance

### 2015 out-of-time validation

| Model | AUC | Gini | KS |
|---|---:|---:|---:|
| Grade only | 0.7026 | 0.4053 | 0.2990 |
| WoE scorecard | 0.7111 | 0.4223 | 0.3079 |
| **Selected LightGBM** | **0.7154** | **0.4308** | **0.3219** |
| Grade and interest-rate comparison | 0.7307 | 0.4613 | 0.3363 |

The grade and interest-rate variant is a comparison experiment, not the selected application model.

The selected LightGBM model achieves a Brier score of 0.1447, compared with 0.1459 for the scorecard.

### Calibration

| Segment | Mean predicted PD | Observed default rate |
|---|---:|---:|
| Overall | 19.1% | 20.1% |
| 36-month loans | 15.0% | 14.8% |
| 60-month loans | 31.5% | 36.4% |

The 60-month segment has a material calibration gap and requires monitoring.

## 7. Business Decision Policy

The model's PD estimates are translated into three decision categories.

| Predicted PD | Decision |
|---|---|
| Below 12.5% | Approve |
| 12.5% to below 25% | Manual Review |
| 25% or higher | Decline |

The approval threshold is derived from the assumed lending economics:

- Repaid loan gain: 10%.
- Default loss: 70%.
- Break-even PD: 12.5%.

The 25% decline threshold is a predefined risk-policy boundary. Neither threshold is optimized against the 2015 validation outcomes.

### Historical 2015 policy results

| Measure | Result |
|---|---:|
| Approved | 35.1% |
| Manual review | 36.2% |
| Declined | 28.7% |
| Default rate among approved | 8.6% |
| Default rate among declined | 36.8% |
| Profit on approved loans | Approximately $2.29 million |

Profit is calculated for approved loans only. No financial outcome is assumed for applications sent to manual review.

These results describe historical evaluation performance and are not guarantees of future returns.

## 8. Explainability and Input Controls

The prediction pipeline provides:

- Input validation and contract checks.
- PD estimates and decision categories.
- Up to four risk-related reason codes.
- Out-of-domain warnings.
- Manual-review overrides for selected unsupported inputs.

Examples of out-of-domain conditions include joint applications, FICO scores below 660, and DTI above 100.

These controls help prevent unsupported automatic decisions and provide additional information for human review.

## 9. Monitoring and Limitations

The project evaluates population stability, calibration, segment performance, robustness, and prediction latency.

Important limitations include:

- **Accepted-loan selection bias:** the development data contains accepted loans, not the complete population of rejected applicants.
- **Incomplete loan outcomes:** some 60-month loans had not fully matured at the observation date.
- **Calibration differences:** the 60-month segment requires additional attention.
- **Geographic proxies:** state is retained for monitoring rather than as a predictive feature, but indirect proxies may remain.
- **Out-of-domain applications:** joint applications and other unsupported cases require additional assessment.

The model requires further independent validation, fairness assessment, governance, and operational testing before production lending use.

## 10. Prediction Latency

The full single-record prediction pipeline was benchmarked over 300 records.

| Metric | Latency |
|---|---:|
| p50 | 61.2 ms |
| p95 | 74.3 ms |
| p99 | 119.1 ms |
| Maximum | 129.1 ms |
| Target | 100 ms |

The p99 latency exceeds the 100 ms target, so further optimization is needed for strict latency requirements.

## 11. Technology Stack

- Python
- Pandas and NumPy
- SciPy
- Scikit-learn
- LightGBM
- Joblib
- PyYAML
- Gradio and Streamlit application components
- Pytest
- Git and GitHub

## 12. Repository Structure

```text
credit-risk/
│
├── app.py
├── streamlit_app.py
├── README.md
├── requirements.txt
│
├── models/
├── contract/
├── src/
│
└── source_project/
    ├── app.py
    ├── README.md
    ├── DESIGN.md
    ├── model_card.md
    ├── requirements.txt
    │
    ├── configs/
    ├── contract/
    ├── models/
    ├── notebooks/
    ├── reports/
    ├── src/
    └── tests/
```

The `source_project/` directory contains the analytical implementation, technical documentation, experiment notebooks, reports, and automated tests.

## 13. Running the Application

Clone the repository:

```bash
git clone https://github.com/Eraftuale2024/credit-risk.git
cd credit-risk
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install the root application dependencies:

```bash
pip install -r requirements.txt
```

Run the deployed application using its configured entry point:

```bash
streamlit run streamlit_app.py
```

For the analytical project, use the separate instructions in `source_project/README.md`.

Some contract parity and training checks require private assignment data and reference fixtures. Those files are intentionally excluded from the public repository.

## 14. Documentation

The analytical project includes:

- `source_project/DESIGN.md` — technical architecture and modeling methodology.
- `source_project/model_card.md` — model purpose, performance, limitations, and intended use.
- `source_project/contract/data_spec.json` — data contract specification.
- `source_project/contract/findings.md` — data quality and leakage findings.
- `source_project/notebooks/` — analytical workflow.
- `source_project/reports/` — aggregate evaluation and experiment results.
- `source_project/tests/` — automated tests.

## 15. Responsible Use

This project is an educational and technical demonstration of a credit-risk decision-support system.

It is not a production-certified lending system and has not been established as compliant with any specific jurisdiction's lending or fair-credit regulations.

Production deployment would require independent model validation, fairness and bias assessment, privacy and security review, governance approval, and ongoing monitoring.

---

**Project:** Credit Risk Prediction — Probability of Default
**Primary model:** LightGBM with Platt calibration
**Contract version:** 1.0.0
**Repository:** [Eraftuale2024/credit-risk](https://github.com/Eraftuale2024/credit-risk)
**Author:** Abraham Getachew
