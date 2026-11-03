# Credit Risk Prediction — Probability of Default (PD)

An end-to-end machine learning application for estimating the probability of default on unsecured personal loans. The system supports credit-risk assessment through three decision categories: **Approve, Manual Review, and Decline**.

The application uses a trained LightGBM model, a fitted preprocessing contract, input validation, probability calibration, and interpretable reason codes. It is deployed as an interactive Streamlit application.

---

## 1. Business Problem

Lending institutions need to assess the likelihood that a loan applicant will fail to repay a loan.

Approving high-risk applications can increase credit losses, while rejecting applicants unnecessarily can reduce lending opportunities and potential revenue.

This project develops a Probability of Default (PD) model to support more consistent credit decisions.

**Business objectives:**
- Estimate an applicant's probability of default at application time.
- Identify applicants suitable for automatic approval.
- Route intermediate-risk applicants for manual assessment.
- Identify applications with sufficiently high predicted risk for decline.
- Provide explanations and warnings to support credit-risk review.

The model is intended as a decision-support system, not an autonomous replacement for a credit officer.

## 2. Application Overview

The application allows users to enter applicant information or select an illustrative applicant profile.

The prediction pipeline returns:

- Probability of Default (PD).
- Recommended decision category.
- Decision basis.
- Input validation warnings.
- Out-of-domain review flags.
- Main reason codes contributing to higher predicted risk.

**Decision workflow:**

1. Collect applicant and credit-bureau information.
2. Validate the submitted information.
3. Prepare the input using the original prediction pipeline.
4. Apply the trained model and probability calibration.
5. Generate the PD estimate.
6. Apply the predefined decision thresholds.
7. Display the decision and supporting explanations.

## 3. Dataset and Time-Based Design

The project uses the files supplied in the dedicated credit-risk take-home dataset:

- `applications.csv`
- `credit_bureau.csv`
- `loan_servicing.csv`
- `holdout_applications.csv`

The project separates development, validation, and scoring periods to reduce temporal leakage.

| Period | Loans | Default rate | Purpose |
|---|---:|---:|---|
| 2013–2014 | 14,365 | 17.3% | Model training |
| 2015 | 15,623 | 20.1% | Out-of-time validation |
| 2016 | 5,000 | Not available | Holdout scoring and drift monitoring |

The preprocessing contract is fitted using the training period only.

**Data leakage controls**

The analysis identified `bureau_recovery_flag` as an outcome-related variable: its value of 1 was associated with a 100% default rate in the development data. It was therefore excluded from model inputs.

Loan-servicing information was also excluded from application-time prediction because it represents information observed after loan origination.

The model uses application-time information rather than future repayment outcomes.

## 4. Machine Learning Methodology

### Main model: LightGBM

The primary model is a LightGBM classifier.

| Parameter | Configuration |
|---|---|
| Algorithm | LightGBM |
| Number of estimators | 200 |
| Learning rate | 0.03 |
| Number of leaves | 8 |
| Minimum child samples | 100 |
| Class weighting | 4.8 per default |
| Probability calibration | Platt scaling |
| Training sample | 14,365 loans |
| Random seed | 42 |

The model uses monotonic constraints on selected risk-related features. These constraints were designed to preserve expected directional relationships between certain applicant characteristics and default risk.

The predicted probabilities are calibrated using Platt scaling fitted on out-of-fold predictions.

### Baseline model

A logistic regression scorecard based on Weight of Evidence (WoE) transformations was developed as an interpretable baseline.

The comparison helps evaluate whether the boosted model provides meaningful predictive improvement over a more traditional credit-scoring approach.

### Feature engineering

The project evaluates raw applicant and credit-bureau variables, derived features, and factor-based representations.

The `loan_to_income` feature was particularly useful, with a reported Information Value (IV) of 0.10.

The factor scores provided little additional predictive information when combined with raw features.

## 5. Model Performance

### Out-of-time validation: 2015

| Model | AUC | Gini | KS |
|---|---:|---:|---:|
| Grade only | 0.7026 | 0.405 | 0.299 |
| WoE scorecard | 0.7111 | 0.422 | 0.308 |
| **LightGBM — main model** | **0.7154** | **0.431** | **0.322** |
| LightGBM with grade and rate | 0.7307 | 0.461 | 0.336 |

The model with grade and interest rate is a comparison model only, not the selected application model.

### Quarterly stability

The main model's quarterly AUC results were:

| Quarter | AUC |
|---|---:|
| Q1 | 0.697 |
| Q2 | 0.725 |
| Q3 | 0.718 |
| Q4 | 0.719 |

### Calibration

| Segment | Mean predicted PD | Observed default rate |
|---|---:|---:|
| Overall | 19.1% | 20.1% |
| 36-month loans | 15.0% | 14.8% |
| 60-month loans | 31.5% | 36.4% |

The 60-month segment shows a material calibration gap and requires additional monitoring.

## 6. Business Decision Policy

The model's probability estimates are translated into three decision categories using predefined economic thresholds.

### Decision thresholds

| Predicted probability of default | Decision |
|---|---|
| Below 12.5% | Approve |
| 12.5% to below 25% | Manual Review |
| 25% or higher | Decline |

The thresholds are based on the assumed lending economics:

- A default results in a loss of 70% of the loan.
- A repaid loan earns 10%.
- The break-even PD is 12.5%.

The thresholds were derived from the economic assumptions rather than tuned against the 2015 validation period.

### Reported 2015 decision outcomes

| Policy | Approved | Review | Declined |
|---|---:|---:|---:|
| Approve everyone | 100% | — | — |
| Proposed policy | 35.1% | 36.2% | 28.7% |

Under the proposed policy:

- Default rate among approved loans: 8.6%.
- Default rate among declined loans: 36.8%.
- Reported profit on approved loans: +$2.29 million.

The profit calculation covers approved loans only. No financial outcome is assumed for applications routed to manual review.

These are historical evaluation results, not guaranteed future financial returns.

## 7. Explainability and Risk Controls

The application provides supporting information alongside each prediction.

**Reason codes**

The system returns up to four plain-language reasons associated with increased predicted risk, subject to the configured impact threshold.

These explanations are intended to support human review and should be reviewed by qualified credit officers before being used in formal adverse-action notices.

**Input validation**

The prediction pipeline checks required fields, numerical ranges, date formats, categorical values, and other contract constraints.

Missing or invalid information is rejected when the validation contract requires it.

**Out-of-domain handling**

The system forces manual review for selected inputs outside the model's training domain, including:

- Joint applications.
- FICO scores below 660.
- DTI above 100.

Such applications can receive a PD estimate, but they are not automatically approved or declined.

## 8. Monitoring and Known Limitations

### Population Stability Index (PSI)

The project uses the following monitoring thresholds:

| PSI | Monitoring action |
|---|---|
| Above 0.10 | Investigate |
| Above 0.25 | Alert |

Reported PSI values across the 2015–2016 comparison were below 0.03 after correcting the `revol_util` unit issue.

Joint applications have proposed investigation and alert thresholds of 3% and 5%, respectively. Calibration gaps of 3 and 5 percentage points are also proposed monitoring thresholds.

These additional thresholds are project assumptions, not empirically established regulatory limits.

### Known limitations

**Accepted-loan selection bias**

The dataset contains accepted loans, not the full population of rejected applicants. Consequently, the model cannot establish reliable risk estimates for all rejected applicants without additional analysis.

Reject inference or controlled testing should be considered before extending the model to populations outside its development data.

**Incomplete loan outcomes**

Some 60-month loans issued from April 2014 were still in progress at the March 2019 observation snapshot. Their observed default rate may therefore be overstated.

**Calibration differences**

The model shows weaker calibration for 60-month loans and certain borrower segments. Segment-level monitoring is necessary.

**Limited long-term loan approvals**

Approximately 2% of approved loans in the historical evaluation were 60-month loans.

**Potential indirect geographic effects**

Location is excluded as a predictive feature because it can act as a proxy for protected characteristics. State is retained only for monitoring. Other variables may still indirectly encode geographic information, which has not been fully tested.

**Decision-support limitation**

The model should be used as a second opinion alongside human credit assessment, documented lending policies, fairness reviews, and ongoing performance monitoring.

## 9. Performance and Latency

The full single-record prediction pipeline was benchmarked on a laptop.

| Metric | Latency |
|---|---:|
| p50 | 61 ms |
| p95 | 74 ms |
| p99 | 119 ms |
| Maximum | 129 ms |
| Target requirement | 100 ms |

The p99 and maximum latency exceed the 100 ms target. Further optimization may be necessary for workloads requiring strict latency guarantees.

Benchmark reference: `reports/latency_benchmark.txt`.

## 10. Technology Stack

- **Language:** Python
- **Machine learning:** LightGBM, scikit-learn
- **Data processing:** Pandas, NumPy
- **Scientific computing:** SciPy
- **Model persistence:** Joblib
- **Application interface:** Streamlit
- **Deployment:** Streamlit Community Cloud

## 11. Project Structure

```text
credit-risk-pd/
│
├── streamlit_app.py
├── app.py
├── README.md
├── requirements.txt
│
├── models/
│   └── pd_model.joblib
│
├── contract/
│   ├── data_spec.json
│   └── transform.joblib
│
└── src/
    ├── __init__.py
    ├── predict.py
    ├── demo.py
    ├── features.py
    ├── treat.py
    └── contract_transform.py
```

The deployed Streamlit application uses the original prediction modules and saved model artifacts.

## 12. Run Locally

Clone the repository:

```bash
git clone https://github.com/abrahamgetachew/credit-risk-pd.git
cd credit-risk-pd
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run streamlit_app.py
```

## 13. Deployment

The application is designed for deployment through Streamlit Community Cloud.

The deployment entry point is:

```text
streamlit_app.py
```

The application requires the saved model, transformation contract, data specification, source modules, and pinned Python dependencies.

No training dataset is required to execute an individual prediction.

## 14. Responsible Use

This project is an educational and technical demonstration of a credit-risk decision-support system.

It is not a production-certified lending system and has not been established as compliant with any particular jurisdiction's credit, privacy, or fair-lending regulations.

Before production use, it would require independent validation, fairness and bias assessment, security review, governance approval, operational monitoring, and review by qualified credit-risk professionals.

---

**Project:** Credit Risk Prediction — Probability of Default

**Model:** LightGBM with calibrated probability estimates

**Contract:** 1.0.0

**Application:** Streamlit

**Author:** Abraham Getachew
