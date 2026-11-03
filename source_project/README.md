\# Credit Risk Prediction — Data Contract and PD Model



\## 1. Project Overview



This project develops a reproducible credit-risk assessment pipeline that estimates the Probability of Default (PD) for loan applicants.



The solution combines two main components:



\- \*\*Data Contract Engineering:\*\* data quality assessment, data cleaning, feature engineering, and reproducible transformations.

\- \*\*Machine Learning Engineering:\*\* model development, probability calibration, performance evaluation, and application-level prediction.



The selected model uses LightGBM with raw application variables and engineered credit-risk factors.



\## 2. Project Architecture



| Directory / File | Purpose |

|---|---|

| `app.py` | Prediction application interface |

| `configs/` | Pipeline and model configuration |

| `contract/` | Data specification, findings, and fitted transformation |

| `notebooks/` | Data exploration and analytical development |

| `reports/` | Model evaluation and analysis outputs |

| `src/` | Data transformation, training, evaluation, and prediction code |

| `tests/` | Automated tests |

| `DESIGN.md` | Technical design and methodology |

| `model\_card.md` | Model purpose, performance, and limitations |

| `requirements.txt` | Python dependencies |



\## 3. Data Sources and Privacy



The project uses the four datasets supplied for the assignment:



\- `applications.csv`

\- `credit\_bureau.csv`

\- `loan\_servicing.csv`

\- `holdout\_applications.csv`



Original datasets and borrower-level derived artifacts are not included in this repository to protect data privacy.



Loan-servicing data is reserved for audit and is excluded from model features to prevent post-origination information leakage.



\## 4. Modeling Strategy



The project follows a chronological development strategy:



| Period | Purpose |

|---|---|

| 2013–2014 | Model and transformation development |

| 2015 | Out-of-time validation |

| 2016 | Holdout scoring |



The data transformation contract is fitted using training-period data. Probability calibration uses out-of-fold predictions generated within the training period.



The selected LightGBM model combines raw application features and engineered credit-risk factors.



\## 5. Technology Stack



\- Python

\- Pandas and NumPy

\- Scikit-learn

\- LightGBM

\- SciPy

\- PyArrow

\- Gradio

\- Pytest



\## 6. Installation



Create and activate a virtual environment:



```bash

python -m venv .venv

```



Activate it in Windows PowerShell:



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



Install the dependencies:



```bash

python -m pip install -r requirements.txt

```



\## 7. Running Tests



From the project root:



```bash

python -m pytest tests/

```



Some parity and training workflows require private reference data and the analytical dataset. These files are intentionally excluded from the repository.



\## 8. Documentation



\- `DESIGN.md` — data preparation, contract architecture, feature engineering, modeling, validation, and technical limitations.

\- `model\_card.md` — model purpose, intended use, performance, and risks.

\- `contract/data\_spec.json` — formal data contract specification.

\- `contract/findings.md` — data quality findings and investigation results.



\## 9. Limitations and Responsible Use



The model is intended as a decision-support system, not an autonomous lending authority.



Its predictions are subject to historical-data limitations, selection bias, calibration uncertainty, and potential data drift. Applications flagged as out of distribution or requiring additional review should follow the documented manual review process.



Model performance should be monitored after deployment, and lending decisions should remain subject to appropriate governance and human oversight.
