import pandas as pd

def add_features(d, med_hist, inc_cap):
    f = pd.DataFrame(index=d.index)
    inc = d.annual_inc.clip(upper=inc_cap)
    months = (d.issue_d.dt.year - d.earliest_cr_line.dt.year) * 12 + (d.issue_d.dt.month - d.earliest_cr_line.dt.month)
    f["credit_history_months"] = months.fillna(med_hist)
    f["loan_to_income"] = d.loan_amnt / inc
    f["installment_to_income"] = d.installment * 12 / inc
    f["fico_mid"] = (d.fico_range_low + d.fico_range_high) / 2
    f["never_delinquent"] = d.mths_since_last_delinq.isna().astype(int)
    f["bc_util_missing"] = d.bc_util.isna().astype(int)
    return f