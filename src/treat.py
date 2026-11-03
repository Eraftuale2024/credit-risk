import numpy as np, pandas as pd

HEAVY = ["annual_inc", "revol_bal", "tot_cur_bal", "avg_cur_bal", "total_rev_hi_lim", "tot_hi_cred_lim"]
RATIO = ["revol_util", "bc_util"]
RAW = ["loan_amnt", "dti", "open_acc", "total_acc", "mort_acc", "acc_open_past_24mths", "num_tl_op_past_12m",
       "inq_last_6mths", "delinq_2yrs", "pub_rec", "pub_rec_bankruptcies"]

def treat(d, spec):
    x = pd.DataFrame(index=d.index)
    for c in HEAVY:
        x[c] = np.log1p(d[c].clip(upper=spec[c]["cap_p99.5"]))
    for c in RATIO:
        x[c] = d[c].fillna(spec["missing"][c]["fill"]).clip(upper=spec[c]["cap_p99.5"])
    for c in RAW:
        x[c] = d[c]
    x["fico_mid"] = (d.fico_range_low + d.fico_range_high) / 2
    months = (d.issue_d.dt.year - d.earliest_cr_line.dt.year) * 12 + (d.issue_d.dt.month - d.earliest_cr_line.dt.month)
    x["credit_history_months"] = months.fillna(spec["features"]["credit_history_months_fill"])
    return x