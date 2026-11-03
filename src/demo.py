import re

_COMMON = {"issue_d": "Jun-2016", "application_type": "Individual", "pub_rec": 0, "pub_rec_bankruptcies": 0}

EXAMPLES = {
    "Low risk": {**_COMMON, "loan_amnt": 8000, "term": "36 months", "int_rate": 7.5, "grade": "A", "sub_grade": "A4",
                 "purpose": "credit_card", "annual_inc": 125000, "emp_length": "10+ years", "emp_title": "engineer",
                 "home_ownership": "MORTGAGE", "verification_status": "Not Verified", "dti": 8.5, "addr_state": "CA",
                 "earliest_cr_line": "Jan-1998", "fico_range_low": 785, "fico_range_high": 789, "inq_last_6mths": 0,
                 "delinq_2yrs": 0, "open_acc": 12, "total_acc": 30, "revol_bal": 6000, "revol_util": 12, "bc_util": 10,
                 "tot_cur_bal": 260000, "avg_cur_bal": 21000, "total_rev_hi_lim": 50000, "tot_hi_cred_lim": 320000,
                 "mort_acc": 2, "acc_open_past_24mths": 2, "num_tl_op_past_12m": 1},
    "Borderline": {**_COMMON, "loan_amnt": 15000, "term": "36 months", "int_rate": 13.5, "grade": "C", "sub_grade": "C3",
                   "purpose": "debt_consolidation", "annual_inc": 65000, "emp_length": "5 years", "emp_title": "office manager",
                   "home_ownership": "RENT", "verification_status": "Source Verified", "dti": 18, "addr_state": "TX",
                   "earliest_cr_line": "Mar-2005", "fico_range_low": 685, "fico_range_high": 689, "inq_last_6mths": 1,
                   "delinq_2yrs": 0, "open_acc": 11, "total_acc": 24, "revol_bal": 14000, "revol_util": 56, "bc_util": 65,
                   "tot_cur_bal": 32000, "avg_cur_bal": 2900, "total_rev_hi_lim": 25000, "tot_hi_cred_lim": 48000,
                   "mort_acc": 0, "acc_open_past_24mths": 4, "num_tl_op_past_12m": 2},
    "High risk": {**_COMMON, "loan_amnt": 17000, "term": "60 months", "int_rate": 21.0, "grade": "E", "sub_grade": "E3",
                  "purpose": "debt_consolidation", "annual_inc": 38000, "emp_length": "< 1 year", "emp_title": "driver",
                  "home_ownership": "RENT", "verification_status": "Verified", "dti": 29, "addr_state": "FL",
                  "earliest_cr_line": "Aug-2011", "fico_range_low": 665, "fico_range_high": 669, "inq_last_6mths": 3,
                  "delinq_2yrs": 1, "mths_since_last_delinq": 10, "open_acc": 9, "total_acc": 15, "revol_bal": 17000,
                  "revol_util": 92, "bc_util": 95, "tot_cur_bal": 20000, "avg_cur_bal": 2200, "total_rev_hi_lim": 18500,
                  "tot_hi_cred_lim": 24000, "mort_acc": 0, "acc_open_past_24mths": 7, "num_tl_op_past_12m": 5},
}


def installment_for(amount, rate, months):
    r = rate / 1200
    return round(amount * r / (1 - (1 + r) ** -months), 2)


def to_record(values):
    rec = {k: v for k, v in values.items() if v is not None and v != ""}
    rec["id"] = "demo"
    try:
        months = int(re.search(r"\d+", str(rec.get("term", ""))).group())
        rec["installment"] = installment_for(float(rec["loan_amnt"]), float(rec["int_rate"]), months)
    except (KeyError, TypeError, ValueError, AttributeError, ZeroDivisionError):
        pass
    return rec