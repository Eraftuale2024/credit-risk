import numpy as np, pandas as pd
from src.verify_contract import verify
from src.predict import _calibrate
from src.contract_transform import RAW_INPUT

def test_reference_rows_reproduced():
    assert verify("contract") == 0

def test_single_record_path_equals_batch_path(scorer, records):
    batch = pd.DataFrame([{k: r.get(k) for k in RAW_INPUT} for r in records], columns=RAW_INPUT)
    F = scorer.ct.transform(batch); b = scorer.bundle["lgbm"]
    p = _calibrate(b["model"].predict_proba(F[b["columns"]])[:, 1], b["calibration"])
    got = np.array([scorer.predict(r)["pd"] for r in records])
    assert np.abs(got - p).max() < 6e-5  # pd is rounded to 4 decimals

def test_main_models_exclude_banned_columns(scorer):
    ex = scorer.spec["excluded_columns"]
    banned = set(ex["post_issue_servicing"]) | set(ex["outcome_derived"]) | {"defaulted"}
    platform = {"int_rate", "grade", "grade_num", "sub_grade", "installment", "installment_to_income", "addr_state", "zip_code"}
    platform |= {"woe_" + c for c in platform}
    for name in ("lgbm", "scorecard"):
        cols = set(scorer.bundle[name]["columns"])
        assert not cols & banned
        assert not cols & platform