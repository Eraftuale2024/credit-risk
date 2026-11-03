import pandas as pd
from src.predict import REASONS, MIN_IMPACT
from src.contract_transform import RAW_INPUT

BANNED = ("grade", "interest", "state", "verif", "factor", "ownership", "public record", "region")

def test_reason_format(scorer, records):
    for r in records:
        rc = scorer.predict(r)["reason_codes"]
        assert len(rc) <= 4
        assert [x["impact"] for x in rc] == sorted((x["impact"] for x in rc), reverse=True)
        for x in rc:
            assert x["code"] in REASONS and x["reason"] == REASONS[x["code"]][1]
            assert x["impact"] >= MIN_IMPACT - 0.0005
            assert not any(w in x["reason"].lower() for w in BANNED)

def test_reasons_only_when_applicant_is_on_the_unfavorable_side(scorer, records):
    for r in records:
        codes = {x["code"] for x in scorer.predict(r)["reason_codes"]}
        c = scorer.ct.clean(pd.DataFrame([{k: r.get(k) for k in RAW_INPUT}], columns=RAW_INPUT)).iloc[0]
        if "credit_score" in codes: assert (c.fico_range_low + c.fico_range_high) / 2 < scorer.ref["fico_mid"]
        if "income" in codes: assert c.annual_inc < scorer.ref["annual_inc"]
        if "term" in codes: assert c.term == 60
        if "dti" in codes: assert c.dti > scorer.ref["dti"]

def test_strong_applicant_gets_no_income_or_score_reason(scorer, base):
    r = dict(base, annual_inc="150000", fico_range_low="760", fico_range_high="764")
    assert not {"income", "credit_score"} & {x["code"] for x in scorer.predict(r)["reason_codes"]}

def test_reasons_are_reproducible(scorer, base):
    assert scorer.predict(base) == scorer.predict(base)