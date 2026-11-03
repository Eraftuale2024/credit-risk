import pytest
from src.predict import ValidationError

def mod(base, **kw):
    r = dict(base)
    for k, v in kw.items():
        if v is None: r.pop(k, None)
        else: r[k] = v
    return r

@pytest.mark.parametrize("field", ["annual_inc", "loan_amnt", "term", "int_rate", "installment", "issue_d", "fico_range_low", "revol_bal", "addr_state"])
def test_missing_required_field_is_rejected(scorer, base, field):
    with pytest.raises(ValidationError) as e:
        scorer.predict(mod(base, **{field: None}))
    assert field in str(e.value)

@pytest.mark.parametrize("kw,needle", [
    ({"annual_inc": "abc"}, "annual_inc"), ({"annual_inc": "50000000"}, "valid range"), ({"term": "48 months"}, "term"),
    ({"dti": "-5"}, "dti"), ({"grade": "Z"}, "grade"), ({"issue_d": "garbage"}, "issue_d"), ({"addr_state": "Atlantis"}, "addr_state")])
def test_bad_values_are_rejected(scorer, base, kw, needle):
    with pytest.raises(ValidationError) as e:
        scorer.predict(mod(base, **kw))
    assert needle in str(e.value)

def test_installment_must_match_amount_rate_and_term(scorer, base):
    with pytest.raises(ValidationError) as e:
        scorer.predict(mod(base, installment=str(float(base["installment"]) + 50)))
    assert "installment" in str(e.value)

def test_all_problems_are_reported_together(scorer, base):
    with pytest.raises(ValidationError) as e:
        scorer.predict(mod(base, annual_inc="abc", term="48 months"))
    assert "annual_inc" in str(e.value) and "term" in str(e.value)

def test_extra_columns_are_ignored_and_reported(scorer, base):
    out = scorer.predict(dict(base, bureau_recovery_flag="1", defaulted="1"))
    assert any("ignored columns" in w for w in out["warnings"])
    assert out["pd"] == scorer.predict(base)["pd"]

def test_input_record_is_not_modified(scorer, base):
    before = dict(base); scorer.predict(base)
    assert base == before

@pytest.mark.parametrize("kw", [{"dti": "150"}, {"application_type": "Joint App"}, {"fico_range_low": "650", "fico_range_high": "654"}])
def test_out_of_domain_records_are_scored_but_sent_to_review(scorer, base, kw):
    out = scorer.predict(mod(base, **kw))
    assert out["out_of_domain"] and out["flags"] and out["decision"] == "review"