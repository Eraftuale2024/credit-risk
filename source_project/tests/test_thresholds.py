import joblib, pytest
from src.predict import Scorer

def _set(scorer, monkeypatch, approve, decline):
    monkeypatch.setattr(scorer, "thresholds", {**scorer.thresholds, "approve_below": approve, "decline_at_or_above": decline})

def test_thresholds_follow_the_economics(scorer):
    th = scorer.thresholds
    assert th["break_even_pd"] == pytest.approx(0.10 / (0.10 + 0.70))
    assert th["approve_below"] == pytest.approx(th["break_even_pd"])
    assert th["approve_below"] < th["decline_at_or_above"]

def test_expected_value_changes_sign_at_break_even():
    ev = lambda p: 0.10 * (1 - p) - 0.70 * p
    assert ev(0.12) > 0 > ev(0.13)

def test_bands(scorer, base, monkeypatch):
    p = scorer.predict(base)["pd"]
    _set(scorer, monkeypatch, p + 0.02, p + 0.05); assert scorer.predict(base)["decision"] == "approve"
    _set(scorer, monkeypatch, p - 0.02, p + 0.02); assert scorer.predict(base)["decision"] == "review"
    _set(scorer, monkeypatch, p - 0.05, p - 0.02); assert scorer.predict(base)["decision"] == "decline"

def test_flagged_record_is_never_auto_decided(scorer, base, monkeypatch):
    _set(scorer, monkeypatch, 1.1, 1.2)  # would approve everyone
    out = scorer.predict(dict(base, dti="150"))
    assert out["decision"] == "review" and out["out_of_domain"]

def test_model_without_thresholds_is_refused(tmp_path):
    b = joblib.load("models/pd_model.joblib"); b["metadata"]["thresholds"] = None
    joblib.dump(b, tmp_path / "m.joblib")
    with pytest.raises(RuntimeError):
        Scorer(model_path=str(tmp_path / "m.joblib"))