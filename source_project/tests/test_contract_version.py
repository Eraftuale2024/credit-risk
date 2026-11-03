import json, shutil, joblib, pytest
from src.predict import Scorer, ContractVersionError

def test_matching_versions_load(scorer):
    assert scorer.meta["contract_version"] == scorer.spec["contract_version"] == scorer.ct.contract_version

def test_expected_version_mismatch_is_refused():
    with pytest.raises(ContractVersionError):
        Scorer(expected_version="9.9.9")

def test_model_trained_on_another_contract_is_refused(tmp_path):
    b = joblib.load("models/pd_model.joblib"); b["metadata"]["contract_version"] = "0.9.0"
    joblib.dump(b, tmp_path / "m.joblib")
    with pytest.raises(ContractVersionError):
        Scorer(model_path=str(tmp_path / "m.joblib"))

def test_spec_of_another_version_is_refused(tmp_path):
    d = tmp_path / "contract"; d.mkdir()
    shutil.copy("contract/transform.joblib", d)
    spec = json.load(open("contract/data_spec.json")); spec["contract_version"] = "2.0.0"
    json.dump(spec, open(d / "data_spec.json", "w"))
    with pytest.raises(ContractVersionError):
        Scorer(contract_dir=str(d))