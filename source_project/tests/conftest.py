import os, sys, pathlib, pytest, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
from src.predict import Scorer
from src.contract_transform import RAW_INPUT

@pytest.fixture(scope="session")
def scorer():
    return Scorer()

@pytest.fixture(scope="session")
def records():
    ref = pd.read_csv("contract/reference_transforms.csv", dtype=str)[RAW_INPUT]
    return [{k: v for k, v in r.items() if isinstance(v, str) and v != ""} for r in ref.to_dict("records")]

@pytest.fixture(scope="session")
def base(scorer, records):
    for r in records:
        out = scorer.predict(r)
        if not out["flags"] and not out["warnings"]:
            return r
    raise RuntimeError("no plain reference record found")