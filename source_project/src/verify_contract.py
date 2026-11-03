import sys, joblib, numpy as np, pandas as pd
from src.contract_transform import RAW_INPUT, to_plain

TOL = 1e-6

def verify(contract="contract"):
    ct = joblib.load(f"{contract}/transform.joblib")
    ref = pd.read_csv(f"{contract}/reference_transforms.csv", dtype=str)
    out = to_plain(ct.transform(ref[RAW_INPUT]))
    worst, bad = 0.0, []
    for c in out.columns:
        e = ref["exp_" + c]
        if pd.api.types.is_numeric_dtype(out[c]):
            a, b = out[c].astype("float64").to_numpy(), pd.to_numeric(e, errors="coerce").to_numpy(dtype=float)
            if (np.isnan(a) != np.isnan(b)).any(): bad.append(c); continue
            d = float(np.nanmax(np.abs(a - b))) if (~np.isnan(a)).any() else 0.0
            worst = max(worst, d)
            if d > TOL: bad.append(c)
        elif not (out[c].astype("object").where(out[c].notna(), "").astype(str).to_numpy() == e.fillna("").astype(str).to_numpy()).all():
            bad.append(c)
    status = "PARITY OK" if not bad else "PARITY FAILED"
    print(f"{status} | rows: {len(ref)} | columns checked: {len(out.columns)} | max abs diff: {worst:.3g} (tolerance {TOL:g})")
    if bad: print("mismatching columns:", bad)
    return 0 if not bad else 1

if __name__ == "__main__":
    sys.exit(verify())