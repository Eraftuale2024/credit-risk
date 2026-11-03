import json, numpy as np, pandas as pd
from src.predict import Scorer
from src.contract_transform import RAW_INPUT

s = Scorer()
ref = pd.read_csv("contract/reference_transforms.csv", dtype=str)[RAW_INPUT]
recs = [{k: v for k, v in r.items() if isinstance(v, str) and v != ""} for r in ref.to_dict("records")]
batch = pd.DataFrame([{k: r.get(k) for k in RAW_INPUT} for r in recs], columns=RAW_INPUT)
Fb = s.ct.transform(batch)
bad, fw = [], []
for i, r in enumerate(recs):
    df, c, flags, warns = s.validate(r)
    fw.append((flags, warns))
    Fs = s.ct.transform(df, cleaned=c)
    num = [col for col in Fb.columns if pd.api.types.is_numeric_dtype(Fb[col]) and col in Fs.columns]
    a = Fb.iloc[[i]][num].astype(float).to_numpy()[0]; b = Fs[num].astype(float).to_numpy()[0]
    d = np.where(np.isnan(a) != np.isnan(b), np.inf, np.abs(np.nan_to_num(a) - np.nan_to_num(b)))
    if d.max() > 1e-9:
        bad.append((r["id"], [(num[j], round(float(d[j]), 4)) for j in np.argsort(-d)[:4]], flags, warns))
print(len(bad), "of", len(recs), "records differ between the batch row and the single-record transform")
for x in bad[:6]: print(x)
print("records with a flag or warning:", sum(1 for f, w in fw if f or w))
print("first record flags and warnings:", fw[0])
print(json.dumps(s.predict(recs[0]), indent=1))