import cProfile, pstats, pandas as pd
from src.predict import Scorer
from src.contract_transform import RAW_INPUT

s = Scorer()
ref = pd.read_csv("contract/reference_transforms.csv", dtype=str)[RAW_INPUT]
recs = [{k: v for k, v in r.items() if isinstance(v, str) and v != ""} for r in ref.to_dict("records")]
for r in recs[:5]: s.predict(r)
pr = cProfile.Profile(); pr.enable()
for i in range(100): s.predict(recs[i % len(recs)])
pr.disable()
pstats.Stats(pr).sort_stats("cumulative").print_stats(22)