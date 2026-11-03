import time, numpy as np, pandas as pd
from src.predict import Scorer, ValidationError
from src.contract_transform import RAW_INPUT

def main(n=300):
    t0 = time.perf_counter(); s = Scorer(); load = time.perf_counter() - t0
    ref = pd.read_csv("contract/reference_transforms.csv", dtype=str)[RAW_INPUT]
    recs = [{k: v for k, v in r.items() if isinstance(v, str) and v != ""} for r in ref.to_dict("records")]
    for r in recs[:5]:
        try: s.predict(r)
        except ValidationError: pass
    ms, rejected = [], 0
    for i in range(n):
        t = time.perf_counter()
        try: s.predict(recs[i % len(recs)])
        except ValidationError: rejected += 1; continue
        ms.append((time.perf_counter() - t) * 1000)
    q = np.percentile(ms, [50, 95, 99])
    text = (f"Latency benchmark: {len(ms)} single-record predictions through the full path (validation, transform, model, reason codes)\n"
            f"startup (load model, transform, spec): {load:.2f} s\n"
            f"p50 {q[0]:.1f} ms | p95 {q[1]:.1f} ms | p99 {q[2]:.1f} ms | max {max(ms):.1f} ms | mean {np.mean(ms):.1f} ms\n"
            f"requirement: under 100 ms per application | rejected by validation: {rejected}\n")
    print(text); open("reports/latency_benchmark.txt", "w").write(text)

if __name__ == "__main__":
    main()