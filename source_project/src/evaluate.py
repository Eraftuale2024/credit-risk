import json, joblib, numpy as np, pandas as pd, yaml, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, brier_score_loss
from src.train import apply_calibration, grade_matrix, proba, GRADES

_R = {"Northeast": "CT ME MA NH RI VT NJ NY PA", "Midwest": "IL IN MI OH WI IA KS MN MO NE ND SD",
      "South": "DE FL GA MD NC SC VA DC WV AL KY MS TN AR LA OK TX", "West": "AZ CO ID MT NV NM UT WY AK CA HI OR WA"}
REGION = {s: r for r, names in _R.items() for s in names.split()}

def metrics(y, p):
    auc = roc_auc_score(y, p)
    return {"AUC": auc, "Gini": 2 * auc - 1, "KS": ks_2samp(p[y == 1], p[y == 0]).statistic}

def score_all(b, d):
    d = d.copy(); d["grade_num"] = d.grade.astype(str).map({g: i for i, g in enumerate(GRADES)})
    out = pd.DataFrame(index=d.index)
    out["scorecard"] = proba(b["scorecard"]["model"], d[b["scorecard"]["columns"]])
    out["lgbm"] = apply_calibration(proba(b["lgbm"]["model"], d[b["lgbm"]["columns"]]), b["lgbm"]["calibration"])
    out["platform_variant"] = proba(b["platform_variant"]["model"], d[b["platform_variant"]["columns"]])
    out["grade_only"] = proba(b["grade_only"]["model"], grade_matrix(d))
    return out

def decile_curve(p, y, n=10):
    d = pd.DataFrame({"p": np.asarray(p), "y": np.asarray(y)}); d["b"] = pd.qcut(d.p.rank(method="first"), n, labels=False)
    g = d.groupby("b").mean(); return g.p.to_numpy(), g.y.to_numpy()

def policy(p, y, amt, lo, hi, pol):
    ap, de = p < lo, p >= hi; rv = ~ap & ~de
    pnl = amt * (pol["gain"] * (1 - y) - pol["loss"] * y)
    return {"approve %": ap.mean(), "review %": rv.mean(), "decline %": de.mean(),
            "default rate approved": y[ap].mean() if ap.any() else np.nan, "default rate declined": y[de].mean() if de.any() else np.nan,
            "profit approved ($k)": pnl[ap].sum() / 1e3, "profit per $ approved": pnl[ap].sum() / amt[ap].sum() if ap.any() else np.nan}

def seg(key, p, y, thr):
    d = pd.DataFrame({"key": pd.Series(key).astype(str).to_numpy(), "ap": p < thr, "y": y, "p": p}); g = d.groupby("key")
    t = pd.DataFrame({"loans": g.size(), "actual default": g.y.mean(), "mean PD": g.p.mean(), "approve %": g.ap.mean(), "n approved": g.ap.sum()})
    t["default rate approved"] = d[d.ap].groupby("key").y.mean(); return t

def main(cfg_path="configs/config.yaml"):
    cfg = yaml.safe_load(open(cfg_path)); pol = cfg["policy"]
    b = joblib.load(cfg["paths"]["model"]); A = pd.read_parquet(cfg["contract"]["dataset"])
    v = A[A.issue_d > pd.Timestamp(cfg["split"]["fit_cutoff"])].reset_index(drop=True)
    P = score_all(b, v); y = v.defaulted.to_numpy(); amt = v.loan_amnt.to_numpy(float)
    print(f"validation: {len(v)} loans issued {v.issue_d.min():%Y-%m} to {v.issue_d.max():%Y-%m} | default rate {y.mean():.3f}")

    tab = pd.DataFrame({m: metrics(y, P[m].to_numpy()) for m in P}).T
    cal = ["scorecard", "lgbm"]
    tab["Brier"] = pd.Series({m: brier_score_loss(y, P[m]) for m in cal}); tab["mean PD"] = P[cal].mean(); tab["actual"] = y.mean()
    print("\n== Overall (platform_variant PD is uncalibrated: ranking only) =="); print(tab.round(4).to_string())

    def auc_by(key):
        return pd.DataFrame({m: {g: roc_auc_score(y[i], P[m].to_numpy()[i]) for g, i in v.groupby(key).indices.items()} for m in ["scorecard", "lgbm", "grade_only"]})
    byq, byt = auc_by(v.issue_d.dt.to_period("Q").astype(str)), auc_by(v.term)
    print("\n== AUC by quarter =="); print(byq.round(4).to_string()); print("\n== AUC by term =="); print(byt.round(4).to_string())
    ct = pd.DataFrame({"loans": v.groupby("term").size(), "actual default": v.groupby("term").defaulted.mean(),
                       "mean PD lgbm": P["lgbm"].groupby(v.term).mean(), "mean PD scorecard": P["scorecard"].groupby(v.term).mean()})
    print("\n== Calibration by term =="); print(ct.round(4).to_string())

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for m in cal: ax[0].plot(*decile_curve(P[m], y), "o-", label=m)
    for t in (36, 60):
        k = (v.term == t).to_numpy(); ax[1].plot(*decile_curve(P["lgbm"].to_numpy()[k], y[k]), "o-", label=f"lgbm, {t}-month")
    for a, t in zip(ax, ["Calibration by PD decile, 2015", "LightGBM calibration by term, 2015"]):
        a.plot([0, .6], [0, .6], "k--"); a.set_xlabel("mean predicted PD"); a.set_ylabel("actual default rate"); a.set_title(t); a.legend()
    plt.tight_layout(); plt.savefig("reports/calibration_2015.png", dpi=110)

    cases = [(.10, .25), (pol["approve"], pol["decline"]), (.15, .25), (pol["approve"], .20), (pol["approve"], .30)]
    rows = {"approve everyone": policy(P["lgbm"].to_numpy(), y, amt, np.inf, np.inf, pol)}
    for m in ["lgbm", "scorecard"]:
        for lo, hi in cases:
            name = f"{m}: approve <{lo:.1%}, decline >={hi:.0%}" + (" *" if (lo, hi) == (pol["approve"], pol["decline"]) else "")
            rows[name] = policy(P[m].to_numpy(), y, amt, lo, hi, pol)
    pt = pd.DataFrame(rows).T; print("\n== Decision policy on 2015 (* = proposed; others are sensitivity only) =="); print(pt.round(3).to_string())

    p = P["lgbm"].to_numpy(); thr = pol["approve"]
    v["region"] = v.addr_state.map(REGION).fillna("Other")
    v["income band"] = pd.cut(v.annual_inc, [0, 40000, 60000, 90000, np.inf], labels=["<40k", "40-60k", "60-90k", "90k+"], right=False)
    sr, si = seg(v.region, p, y, thr), seg(v["income band"], p, y, thr)
    print("\n== By region (LightGBM, approve below the proposed threshold) =="); print(sr.round(3).to_string())
    print("\n== By income band =="); print(si.round(3).to_string())

    tab.to_csv("reports/validation_metrics.csv"); pt.to_csv("reports/policy_table.csv"); pd.concat([sr, si], keys=["region", "income band"]).to_csv("reports/segments_2015.csv")
    b["metadata"]["thresholds"] = {"approve_below": pol["approve"], "decline_at_or_above": pol["decline"], "break_even_pd": pol["gain"] / (pol["gain"] + pol["loss"]),
        "basis": "approve below break-even PD; decline where expected loss equals one full gain; set from economics, not tuned on 2015"}
    joblib.dump(b, cfg["paths"]["model"]); json.dump(b["metadata"], open(cfg["paths"]["metadata"], "w"), indent=1)
    print("\nthresholds saved to metadata:", b["metadata"]["thresholds"])

if __name__ == "__main__":
    main()