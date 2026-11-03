import joblib, numpy as np, pandas as pd, yaml
from src.train import apply_calibration, proba
from src.evaluate import REGION

def psi(a, b, bins=10):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    if a.nunique() <= 10:
        k = sorted(set(a) | set(b))
        pa = a.value_counts(normalize=True).reindex(k, fill_value=0).to_numpy()
        pb = b.value_counts(normalize=True).reindex(k, fill_value=0).to_numpy()
    else:
        e = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1))); e[0], e[-1] = -np.inf, np.inf
        pa, pb = np.histogram(a, e)[0] / len(a), np.histogram(b, e)[0] / len(b)
    pa, pb = np.clip(pa, 1e-4, None), np.clip(pb, 1e-4, None)
    return float(((pb - pa) * np.log(pb / pa)).sum())

def raw_rows(ids, D="data/"):
    app = pd.read_csv(D + "applications.csv", low_memory=False).drop_duplicates("id", keep="first")
    bur = pd.read_csv(D + "credit_bureau.csv", low_memory=False).drop(columns=["bureau_recovery_flag"])
    raw = app.merge(bur, on="id", how="inner", validate="one_to_one")
    return raw[raw.id.isin(ids)].reset_index(drop=True)

decide = lambda p, pol: np.where(p < pol["approve"], 0, np.where(p >= pol["decline"], 2, 1))

def main(cfg_path="configs/config.yaml"):
    cfg = yaml.safe_load(open(cfg_path)); pol = cfg["policy"]; cut = pd.Timestamp(cfg["split"]["fit_cutoff"])
    b = joblib.load(cfg["paths"]["model"]); ct = joblib.load(cfg["contract"]["transform"])
    A = pd.read_parquet(cfg["contract"]["dataset"])
    tr, v = A[A.issue_d <= cut].reset_index(drop=True), A[A.issue_d > cut].reset_index(drop=True)
    cols = b["lgbm"]["columns"]
    score = lambda F: apply_calibration(proba(b["lgbm"]["model"], F[cols]), b["lgbm"]["calibration"])
    y = v.defaulted.to_numpy(); p = score(v); v["pd_hat"] = p
    v["quarter"] = v.issue_d.dt.to_period("Q").astype(str); v["region"] = v.addr_state.map(REGION).fillna("Other"); v["is60"] = (v.term == 60).astype(int)

    print("== 1. Calibration by segment, 2015 (z tests actual against predicted; n >= 300) ==")
    seg = {"term": v.term, "grade": v.grade, "purpose": v.purpose_g, "home ownership": v.home_ownership, "verification": v.verification_status,
           "job group": v.job_group, "employment length": v.emp_length_g, "mortgage account": v.mort_acc > 0, "quarter": v.quarter, "region": v.region,
           "application type": v.application_type,
           "income band": pd.cut(v.annual_inc, [0, 40000, 60000, 90000, np.inf], labels=["<40k", "40-60k", "60-90k", "90k+"], right=False),
           "fico band": pd.cut(v.fico_mid, [0, 680, 700, 720, 760, 900], labels=["<680", "680-700", "700-720", "720-760", "760+"], right=False),
           "dti band": pd.cut(v.dti, [-1, 10, 20, 30, 101], labels=["<10", "10-20", "20-30", "30+"])}
    rows = []
    for name, s in seg.items():
        d = pd.DataFrame({"k": s.astype(str).to_numpy(), "p": p, "y": y}).groupby("k").agg(n=("y", "size"), mean_pd=("p", "mean"), actual=("y", "mean"))
        d["z"] = (d.actual - d.mean_pd) / np.sqrt(d.mean_pd * (1 - d.mean_pd) / d.n); d.insert(0, "segment", name); rows.append(d.reset_index())
    cal = pd.concat(rows, ignore_index=True); cal["gap"] = cal.actual - cal.mean_pd; big = cal[cal.n >= 300]
    print(f"{len(big)} segments with >= 300 loans | {int((big.z.abs() >= 3).sum())} with |z| >= 3 | {int((big.z.abs() >= 2).sum())} with |z| >= 2 (about {0.05 * len(big):.0f} expected by chance)")
    print(big.reindex(big.z.abs().sort_values(ascending=False).index).head(12).round(3).to_string(index=False))
    cal.to_csv("reports/calibration_segments.csv", index=False)

    print("\n== 2. Most confident errors, 2015 ==")
    ec = ["id", "term", "grade", "purpose", "loan_amnt", "annual_inc", "dti", "fico_mid", "pd_hat"]
    print("defaults the model was surest were safe:"); print(v[v.defaulted == 1].nsmallest(10, "pd_hat")[ec].round(3).to_string(index=False))
    print("repaid loans the model was surest would default:"); print(v[v.defaulted == 0].nlargest(10, "pd_hat")[ec].round(3).to_string(index=False))
    ap, de = p < pol["approve"], p >= pol["decline"]
    print(f"\ndefaults approved: {int((ap & (y == 1)).sum())} of {int(y.sum())} ({(ap & (y == 1)).sum() / y.sum():.1%}) | repaid loans declined: {int((de & (y == 0)).sum())} of {int((y == 0).sum())} ({(de & (y == 0)).sum() / (y == 0).sum():.1%})")
    pat = ["is60", "dti", "loan_to_income", "fico_mid", "acc_open_past_24mths", "annual_inc", "recent_activity", "utilization_stress"]
    g = v[ap].groupby("defaulted")[pat].mean().T; g.columns = ["repaid", "defaulted"]; g["gap in std"] = (g.defaulted - g.repaid) / v[pat].std()
    print("\napproved loans: defaulted vs repaid"); print(g.round(3).to_string())

    print("\n== 3. Robustness: raw rows -> transform -> model, one input damaged at a time (1,000 raw 2015 rows) ==")
    base_raw = raw_rows(set(v.id)).sample(1000, random_state=0).reset_index(drop=True)
    base = score(ct.transform(base_raw)); ref = v.set_index("id").pd_hat.reindex(base_raw.id).to_numpy()
    print("pipeline check, max |PD - evaluation PD|:", float(np.abs(base - ref).max()))
    def mod(**kw):
        def f(r):
            r = r.copy()
            for c, val in kw.items(): r[c] = val(r[c]) if callable(val) else val
            return r
        return f
    tests = {"annual_inc missing": mod(annual_inc=np.nan), "annual_inc entered in thousands (/1000)": mod(annual_inc=lambda s: s / 1000),
             "annual_inc = 50,000,000 (above spec)": mod(annual_inc=5e7), "dti = 150 (above spec)": mod(dti=150.0), "revol_util = 300": mod(revol_util=300.0),
             "bc_util missing": mod(bc_util=np.nan), "fico 300-304 (below training range)": mod(fico_range_low=300.0, fico_range_high=304.0),
             "6 balance and limit fields missing": mod(**{c: np.nan for c in ["revol_bal", "tot_cur_bal", "avg_cur_bal", "total_rev_hi_lim", "tot_hi_cred_lim", "mort_acc"]}),
             "loan_amnt x10 (transform repairs it)": mod(loan_amnt=lambda s: s * 10), "joint application": mod(application_type="Joint App")}
    out = {}
    for name, f in tests.items():
        try:
            q = score(ct.transform(f(base_raw))); ok = ~np.isnan(q)
            out[name] = {"NaN scores %": 100 * (~ok).mean(), "mean PD change": np.nanmean(q - base), "max |change|": np.nanmax(np.abs(q - base)),
                         "decision changed %": 100 * ((decide(q, pol) != decide(base, pol)) & ok).mean()}
        except Exception as e: out[name] = {"error": repr(e)[:90]}
    rb = pd.DataFrame(out).T; print(rb.round(4).to_string()); rb.to_csv("reports/robustness.csv")

    print("\n== 4. Holdout 2016 ==")
    h = pd.read_csv("data/holdout_applications.csv", low_memory=False); Hf = ct.transform(h); Hf["pd_hat"] = score(Hf)
    assert len(Hf) == 5000 and Hf["id"].is_unique and Hf["pd_hat"].notna().all(), "holdout scoring problem"
    pd.DataFrame({"id": Hf["id"].to_numpy(), "pd": Hf["pd_hat"].to_numpy()}).to_csv("holdout_pd.csv", index=False)
    d16 = decide(Hf["pd_hat"].to_numpy(), pol)
    print(f"scored {len(Hf)} loans -> holdout_pd.csv | mean PD 2016 {Hf.pd_hat.mean():.3f} vs 2015 {p.mean():.3f} | approve {np.mean(d16 == 0):.1%}, review {np.mean(d16 == 1):.1%}, decline {np.mean(d16 == 2):.1%}")

    print("\n== 5. Drift: PSI, 2015 -> 2016 (and 2014 -> 2015 for context) ==")
    imp = pd.Series(b["lgbm"]["model"].booster_.feature_importance(importance_type="gain"), index=cols).sort_values(ascending=False); top = imp.head(10).index.tolist()
    ps = pd.DataFrame({"gain share": (imp[top] / imp.sum()).round(3), "PSI 2014->2015": [psi(tr[c], v[c]) for c in top], "PSI 2015->2016": [psi(v[c], Hf[c]) for c in top]})
    ps.loc["PD (calibrated)"] = [np.nan, np.nan, psi(p, Hf["pd_hat"])]; print(ps.round(3).to_string()); ps.to_csv("reports/drift_psi.csv")
    dd = {"60-month share": lambda d: (d.term == 60).mean(), "joint applications": lambda d: (d.application_type == "Joint App").mean(),
          "dti above 100": lambda d: (d.dti > 100).mean(), "never delinquent": lambda d: d.never_delinquent.mean(),
          "bc_util missing": lambda d: d.bc_util_missing.mean(), "emp_length missing": lambda d: (d.emp_length_g == "MISSING").mean()}
    print("\ninput drift"); print(pd.DataFrame({k: [f(tr), f(v), f(Hf)] for k, f in dd.items()}, index=["2014", "2015", "2016"]).T.round(4).to_string())

if __name__ == "__main__":
    main()