import os, sys, json, hashlib, subprocess, datetime as dt
import numpy as np, pandas as pd, joblib, yaml, lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from src.verify_contract import verify

GRADES = list("ABCDEFG")
proba = lambda m, X: m.predict_proba(X)[:, 1]
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

def apply_calibration(p, cal): return 1 / (1 + np.exp(-(cal["a"] * logit(np.asarray(p)) + cal["b"])))
def grade_matrix(d): return pd.get_dummies(d["grade"].astype(str)).reindex(columns=GRADES, fill_value=0).astype(float)
def git_commit():
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return "unknown"

def load_and_validate(cfg):
    c = cfg["contract"]; spec = json.load(open(c["spec"])); ct = joblib.load(c["transform"])
    assert spec["contract_version"] == ct.contract_version == c["version"], \
        f"contract version mismatch: config {c['version']}, spec {spec['contract_version']}, transform {ct.contract_version}"
    assert verify(c["dir"]) == 0, "transform does not reproduce reference_transforms.csv"
    A = pd.read_parquet(c["dataset"])
    banned = set(spec["excluded_columns"]["post_issue_servicing"]) | set(spec["excluded_columns"]["outcome_derived"])
    assert not banned & set(A.columns), f"excluded columns in training data: {banned & set(A.columns)}"
    assert set(A.defaulted.unique()) <= {0, 1}, "label must be 0/1"
    sc = [f"woe_{v}" for v in spec["woe"]["scorecard_vars"] if v not in cfg["features"]["scorecard_exclude"]]
    A["grade_num"] = A.grade.astype(str).map({g: i for i, g in enumerate(GRADES)})
    f = cfg["features"]
    sets = {"raw": f["raw"], "factors": f["factors"], "raw+factors": f["raw"] + f["factors"],
            "raw+platform (comparison only)": f["raw"] + ["int_rate", "grade_num"]}
    must = set(sc) | set(f["factors"]) | {"woe_home_ownership", "woe_verification_status"}
    assert A[list(must)].notna().all().all(), "gaps in WoE or factor columns"
    assert all(c_ in A.columns for cols in sets.values() for c_ in cols), "feature missing from the dataset"
    return spec, A, sc, sets

def cv_auc(fit, cols, df, y, q, test_q):
    X = df[cols]; oof = pd.Series(np.nan, index=X.index); aucs = []
    for tq in test_q:
        a, b = (q < tq).to_numpy(), (q == tq).to_numpy()
        p = proba(fit(X[a], y[a]), X[b]); oof[b] = p; aucs.append(roc_auc_score(y[b], p))
    return float(np.mean(aucs)), oof

def main(cfg_path="configs/config.yaml"):
    cfg = yaml.safe_load(open(cfg_path)); seed = cfg["seed"]; np.random.seed(seed)
    spec, A, sc_cols, sets = load_and_validate(cfg)
    run = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    def log(**r):
        p = cfg["paths"]["log"]; os.makedirs(os.path.dirname(p), exist_ok=True)
        pd.DataFrame([{"run": run, "seed": seed, **r}]).to_csv(p, mode="a", header=not os.path.exists(p), index=False)

    A = A.sort_values("issue_d").reset_index(drop=True)
    tr = A[A.issue_d <= pd.Timestamp(cfg["split"]["fit_cutoff"])].reset_index(drop=True)
    y = tr.defaulted.to_numpy(); q = tr.issue_d.dt.to_period("Q"); test_q = [pd.Period(p) for p in cfg["split"]["cv_test_quarters"]]
    pos_w = float((1 - y.mean()) / y.mean())
    print(f"train: {len(tr)} loans up to {tr.issue_d.max():%Y-%m-%d} | default rate {y.mean():.3f} | scale_pos_weight {pos_w:.2f}")

    for c, d in cfg["monotone"].items():
        rho = spearmanr(tr[c], y, nan_policy="omit")[0]
        print(f"monotone check {c:22} constraint {d:+d} | Spearman with default {rho:+.3f}")
        assert np.sign(rho) == d, f"constraint on {c} disagrees with the training data"

    def lgb_fit(cols, params):
        def fit(X, y_):
            return lgb.LGBMClassifier(objective="binary", scale_pos_weight=pos_w, random_state=seed, deterministic=True, force_row_wise=True,
                n_jobs=1, verbose=-1, monotone_constraints=[cfg["monotone"].get(c, 0) for c in cols], **cfg["lgbm"]["fixed"], **params).fit(X, y_)
        return fit

    best_C, best_a = None, -1
    for C in cfg["scorecard"]["C"]:
        a, _ = cv_auc(lambda X, y_: LogisticRegression(C=C, max_iter=2000).fit(X, y_), sc_cols, tr, y, q, test_q)
        log(stage="tune", model="scorecard", feature_set="woe_scorecard", params=json.dumps({"C": C}), cv_auc=round(a, 4)); print(f"scorecard C={C}: CV AUC {a:.4f}")
        if a > best_a: best_C, best_a = C, a

    grid = cfg["lgbm"]["grid"]; res = []
    for p in grid:
        a, _ = cv_auc(lgb_fit(sets["raw+factors"], p), sets["raw+factors"], tr, y, q, test_q); res.append(a)
        log(stage="tune", model="lgbm", feature_set="raw+factors", params=json.dumps(p), cv_auc=round(a, 4)); print(f"lgbm {p}: CV AUC {a:.4f}")
    best_p = grid[int(np.argmax(res))]

    scores, oofs = {}, {}
    for name, cols in sets.items():
        scores[name], oofs[name] = cv_auc(lgb_fit(cols, best_p), cols, tr, y, q, test_q)
        log(stage="ablation", model="lgbm", feature_set=name, params=json.dumps(best_p), cv_auc=round(scores[name], 4)); print(f"ablation {name:34} CV AUC {scores[name]:.4f}")
    main_name = max((n for n in scores if "comparison" not in n), key=scores.get); print("main feature set:", main_name)

    oof = oofs[main_name]; m = oof.notna().to_numpy()
    pl = LogisticRegression(C=1e6, max_iter=1000).fit(logit(oof[m].to_numpy()).reshape(-1, 1), y[m])
    cal = {"a": float(pl.coef_[0, 0]), "b": float(pl.intercept_[0])}
    print(f"calibration on {m.sum()} out-of-fold loans | mean raw {oof[m].mean():.3f} -> calibrated {apply_calibration(oof[m], cal).mean():.3f} | actual {y[m].mean():.3f}")

    sc_model = LogisticRegression(C=best_C, max_iter=2000).fit(tr[sc_cols], y)
    print("scorecard coefficients above 0 (expected none, WoE is higher for safer groups):", int((sc_model.coef_[0] > 0).sum()), "of", len(sc_cols))
    main_cols = sets[main_name]; lgb_main = lgb_fit(main_cols, best_p)(tr[main_cols], y)
    plat_cols = sets["raw+platform (comparison only)"]; lgb_plat = lgb_fit(plat_cols, best_p)(tr[plat_cols], y)
    g_model = LogisticRegression(max_iter=2000).fit(grade_matrix(tr), y)

    meta = {"git_commit": git_commit(), "training_data_hash": hashlib.sha256(open(cfg["contract"]["dataset"], "rb").read()).hexdigest()[:16],
            "training_date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "contract_version": spec["contract_version"],
            "fit_cutoff": cfg["split"]["fit_cutoff"], "seed": seed, "n_train": int(len(tr)), "train_default_rate": round(float(y.mean()), 4),
            "main_model": "lgbm", "main_feature_set": main_name, "lgbm_params": best_p, "scorecard_C": best_C, "cv_auc": {k: round(v, 4) for k, v in scores.items()}, "thresholds": None}
    bundle = {"metadata": meta, "scorecard": {"model": sc_model, "columns": sc_cols},
              "lgbm": {"model": lgb_main, "columns": main_cols, "calibration": cal},
              "platform_variant": {"model": lgb_plat, "columns": plat_cols}, "grade_only": {"model": g_model}}
    os.makedirs(os.path.dirname(cfg["paths"]["model"]), exist_ok=True)
    joblib.dump(bundle, cfg["paths"]["model"]); json.dump(meta, open(cfg["paths"]["metadata"], "w"), indent=1)
    log(stage="final", model="lgbm", feature_set=main_name, params=json.dumps(best_p), cv_auc=round(scores[main_name], 4))
    print("saved:", cfg["paths"]["model"], "| git commit:", meta["git_commit"][:10], "| data hash:", meta["training_data_hash"])

if __name__ == "__main__":
    main()