import sys, json, argparse
import numpy as np, pandas as pd, joblib
from src.contract_transform import RAW_INPUT

FICO_TRAIN_MIN = 660
MIN_IMPACT = 0.02
REASONS = {
    "term": (["term"], "Longer loan term"),
    "loan_size": (["loan_to_income", "loan_amnt"], "Loan amount is high relative to income"),
    "dti": (["dti", "debt_burden"], "High debt payments relative to income"),
    "utilization": (["revol_util", "bc_util", "utilization_stress"], "High use of available credit"),
    "recent_credit": (["acc_open_past_24mths", "num_tl_op_past_12m", "inq_last_6mths", "recent_activity"],
                      "Many recently opened credit accounts or inquiries"),
    "credit_score": (["fico_mid"], "Lower credit score"),
    "income": (["annual_inc"], "Lower income"),
    "credit_size": (["credit_footprint", "tot_cur_bal", "tot_hi_cred_lim", "avg_cur_bal", "mort_acc"],
                    "Limited overall credit relationship (low total balances and limits)"),
}

# training medians (loans up to 2014-12-31); a reason is shown only if the applicant is on the unfavorable side
REFERENCE = {"fico_mid": 687, "annual_inc": 65000, "dti": 17.23, "loan_to_income": 0.21, "acc_open_past_24mths": 4,
             "num_tl_op_past_12m": 2, "tot_cur_bal": 80428, "tot_hi_cred_lim": 108769}


def _unfavorable(key, r, ref):
    g = lambda c: r[c] if c in r.index else np.nan
    return bool({"term": g("term") == 60, "loan_size": g("loan_to_income") > ref["loan_to_income"], "dti": g("dti") > ref["dti"],
                 "utilization": g("revol_util") > ref["revol_util"] or g("bc_util") > ref["bc_util"],
                 "recent_credit": g("acc_open_past_24mths") > ref["acc_open_past_24mths"] or g("num_tl_op_past_12m") > ref["num_tl_op_past_12m"],
                 "credit_score": g("fico_mid") < ref["fico_mid"], "income": g("annual_inc") < ref["annual_inc"],
                 "credit_size": g("tot_hi_cred_lim") < ref["tot_hi_cred_lim"] or g("tot_cur_bal") < ref["tot_cur_bal"]}[key])


class ContractVersionError(RuntimeError):
    pass


class ValidationError(ValueError):
    def __init__(self, errors):
        super().__init__("; ".join(errors)); self.errors = errors


def _blank(v):
    if v is None or v is pd.NA: return True
    if isinstance(v, float) and np.isnan(v): return True
    return isinstance(v, str) and v.strip() == ""


def _calibrate(p, cal):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(cal["a"] * np.log(p / (1 - p)) + cal["b"])))


class Scorer:
    def __init__(self, model_path="models/pd_model.joblib", contract_dir="contract", expected_version=None):
        self.bundle = joblib.load(model_path)
        self.ct = joblib.load(f"{contract_dir}/transform.joblib")
        self.spec = json.load(open(f"{contract_dir}/data_spec.json"))
        self.meta = self.bundle["metadata"]
        want = expected_version or self.spec["contract_version"]
        found = {"model": self.meta["contract_version"], "spec": self.spec["contract_version"], "transform": self.ct.contract_version}
        if any(v != want for v in found.values()):
            raise ContractVersionError(f"expected contract version {want}, found {found}")
        self.thresholds = self.meta.get("thresholds")
        if not self.thresholds:
            raise RuntimeError("model metadata has no decision thresholds; run src.evaluate first")
        if self.bundle["lgbm"]["calibration"]["a"] <= 0:
            raise RuntimeError("calibration slope must be positive")
        self.fields = {f["name"]: f for f in self.spec["fields"]}
        self.ref = dict(REFERENCE, revol_util=self.spec["caps"]["missing"]["revol_util"]["fill"],
                        bc_util=self.spec["caps"]["missing"]["bc_util"]["fill"])
        self.required = [n for n, f in self.fields.items() if f["missing_strategy"] == "reject"]
        self.model_version = f"pd-lgbm | contract {want} | commit {self.meta['git_commit'][:7]} | trained {self.meta['training_date'][:10]}"

    def validate(self, record):
        flags, warns = [], []
        ignored = sorted(k for k in record if k not in RAW_INPUT)
        if ignored:
            warns.append(f"ignored columns (never used by the model): {ignored}")
        rec = {k: record.get(k) for k in RAW_INPUT}
        errs = [f"{k}: required field is missing" for k in self.required if _blank(rec[k])]
        if errs:
            raise ValidationError(errs)
        df = pd.DataFrame([rec], columns=RAW_INPUT)
        c = self.ct.clean(df)
        if pd.isna(c.issue_d.iloc[0]):
            errs.append("issue_d: cannot be read as a date")
        elif not (pd.Timestamp(self.fields["issue_d"]["min"]) <= c.issue_d.iloc[0] <= pd.Timestamp(self.fields["issue_d"]["max"])):
            warns.append("issue_d is outside the data window of the training contract")
        if not _blank(rec["earliest_cr_line"]) and pd.isna(c.earliest_cr_line.iloc[0]) and c.cr_line_invalid.iloc[0] == 0:
            errs.append("earliest_cr_line: cannot be read as a date")
        capped_dti = False
        for name, f in self.fields.items():
            if f["type"] in ("float", "int"):
                v = c[name].iloc[0]
                given = not _blank(rec[name])
                if given and pd.isna(v):
                    errs.append(f"{name}: cannot be read as a number")
                    continue
                if pd.isna(v):
                    continue
                if name == "dti":
                    if v < 0:
                        errs.append("dti: must not be negative")
                    elif v > 100:
                        flags.append("dti above 100 (capped at 100)")
                        capped_dti = True
                    continue
                if (f["min"] is not None and v < f["min"]) or (f["max"] is not None and v > f["max"]):
                    errs.append(f"{name}: {v:g} outside the valid range {f['min']} to {f['max']}")
                elif f["type"] == "int" and v != round(v):
                    errs.append(f"{name}: must be a whole number")
            elif "allowed_values" in f:
                v = c[name].iloc[0]
                if not _blank(v) and v not in f["allowed_values"]:
                    errs.append(f"{name}: '{v}' is not an allowed value")
        if c.annual_inc.iloc[0] <= 0:
            errs.append("annual_inc: must be above 0")
        if c.term.iloc[0] not in (36, 60):
            errs.append("term: must be 36 or 60 months")
        st = c.addr_state.iloc[0]
        if pd.isna(st) or not (isinstance(st, str) and len(st) == 2 and st.isalpha() and st.isupper()):
            errs.append("addr_state: must be a two-letter state code")
        if not errs:
            r = c.int_rate.iloc[0] / 1200
            n = c.term.iloc[0]
            pmt = c.loan_amnt.iloc[0] * r / (1 - (1 + r) ** -n)
            if abs(pmt - c.installment.iloc[0]) > 1:
                errs.append(f"installment {c.installment.iloc[0]:.2f} does not match loan_amnt, int_rate and term (expected about {pmt:.2f})")
        if errs:
            raise ValidationError(errs)

        raw_inc, raw_amt = pd.to_numeric(df["annual_inc"]).iloc[0], pd.to_numeric(df["loan_amnt"]).iloc[0]
        if 0 < raw_inc < 1000:
            warns.append("annual_inc below 1,000 was treated as thousands")
        if abs(raw_amt - c.loan_amnt.iloc[0]) > 0.01:
            warns.append(f"loan_amnt {raw_amt:g} was repaired to {c.loan_amnt.iloc[0]:g}")
        if c.cr_line_invalid.iloc[0] == 1:
            warns.append("earliest_cr_line is after the issue date and was ignored")
        if c.application_type.iloc[0] == "Joint App":
            flags.append("joint application (none in training data)")
        if c.fico_range_low.iloc[0] < FICO_TRAIN_MIN:
            flags.append(f"FICO below {FICO_TRAIN_MIN}, the lowest in training")
        u, lim, bal = c.revol_util.iloc[0], c.total_rev_hi_lim.iloc[0], c.revol_bal.iloc[0]
        if not pd.isna(u) and u <= 1.5 and lim > 0:
            imp = bal / lim * 100
            if imp > 5 and abs(u * 100 - imp) < abs(u - imp):
                flags.append(f"revol_util {u:g} looks like a fraction of 1 (balance/limit gives {imp:.0f}%); supply a percentage")
        if capped_dti:
            df["dti"] = df["dti"].astype(object)
            df.loc[0, "dti"] = 100.0
            c = self.ct.clean(df)
        return df, c, flags, warns

    def predict(self, record):
        df, c, flags, warns = self.validate(record)
        F = self.ct.transform(df, cleaned=c)
        cols = self.bundle["lgbm"]["columns"]
        X = F[cols]
        m = self.bundle["lgbm"]["model"]
        p = float(_calibrate(m.predict_proba(X)[:, 1], self.bundle["lgbm"]["calibration"])[0])
        if not np.isfinite(p):
            raise RuntimeError("model returned a non-finite PD")
        contrib = pd.Series(m.predict(X, pred_contrib=True)[0][:-1], index=cols)
        reasons = []
        for key, (feats, text) in REASONS.items():
            s = float(contrib[[f for f in feats if f in contrib.index]].sum())
            if s > MIN_IMPACT and _unfavorable(key, F.iloc[0], self.ref):
                reasons.append({"code": key, "reason": text, "impact": round(s, 3)})
        reasons = sorted(reasons, key=lambda r: -r["impact"])[:4]
        th = self.thresholds
        decision = "approve" if p < th["approve_below"] else "decline" if p >= th["decline_at_or_above"] else "review"
        basis = f"PD against thresholds {th['approve_below']} / {th['decline_at_or_above']}"
        if flags:
            decision, basis = "review", "out_of_domain: " + "; ".join(flags)
        return {"pd": round(p, 4), "decision": decision, "decision_basis": basis, "model_version": self.model_version,
                "contract_version": self.meta["contract_version"], "reason_codes": reasons, "out_of_domain": bool(flags),
                "flags": flags, "warnings": warns}


_SCORER = None


def predict(record, **kw):
    global _SCORER
    if _SCORER is None:
        _SCORER = Scorer(**kw)
    return _SCORER.predict(record)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score one raw application (JSON) with the PD model")
    ap.add_argument("--input", help="JSON file; reads stdin if omitted")
    ap.add_argument("--model", default="models/pd_model.joblib")
    ap.add_argument("--contract", default="contract")
    a = ap.parse_args(argv)
    record = json.load(open(a.input) if a.input else sys.stdin)
    try:
        out = Scorer(a.model, a.contract).predict(record)
    except ValidationError as e:
        print(json.dumps({"error": "invalid input", "details": e.errors}, indent=1))
        return 2
    except ContractVersionError as e:
        print(json.dumps({"error": "contract version mismatch", "details": str(e)}, indent=1))
        return 3
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())