import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from src.treat import treat
from src.features import add_features

CONTRACT_VERSION = "1.0.0"
FIT_CUTOFF = "2014-12-31"
STATE_NAMES = {"CALIFORNIA": "CA", "TEXAS": "TX", "NEW YORK": "NY", "FLORIDA": "FL", "GEORGIA": "GA", "PENNSYLVANIA": "PA",
               "NEW JERSEY": "NJ", "OHIO": "OH", "ILLINOIS": "IL", "NORTH CAROLINA": "NC", "MICHIGAN": "MI", "VIRGINIA": "VA"}
NUMERIC = ["loan_amnt", "int_rate", "installment", "annual_inc", "dti", "fico_range_low", "fico_range_high", "inq_last_6mths",
           "delinq_2yrs", "mths_since_last_delinq", "pub_rec", "pub_rec_bankruptcies", "open_acc", "total_acc", "revol_bal",
           "revol_util", "bc_util", "tot_cur_bal", "avg_cur_bal", "total_rev_hi_lim", "tot_hi_cred_lim", "mort_acc",
           "acc_open_past_24mths", "num_tl_op_past_12m"]
CATEGORICAL = ["grade", "sub_grade", "purpose", "home_ownership", "verification_status", "application_type", "emp_length"]
OUT_RAW = ["id", "issue_d", "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade", "purpose", "application_type",
           "emp_length", "home_ownership", "annual_inc", "verification_status", "dti", "addr_state", "earliest_cr_line",
           "fico_range_low", "fico_range_high", "inq_last_6mths", "delinq_2yrs", "mths_since_last_delinq", "pub_rec",
           "pub_rec_bankruptcies", "open_acc", "total_acc", "revol_bal", "revol_util", "bc_util", "tot_cur_bal", "avg_cur_bal",
           "total_rev_hi_lim", "tot_hi_cred_lim", "mort_acc", "acc_open_past_24mths", "num_tl_op_past_12m", "cr_line_invalid"]
RAW_INPUT = [c for c in OUT_RAW if c != "cr_line_invalid"] + ["emp_title"]


def parse_date(s):
    s = s.astype("string").str.strip()
    out = pd.to_datetime(s, format="%b-%Y", errors="coerce")
    for f in ["%Y-%m-%d", "%m/%Y"]:
        out = out.fillna(pd.to_datetime(s, format=f, errors="coerce"))
    return out


def to_plain(df):
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d").fillna("")
    return out


class ContractTransform:
    def __init__(self, caps, woe_spec, factor_spec):
        self.caps, self.woe_spec, self.factor_spec = caps, woe_spec, factor_spec
        self.contract_version, self.fit_cutoff, self.pca = CONTRACT_VERSION, FIT_CUTOFF, None

    @classmethod
    def build(cls, raw_fit, caps, woe_spec, factor_spec, n_pc=7):
        obj = cls(caps, woe_spec, factor_spec)
        T = treat(obj.clean(raw_fit), caps)
        mu, sd = T.mean().to_numpy(), T.std(ddof=0).to_numpy()
        p = PCA(n_components=n_pc, svd_solver="full").fit((T.to_numpy() - mu) / sd)
        obj.pca = {"columns": list(T.columns), "mean": mu, "scale": sd, "components": p.components_, "var_ratio": p.explained_variance_ratio_}
        return obj

    def clean(self, raw):
        d = raw.copy()
        d["issue_d"], d["earliest_cr_line"] = parse_date(d["issue_d"]), parse_date(d["earliest_cr_line"])
        d["term"] = pd.to_numeric(d["term"].astype("string").str.extract(r"(\d+)")[0], errors="coerce")
        for c in NUMERIC: d[c] = pd.to_numeric(d[c], errors="coerce")
        for c in CATEGORICAL: d[c] = d[c].astype("string").str.strip()
        d["addr_state"] = d["addr_state"].astype("string").str.strip().str.upper().replace(STATE_NAMES)
        m = (d.annual_inc > 0) & (d.annual_inc < 1000); d.loc[m, "annual_inc"] *= 1000
        r = d.int_rate / 1200; implied = d.installment * (1 - (1 + r) ** -d.term) / r
        d.loc[(d.loan_amnt / implied - 10).abs() < 0.05, "loan_amnt"] /= 10
        sw = d.fico_range_low > d.fico_range_high
        d.loc[sw, ["fico_range_low", "fico_range_high"]] = d.loc[sw, ["fico_range_high", "fico_range_low"]].to_numpy()
        for c in ["revol_util", "bc_util"]: d[c] = d[c].abs()
        u = d["revol_util"]
        if u.notna().sum() >= 100 and u.quantile(0.99) <= 1.5:
            d["revol_util"] = u * 100
        bad = d.earliest_cr_line > d.issue_d
        d["cr_line_invalid"] = bad.astype(int); d.loc[bad, "earliest_cr_line"] = pd.NaT
        return d

    def _job_group(self, title):
        t = title.astype("string").str.lower().str.replace(r"[^a-z ]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
        rules = self.woe_spec["job_rules"]
        g = np.select([t.fillna("").str.contains(p, regex=True).to_numpy(dtype=bool) for _, p in rules], [n for n, _ in rules], default="other")
        g = pd.Series(np.where(t.isna().to_numpy(), "unknown", g), index=title.index)
        return g.where(~g.isin(self.woe_spec["job_merged"]), "other")

    def groups(self, c):
        ws, g = self.woe_spec, pd.DataFrame(index=c.index)
        g["job_group"] = self._job_group(c["emp_title"])
        num = c["emp_length"].map(ws["emp_length_map"]).astype(float)
        g["emp_length_g"] = pd.cut(num, [-1, 0, 9, 10], labels=["<1y", "1-9y", "10+y"]).astype(object).where(num.notna(), "MISSING")
        g["purpose_g"] = c["purpose"].where(~c["purpose"].isin(ws["purpose_merged"]), "other")
        g["state_g"] = c["addr_state"].where(~c["addr_state"].isin(ws["states_merged"]), "OTHER_STATE")
        for col in ["inq_last_6mths", "mort_acc"]:
            g[col + "_g"] = c[col].clip(upper=3).round().astype("Int64").astype(str)
        return g

    def _cat_key(self, v, c, g):
        if v == "term": return c["term"].astype("Int64").astype(str)
        if v in g.columns: return g[v]
        return c[v].astype(object).where(c[v].notna(), "MISSING").astype(str)

    def woe(self, c, F, g):
        ws, out = self.woe_spec, {}
        for v in ws["scorecard_vars"]:
            if v in ws["numeric_bin_edges"]:
                x = F[v] if v in F.columns else (c[v].fillna(self.caps["missing"][v]["fill"]) if v in ("revol_util", "bc_util") else c[v])
                k = np.searchsorted(np.array(ws["numeric_bin_edges"][v]), x.to_numpy(dtype=float), side="left")
                key = pd.Series(np.where(x.isna().to_numpy(), "MISSING", [f"{i:02d}" for i in k]), index=c.index)
            else:
                key = self._cat_key(v, c, g)
            out["woe_" + v] = key.map(ws["woe"][v]).astype(float)
        return pd.DataFrame(out, index=c.index)

    def transform(self, raw, cleaned=None):
        c = self.clean(raw) if cleaned is None else cleaned
        T = treat(c, self.caps)
        F = add_features(c, self.caps["features"]["credit_history_months_fill"], self.caps["annual_inc"]["cap_p99.5"])
        fs, p = self.factor_spec, self.pca
        Z = ((T[fs["fields"]] - np.array(fs["mean"])) / np.array(fs["std"])).to_numpy()
        FS = pd.DataFrame(Z @ np.array(fs["weights"]), columns=[f["name"] for f in fs["factors"]], index=c.index)
        PC = pd.DataFrame(((T[p["columns"]].to_numpy() - p["mean"]) / p["scale"]) @ p["components"].T,
                          columns=[f"pc{i + 1}" for i in range(len(p["components"]))], index=c.index)
        G = self.groups(c)
        return pd.concat([c[OUT_RAW], F, G, FS, PC, self.woe(c, F, G)], axis=1)