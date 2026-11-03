import json
import gradio as gr
from src.predict import Scorer, ValidationError
from src.demo import EXAMPLES, to_record

scorer = Scorer()
spec = json.load(open("contract/data_spec.json"))
FIELDS = {f["name"]: f for f in spec["fields"]}
GROUPS = {
    "Loan": ["loan_amnt", "term", "int_rate", "grade", "sub_grade", "purpose", "application_type", "issue_d"],
    "Applicant": ["annual_inc", "emp_length", "emp_title", "home_ownership", "verification_status", "dti", "addr_state", "earliest_cr_line"],
    "Credit bureau": ["fico_range_low", "fico_range_high", "inq_last_6mths", "delinq_2yrs", "mths_since_last_delinq", "pub_rec",
                      "pub_rec_bankruptcies", "open_acc", "total_acc", "revol_bal", "revol_util", "bc_util", "tot_cur_bal",
                      "avg_cur_bal", "total_rev_hi_lim", "tot_hi_cred_lim", "mort_acc", "acc_open_past_24mths", "num_tl_op_past_12m"],
}
assert set(FIELDS) - {"id", "installment"} == {c for g in GROUPS.values() for c in g}, "form and data_spec.json disagree"
NAMES = [c for g in GROUPS.values() for c in g]


def make(name):
    f = FIELDS[name]
    need = "required" if f["missing_strategy"] == "reject" else "optional"
    label = name.replace("_", " ")
    if name == "term":
        return gr.Dropdown(["36 months", "60 months"], label=label, info="required")
    if "allowed_values" in f:
        return gr.Dropdown(f["allowed_values"], label=label, info=need)
    if f["type"] in ("float", "int"):
     return gr.Number(label=label, precision=0 if f["type"] == "int" else None,
                         info=f"{need}; valid range {f['min']:,} to {f['max']:,}")
    hint = {"issue_d": "e.g. Jun-2016", "earliest_cr_line": "e.g. Jan-2005", "addr_state": "two-letter code, e.g. CA"}.get(name, "")
    return gr.Textbox(label=label, info=f"{need} {hint}".strip())


def run(*vals):
    try:
        out = scorer.predict(to_record(dict(zip(NAMES, vals))))
    except ValidationError as e:
        return "### Input rejected\n" + "\n".join(f"- {m}" for m in e.errors), "", {}
    except Exception as e:
        return f"### Unexpected error\n{type(e).__name__}: {e}", "", {}
    head = f"## {out['decision'].upper()}: PD {out['pd']:.1%}\n{out['decision_basis']}"
    if out["warnings"]:
        head += "\n\n" + "\n".join(f"- warning: {w}" for w in out["warnings"])
    if out["reason_codes"]:
        why = "**Main reasons pushing the PD up**\n" + "\n".join(f"{i}. {r['reason']}" for i, r in enumerate(out["reason_codes"], 1))
    else:
        why = "No reason above the impact threshold applies to this applicant."
    return head, why, out


def loader(name):
    return lambda: [EXAMPLES[name].get(c) for c in NAMES]


m = scorer.meta
INFO = (f"**Model:** {scorer.model_version}  \n**Training data:** {m['n_train']:,} loans issued up to {m['fit_cutoff']}  \n"
        f"**Thresholds:** approve below {scorer.thresholds['approve_below']:.1%}, decline at {scorer.thresholds['decline_at_or_above']:.0%} and above, review in between  \n"
        "Applicants outside the training domain (joint applications, FICO below 660, dti above 100) are scored but always sent to review. "
        "All example applicants are invented.")

with gr.Blocks(title="Credit risk: probability of default") as demo:
    gr.Markdown("# Probability of default demo")
    gr.Markdown(INFO)
    buttons = {}
    with gr.Row():
        for name in EXAMPLES:
            buttons[name] = gr.Button(f"Example: {name}")
    inputs = []
    for title, cols in GROUPS.items():
        with gr.Accordion(title, open=(title == "Loan")):
            for c in cols:
                inputs.append(make(c))
    for name, btn in buttons.items():
        btn.click(loader(name), outputs=inputs)
    go = gr.Button("Score application", variant="primary")
    head, why, raw = gr.Markdown(), gr.Markdown(), gr.JSON(label="Full response")
    go.click(run, inputs=inputs, outputs=[head, why, raw])

if __name__ == "__main__":
    demo.launch()