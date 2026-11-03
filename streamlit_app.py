
import json
import streamlit as st

from src.predict import Scorer, ValidationError
from src.demo import EXAMPLES, to_record


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# 2. LOAD THE ORIGINAL MODEL AND CONTRACT
# ============================================================

@st.cache_resource
def load_scorer():
    return Scorer()


scorer = load_scorer()

with open("contract/data_spec.json", "r", encoding="utf-8") as file:
    spec = json.load(file)


# ============================================================
# 3. ORIGINAL FIELD DEFINITIONS AND GROUPS
# ============================================================

FIELDS = {f["name"]: f for f in spec["fields"]}

GROUPS = {
    "Loan": [
        "loan_amnt", "term", "int_rate", "grade", "sub_grade",
        "purpose", "application_type", "issue_d"
    ],
    "Applicant": [
        "annual_inc", "emp_length", "emp_title", "home_ownership",
        "verification_status", "dti", "addr_state", "earliest_cr_line"
    ],
    "Credit bureau": [
        "fico_range_low", "fico_range_high", "inq_last_6mths",
        "delinq_2yrs", "mths_since_last_delinq", "pub_rec",
        "pub_rec_bankruptcies", "open_acc", "total_acc", "revol_bal",
        "revol_util", "bc_util", "tot_cur_bal", "avg_cur_bal",
        "total_rev_hi_lim", "tot_hi_cred_lim", "mort_acc",
        "acc_open_past_24mths", "num_tl_op_past_12m"
    ]
}

NAMES = [
    name
    for group in GROUPS.values()
    for name in group
]

assert set(FIELDS) - {"id", "installment"} == set(NAMES), (
    "Form fields and data_spec.json disagree"
)


# ============================================================
# 4. MODEL INFORMATION
# ============================================================

meta = scorer.meta
thresholds = scorer.thresholds

st.title("📊 Credit Risk Prediction")
st.caption("Probability of Default (PD) Assessment")

st.markdown(
    f"""
**Model:** {scorer.model_version}

**Training data:** {meta['n_train']:,} loans issued up to
{meta['fit_cutoff']}

**Decision thresholds:**
- Approve: PD below {thresholds['approve_below']:.1%}
- Review: PD from {thresholds['approve_below']:.1%} to below
  {thresholds['decline_at_or_above']:.0%}
- Decline: PD at or above {thresholds['decline_at_or_above']:.0%}

Applicants outside the training domain (joint applications,
FICO below 660, or DTI above 100) are scored but always sent
to review.

All example applicants are invented.
"""
)

st.divider()


# ============================================================
# 5. EXAMPLE APPLICANT PROFILES
# ============================================================

st.subheader("Try an example applicant")

cols = st.columns(3)

for col, name in zip(cols, EXAMPLES):

    if col.button(
        f"Example: {name}",
        use_container_width=True
    ):

        for field in NAMES:
            value = EXAMPLES[name].get(field)

            if field == "term" and value is None:
                value = "36 months"

            st.session_state[field] = value

        st.rerun()


# ============================================================
# 6. APPLICANT INPUT FORM
# ============================================================

st.subheader("Applicant information")

values = {}

with st.form("credit_risk_form"):

    for group_name, group_fields in GROUPS.items():

        with st.expander(
            group_name,
            expanded=(group_name == "Loan")
        ):

            columns = st.columns(2)

            for index, name in enumerate(group_fields):

                f = FIELDS[name]
                label = name.replace("_", " ").title()

                required = f["missing_strategy"] == "reject"
                need = "Required" if required else "Optional"

                help_text = need

                if f["type"] in ("float", "int"):
                    help_text += (
                        f" | Valid range: {f['min']} to {f['max']}"
                    )

                elif name == "issue_d":
                    help_text += " | Example: Jun-2016"

                elif name == "earliest_cr_line":
                    help_text += " | Example: Jan-2005"

                elif name == "addr_state":
                    help_text += " | Two-letter code, e.g. CA"

                with columns[index % 2]:

                    # ------------------------------------------------
                    # Term
                    # ------------------------------------------------

                    if name == "term":

                        options = [
                            "36 months",
                            "60 months"
                        ]

                        default = st.session_state.get(
                            name,
                            "36 months"
                        )

                        values[name] = st.selectbox(
                            label,
                            options,
                            index=(
                                options.index(default)
                                if default in options else 0
                            ),
                            help=help_text,
                            key=name
                        )

                    # ------------------------------------------------
                    # Categorical fields
                    # ------------------------------------------------

                    elif "allowed_values" in f:

                        options = list(f["allowed_values"])

                        if not required:
                            options = [""] + options

                        default = st.session_state.get(
                            name,
                            options[0] if required else ""
                        )

                        values[name] = st.selectbox(
                            label,
                            options,
                            index=(
                                options.index(default)
                                if default in options else 0
                            ),
                            help=help_text,
                            key=name
                        )

                    # ------------------------------------------------
                    # Integer fields
                    # ------------------------------------------------

                    elif f["type"] == "int":

                        current_value = st.session_state.get(name)

                        if current_value is not None:
                            current_value = int(current_value)

                        values[name] = st.number_input(
                            label,
                            min_value=int(f["min"]),
                            max_value=int(f["max"]),
                            value=current_value,
                            step=1,
                            format="%d",
                            help=help_text,
                            key=name
                        )

                    # ------------------------------------------------
                    # Floating-point fields
                    # ------------------------------------------------

                    elif f["type"] == "float":

                        current_value = st.session_state.get(name)

                        if current_value is not None:
                            current_value = float(current_value)

                        values[name] = st.number_input(
                            label,
                            min_value=float(f["min"]),
                            max_value=float(f["max"]),
                            value=current_value,
                            step=0.01,
                            format="%.2f",
                            help=help_text,
                            key=name
                        )

                    # ------------------------------------------------
                    # Text fields
                    # ------------------------------------------------

                    else:

                        default = st.session_state.get(name, "")

                        values[name] = st.text_input(
                            label,
                            value=(
                                default
                                if default is not None else ""
                            ),
                            help=help_text,
                            key=name
                        )

    submitted = st.form_submit_button(
        "Score application",
        type="primary",
        use_container_width=True
    )


# ============================================================
# 7. PREDICTION USING THE ORIGINAL BUSINESS LOGIC
# ============================================================

if submitted:

    st.divider()
    st.subheader("Prediction results")

    try:

        # Prepare the original applicant record.
        record = to_record(values)

        # Execute the original trained model prediction.
        result = scorer.predict(record)

        decision = result["decision"].upper()
        pd_value = result["pd"]

        # --------------------------------------------------------
        # Decision result
        # --------------------------------------------------------

        if decision == "APPROVE":

            st.success(
                f"APPROVED — PD: {pd_value:.2%}"
            )

        elif decision == "DECLINE":

            st.error(
                f"DECLINED — PD: {pd_value:.2%}"
            )

        else:

            st.warning(
                f"MANUAL REVIEW — PD: {pd_value:.2%}"
            )

        # --------------------------------------------------------
        # Probability of default
        # --------------------------------------------------------

        st.metric(
            "Probability of Default",
            f"{pd_value:.2%}"
        )

        # --------------------------------------------------------
        # Decision basis
        # --------------------------------------------------------

        st.write(
            "**Decision basis:**",
            result["decision_basis"]
        )

        st.write(
            "**Model version:**",
            result["model_version"]
        )

        # --------------------------------------------------------
        # Input warnings
        # --------------------------------------------------------

        if result["warnings"]:

            st.markdown("### Input warnings")

            for warning in result["warnings"]:
                st.warning(warning)

        # --------------------------------------------------------
        # Out-of-domain flags
        # --------------------------------------------------------

        if result["flags"]:

            st.markdown("### Out-of-domain review flags")

            for flag in result["flags"]:
                st.warning(flag)

        # --------------------------------------------------------
        # Reason codes
        # --------------------------------------------------------

        st.markdown("### Main reasons pushing the PD up")

        if result["reason_codes"]:

            for index, reason in enumerate(
                result["reason_codes"],
                start=1
            ):

                st.write(
                    f"{index}. {reason['reason']} "
                    f"(impact: {reason['impact']})"
                )

        else:

            st.info(
                "No reason above the impact threshold applies "
                "to this applicant."
            )

        # --------------------------------------------------------
        # Complete prediction response
        # --------------------------------------------------------

        with st.expander("View full prediction response"):
            st.json(result)

    except ValidationError as error:

        st.error("Input rejected")

        for message in error.errors:
            st.write(f"- {message}")

    except Exception as error:

        st.error(
            f"Unexpected error: {type(error).__name__}: {error}"
        )


# ============================================================
# 8. FOOTER
# ============================================================

st.divider()

st.caption(
    "Credit Risk Prediction | Original trained model and "
    "prediction pipeline | Contract 1.0.0"
)
