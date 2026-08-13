"""
Human-evaluation form for Chapter 9.

Run:
    pip install streamlit pandas
    streamlit run human_evaluation_form.py

Expected cases file:
    sample_evaluation_cases.csv
or set:
    HUMAN_EVAL_CASES=/path/to/your_cases.csv
    HUMAN_EVAL_LOG=/path/to/human_evaluation_log.csv

The app intentionally hides the gold hallucination label and the system name
from the evaluator. The system name is retained only as metadata in the log.
"""

import os
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

CASES_PATH = Path(os.environ.get("HUMAN_EVAL_CASES", "sample_evaluation_cases.csv"))
LOG_PATH = Path(os.environ.get("HUMAN_EVAL_LOG", "human_evaluation_log.csv"))

REQUIRED_CASE_COLUMNS = {
    "stimulus_id", "dataset", "case_id", "system",
    "question", "context", "llm_answer",
    "verifier_explanation", "gold_hallucination"
}

ROLE_OPTIONS = ["Clinician", "AI researcher", "General user"]

st.set_page_config(page_title="Human Evaluation — D-ALP", layout="wide")

st.title("Human Evaluation of Hallucination-Verification Explanations")
st.caption(
    "Two-stage protocol: first judge the LLM output alone; then reveal the "
    "verification explanation and rate its quality."
)

@st.cache_data
def load_cases(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_CASE_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Cases file is missing required columns: {sorted(missing)}")
    return df

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def bool_to_text(v):
    return "Hallucinated" if bool(v) else "Not hallucinated"

def append_row(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame([row])
    if path.exists():
        out.to_csv(path, mode="a", header=False, index=False)
    else:
        out.to_csv(path, index=False)

def existing_submission(participant_id: str, stimulus_id: str) -> bool:
    if not LOG_PATH.exists():
        return False
    try:
        log = pd.read_csv(LOG_PATH, usecols=["participant_id", "stimulus_id"])
    except Exception:
        return False
    return bool(
        ((log["participant_id"].astype(str) == participant_id) &
         (log["stimulus_id"].astype(str) == stimulus_id)).any()
    )

try:
    cases = load_cases(CASES_PATH)
except Exception as e:
    st.error(f"Could not load cases: {e}")
    st.stop()

with st.sidebar:
    st.header("Evaluator")
    participant_id = st.text_input(
        "Participant code",
        help="Use a study code rather than a name or email."
    ).strip()
    participant_role = st.selectbox("Evaluator group", ROLE_OPTIONS)
    st.divider()
    st.write("Progress is stored in:")
    st.code(str(LOG_PATH))

if not participant_id:
    st.info("Enter a participant code in the sidebar to begin.")
    st.stop()

# Randomize display order deterministically per participant.
seed = abs(hash(participant_id)) % (2**32)
display_cases = cases.sample(frac=1, random_state=seed).reset_index(drop=True)

stimulus_options = display_cases["stimulus_id"].tolist()
selected_stimulus = st.selectbox("Evaluation item", stimulus_options)
case = display_cases.loc[display_cases["stimulus_id"] == selected_stimulus].iloc[0]

if existing_submission(participant_id, selected_stimulus):
    st.warning("This participant has already submitted this item. Choose another item.")
    st.stop()

# Reset phase state if the user changes item.
phase_key = f"{participant_id}|{selected_stimulus}"
if st.session_state.get("phase_key") != phase_key:
    st.session_state["phase_key"] = phase_key
    st.session_state["phase1_complete"] = False
    st.session_state.pop("phase1_payload", None)

st.subheader(f"Dataset: {case['dataset']} · Case: {case['case_id']}")
st.caption("The verification system identity and the gold label are blinded.")

st.markdown("### Case material")
if isinstance(case.get("context"), str) and case["context"].strip():
    st.markdown("**Context / evidence**")
    st.info(case["context"])

st.markdown("**Question**")
st.write(case["question"])

st.markdown("**LLM answer**")
st.warning(case["llm_answer"])

st.markdown("---")
st.markdown("## Phase 1 — Judgment before seeing the verification explanation")
st.write(
    "Judge whether the LLM answer contains a hallucination using only the "
    "case material above. Then report how confident you are in that judgment."
)

initial_judgment = st.radio(
    "Initial hallucination judgment",
    ["Hallucinated", "Not hallucinated"],
    horizontal=True,
    disabled=st.session_state["phase1_complete"],
)
confidence_before = st.slider(
    "Confidence before explanation (0–100)",
    min_value=0,
    max_value=100,
    value=60,
    step=1,
    disabled=st.session_state["phase1_complete"],
)

if not st.session_state["phase1_complete"]:
    if st.button("Save initial judgment and reveal explanation", type="primary"):
        st.session_state["phase1_payload"] = {
            "initial_judgment": initial_judgment,
            "confidence_before": confidence_before,
            "phase1_saved_at_utc": utc_now_iso(),
        }
        st.session_state["phase1_complete"] = True
        st.rerun()

if st.session_state["phase1_complete"]:
    p1 = st.session_state["phase1_payload"]

    st.success("Initial judgment saved. The verification explanation is now revealed.")
    st.markdown("## Phase 2 — Evaluation of the verification explanation")
    st.markdown("**Verification explanation**")
    st.success(case["verifier_explanation"])

    st.write(
        "Rate the explanation itself. Ratings use a 1–5 scale "
        "(1 = very poor, 5 = excellent)."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        clarity = st.select_slider(
            "Clarity",
            options=[1, 2, 3, 4, 5],
            value=4,
            help="How easy is the explanation to understand?"
        )
    with c2:
        coherence = st.select_slider(
            "Coherence",
            options=[1, 2, 3, 4, 5],
            value=4,
            help="How logically connected and internally consistent is it?"
        )
    with c3:
        trust = st.select_slider(
            "Trust",
            options=[1, 2, 3, 4, 5],
            value=4,
            help="How trustworthy does the verification explanation appear?"
        )

    final_judgment = st.radio(
        "Hallucination judgment after seeing the explanation",
        ["Hallucinated", "Not hallucinated"],
        index=0 if p1["initial_judgment"] == "Hallucinated" else 1,
        horizontal=True,
    )
    confidence_after = st.slider(
        "Confidence after explanation (0–100)",
        min_value=0,
        max_value=100,
        value=max(0, min(100, p1["confidence_before"] + 10)),
        step=1,
    )

    useful_evidence = st.multiselect(
        "Which parts of the verifier explanation were most useful? (optional)",
        [
            "Abductive support / missing premises",
            "Counter-abductive rival hypotheses",
            "Integrity-constraint violation",
            "Discourse-weighted evidence",
            "Defeat / inconsistency trace",
            "External grounding / retrieval"
        ]
    )
    comments = st.text_area("Comments (optional)", height=100)

    confirm = st.checkbox("I have completed both phases for this item.")

    if st.button("Submit evaluation", type="primary", disabled=not confirm):
        submitted_at = utc_now_iso()
        phase1_time = pd.Timestamp(p1["phase1_saved_at_utc"])
        submit_time = pd.Timestamp(submitted_at)
        seconds_after_explanation = max(
            0.0, (submit_time - phase1_time).total_seconds()
        )

        row = {
            "participant_id": participant_id,
            "participant_role": participant_role,
            "stimulus_id": case["stimulus_id"],
            "dataset": case["dataset"],
            "case_id": case["case_id"],
            "system": case["system"],  # hidden in UI, retained for analysis
            "gold_hallucination": bool(case["gold_hallucination"]),
            "initial_judgment": p1["initial_judgment"],
            "confidence_before": int(p1["confidence_before"]),
            "final_judgment": final_judgment,
            "confidence_after": int(confidence_after),
            "clarity": int(clarity),
            "coherence": int(coherence),
            "trust": int(trust),
            "useful_evidence": "|".join(useful_evidence),
            "comments": comments.strip(),
            "phase1_saved_at_utc": p1["phase1_saved_at_utc"],
            "submitted_at_utc": submitted_at,
            "seconds_after_explanation": round(seconds_after_explanation, 2),
            "is_synthetic_demo": bool(case.get("is_synthetic_demo", False)),
        }
        append_row(LOG_PATH, row)

        st.success("Evaluation submitted.")
        st.session_state["phase1_complete"] = False
        st.session_state.pop("phase1_payload", None)
        st.rerun()
