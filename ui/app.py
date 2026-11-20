"""Simple Streamlit chat UI for the synthetic Clinic Pulse demo."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Clinic Pulse", layout="wide")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import agent  # noqa: E402
import tools  # noqa: E402

DATA_DIR = ROOT / "data"
FEEDBACK_FILE = DATA_DIR / "feedback.jsonl"


def save_feedback(question: str, answer: str, rating: str) -> None:
    """Append one thumbs rating to a local JSONL file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": question,
        "answer": answer,
        "rating": rating,
    }
    with FEEDBACK_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


st.markdown(
    """
    <style>
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .stChatMessage {
            max-width: 100%;
        }
        [data-testid="stBottomBlockContainer"] > div {
            max-width: 100%;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Clinic Pulse")
st.warning("Demo uses synthetic data.")
mode = st.session_state.get("last_mode", agent.active_mode())
st.caption(f"Active mode: {mode}")

reps = tools.list_reps()
rep_id = st.selectbox("Sales rep", options=reps, index=0)

st.subheader("Top at-risk clinics")
risk = tools.get_at_risk_clinics(rep_id, top_n=10)
if risk.get("ok"):
    table = pd.DataFrame(risk["clinics"])
    table["reasons"] = table["reasons"].apply("; ".join)
    st.dataframe(
        table[["clinic_id", "name", "region", "risk_score", "reasons"]],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.error(risk.get("error", "Could not load risk scores"))

st.subheader("Ask a question")
st.caption(
    "Try: Which clinics are most at risk? | Why is clinic C-017 flagged? | "
    "What can I offer clinic C-017?"
)

st.session_state.setdefault("messages", [])
question = st.chat_input("Ask about clinic risk, history, or policy...")
if question:
    result = agent.answer(question, rep_id)
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
    st.session_state.last_result = result
    st.session_state.last_question = question
    st.session_state.last_mode = result["mode"]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

with st.expander("Tools used and citations"):
    result = st.session_state.get("last_result")
    if result:
        st.write("Tool calls")
        st.json(result["tool_calls"])
        st.write("Citations")
        st.json(result["citations"])
    else:
        st.write("Ask a question to see tool calls and citations.")

with st.expander("Feedback"):
    if "last_result" in st.session_state:
        col_up, col_down = st.columns(2)
        if col_up.button("Thumbs up"):
            save_feedback(
                st.session_state.last_question,
                st.session_state.last_result["answer"],
                "up",
            )
            st.success("Thanks for the feedback.")
        if col_down.button("Thumbs down"):
            save_feedback(
                st.session_state.last_question,
                st.session_state.last_result["answer"],
                "down",
            )
            st.success("Thanks for the feedback.")
    else:
        st.write("Ask a question first.")
