"""Simple Streamlit chat UI for the synthetic Clinic Pulse demo."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from html import escape

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


def risk_pill(score: float) -> str:
    """Return a colour-coded score pill."""
    if score >= 0.8:
        background, colour = "#e6f7f4", "#0f6f66"
    elif score >= 0.5:
        background, colour = "#fff4d6", "#8a5b00"
    else:
        background, colour = "#ffe8e6", "#a3261d"
    return (
        f'<span class="cp-pill" style="background:{background};color:{colour}">'
        f"{score:.3f}</span>"
    )


def render_risk_table(clinics: list[dict]) -> str:
    """Render the ranked clinic table as escaped HTML."""
    rows = []
    for clinic in clinics:
        cells = [
            f"<td>{escape(str(clinic['clinic_id']))}</td>",
            f"<td>{escape(str(clinic['name']))}</td>",
            f"<td>{escape(str(clinic['region']))}</td>",
            f"<td>{risk_pill(float(clinic['risk_score']))}</td>",
            f"<td>{escape('; '.join(clinic['reasons']))}</td>",
        ]
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<table class="cp-table"><thead><tr>'
        "<th>Clinic</th><th>Name</th><th>Region</th><th>Risk</th><th>Reasons</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


st.markdown(
    """
    <style>
        body, .stApp { font-family:'Aptos','Segoe UI',sans-serif; }
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
        .cp-status { display:flex; gap:.75rem; align-items:center; margin-bottom:1rem; }
        .cp-banner { background:#e6f7f4; color:#0f4c4c; border:1px solid #c9e6e4; border-radius:.5rem; padding:.5rem .75rem; font-weight:600; }
        .cp-mode { background:#f4fbfb; color:#0f6f66; border:1px solid #c9e6e4; border-radius:999px; padding:.4rem .75rem; font-size:.85rem; }
        .cp-card { background:#fff; border:1px solid #c9e6e4; border-left:4px solid #0f9b8e; border-radius:.5rem; padding:.75rem; }
        .cp-card-label { font-size:.75rem; text-transform:uppercase; letter-spacing:.05em; color:#39716f; }
        .cp-card-value { font-size:1.4rem; font-weight:700; color:#0f4c4c; }
        .cp-table { width:100%; border-collapse:collapse; background:#fff; }
        .cp-table th { background:#0f4c4c; color:#fff; text-align:left; padding:.5rem; }
        .cp-table td { padding:.5rem; border-bottom:1px solid #e3efee; vertical-align:top; }
        .cp-pill { display:inline-block; padding:.15rem .5rem; border-radius:999px; font-weight:700; font-variant-numeric:tabular-nums; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Clinic Pulse")
mode = st.session_state.get("last_mode", agent.active_mode())
st.markdown(
    '<div class="cp-status">'
    '<div class="cp-banner">Demo uses synthetic data.</div>'
    f'<div class="cp-mode">Active mode: {escape(str(mode))}</div>'
    "</div>",
    unsafe_allow_html=True,
)

reps = tools.list_reps()
rep_id = st.selectbox("Sales rep", options=reps, index=0)

st.subheader("Top at-risk clinics")
risk = tools.get_at_risk_clinics(rep_id, top_n=10)
if risk.get("ok"):
    scores = [float(clinic["risk_score"]) for clinic in risk["clinics"]]
    labels = ["Clinics reviewed", "Highest risk score", "Average risk score"]
    values = [
        str(len(scores)),
        f"{max(scores):.3f}" if scores else "-",
        f"{sum(scores) / len(scores):.3f}" if scores else "-",
    ]
    for column, label, value in zip(st.columns(3), labels, values):
        column.markdown(
            f'<div class="cp-card">'
            f'<div class="cp-card-label">{escape(label)}</div>'
            f'<div class="cp-card-value">{escape(value)}</div>'
            "</div>",
            unsafe_allow_html=True,
        )
    st.markdown(render_risk_table(risk["clinics"]), unsafe_allow_html=True)
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
