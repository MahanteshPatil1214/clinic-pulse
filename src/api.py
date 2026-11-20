"""FastAPI REST surface for the Clinic Pulse prototype."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

try:
    import agent
    import tools
except ImportError:
    from src import agent
    from src import tools

app = FastAPI(title="Clinic Pulse", description="Synthetic-data demo API")


class AskRequest(BaseModel):
    """Question submitted by a sales rep."""

    question: str
    rep_id: str = "REP-01"


@app.get("/health")
def health() -> dict[str, Any]:
    """Report service status and the configured agent mode."""
    return {
        "status": "ok",
        "mode": agent.active_mode(),
        "synthetic_data": True,
    }


@app.get("/clinics/at-risk")
def at_risk(rep_id: str, top_n: int = 10) -> dict[str, Any]:
    """Return ranked at-risk clinics for a rep."""
    return tools.get_at_risk_clinics(rep_id, top_n)


@app.get("/clinics/{clinic_id}")
def clinic(clinic_id: str) -> dict[str, Any]:
    """Return one clinic's order history and risk explanation."""
    history = tools.get_clinic_history(clinic_id)
    explanation = tools.explain_risk(clinic_id)
    return {"history": history, "explanation": explanation}


@app.post("/ask")
def ask(request: AskRequest) -> dict[str, Any]:
    """Answer a question through the agent."""
    return agent.answer(request.question, request.rep_id)
