"""Read-only core tools shared by the API, MCP server, agent and briefing."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CLINIC_ID_RE = re.compile(r"^C-\d{3}$")
REP_ID_RE = re.compile(r"^REP-\d{2}$")


def _error(message: str) -> dict[str, Any]:
    """Return a consistent error dictionary."""
    return {"ok": False, "error": message}


@lru_cache(maxsize=1)
def _clinics() -> pd.DataFrame:
    """Load the synthetic clinic master table once."""
    return pd.read_csv(DATA_DIR / "clinics.csv")


@lru_cache(maxsize=1)
def _scores() -> pd.DataFrame:
    """Load the latest risk scores once."""
    return pd.read_csv(DATA_DIR / "latest_scores.csv")


@lru_cache(maxsize=1)
def _orders() -> pd.DataFrame:
    """Load the synthetic order history once."""
    return pd.read_csv(DATA_DIR / "orders.csv")


@lru_cache(maxsize=1)
def _reasons() -> dict[str, list[str]]:
    """Load the saved SHAP explanations once."""
    return json.loads((DATA_DIR / "risk_reasons.json").read_text(encoding="utf-8"))


def _validate_clinic_id(clinic_id: Any) -> str | None:
    """Validate a clinic id and return an error message when invalid."""
    if not isinstance(clinic_id, str) or not CLINIC_ID_RE.fullmatch(clinic_id):
        return "Invalid clinic_id: expected a value like 'C-017'"
    if clinic_id not in set(_clinics()["clinic_id"]):
        return f"Unknown clinic_id: {clinic_id}"
    return None


def get_at_risk_clinics(rep_id: str, top_n: int = 10) -> dict[str, Any]:
    """Return the highest-risk clinics for one sales rep."""
    if not isinstance(rep_id, str) or not REP_ID_RE.fullmatch(rep_id):
        return _error("Invalid rep_id: expected a value like 'REP-01'")
    if isinstance(top_n, bool) or not isinstance(top_n, int) or not 1 <= top_n <= 50:
        return _error("Invalid top_n: expected an integer between 1 and 50")
    clinics = _clinics()
    if rep_id not in set(clinics["rep_id"]):
        return _error(f"Unknown rep_id: {rep_id}")
    ranked = _scores().merge(
        clinics[clinics["rep_id"] == rep_id][
            ["clinic_id", "name", "region", "size"]
        ],
        on="clinic_id",
        how="inner",
    )
    ranked = ranked.sort_values("risk_score", ascending=False).head(top_n)
    reasons = _reasons()
    result = [
        {
            "clinic_id": row.clinic_id,
            "name": row.name,
            "region": row.region,
            "size": row.size,
            "risk_score": round(float(row.risk_score), 3),
            "reasons": reasons.get(row.clinic_id, []),
        }
        for row in ranked.itertuples(index=False)
    ]
    return {"ok": True, "rep_id": rep_id, "count": len(result), "clinics": result}


def get_clinic_history(clinic_id: str) -> dict[str, Any]:
    """Return the last twelve months of orders and visits for one clinic."""
    error = _validate_clinic_id(clinic_id)
    if error:
        return _error(error)
    clinic = _clinics().loc[_clinics()["clinic_id"] == clinic_id].iloc[0]
    history = _orders()
    history = history[history["clinic_id"] == clinic_id].sort_values("month_index")
    months = []
    for row in history.itertuples(index=False):
        total = max(float(row.order_value), 1.0)
        months.append(
            {
                "month": row.month,
                "order_value": float(row.order_value),
                "order_count": int(row.order_count),
                "visits": int(row.visits),
                "product_mix": {
                    "vaccines": round(float(row.vaccines_value) / total, 3),
                    "diagnostics": round(float(row.diagnostics_value) / total, 3),
                    "nutrition": round(float(row.nutrition_value) / total, 3),
                },
            }
        )
    return {
        "ok": True,
        "clinic": {
            "clinic_id": clinic_id,
            "name": str(clinic["name"]),
            "region": str(clinic["region"]),
            "rep_id": str(clinic["rep_id"]),
            "size": str(clinic["size"]),
            "species_mix": str(clinic["species_mix"]),
        },
        "months": months[-12:],
    }


def explain_risk(clinic_id: str) -> dict[str, Any]:
    """Return the top three SHAP reasons and risk score for one clinic."""
    error = _validate_clinic_id(clinic_id)
    if error:
        return _error(error)
    scores = _scores()
    match = scores.loc[scores["clinic_id"] == clinic_id]
    if match.empty:
        return _error(f"No risk score available for {clinic_id}")
    return {
        "ok": True,
        "clinic_id": clinic_id,
        "risk_score": round(float(match.iloc[0]["risk_score"]), 3),
        "reasons": _reasons().get(clinic_id, []),
    }


def search_policy(query: str, top_k: int = 3) -> dict[str, Any]:
    """Return the most relevant policy chunks with citations."""
    if not isinstance(query, str) or not query.strip():
        return _error("Invalid query: expected a non-empty question")
    if len(query.strip()) > 300:
        return _error("Invalid query: maximum length is 300 characters")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 5:
        return _error("Invalid top_k: expected an integer between 1 and 5")
    try:
        try:
            from rag import search as rag_search
        except ImportError:
            from src.rag import search as rag_search
    except ImportError:
        return _error("Policy search is unavailable: rag module not found")
    result = rag_search(query.strip(), top_k=top_k)
    return {
        "ok": True,
        "query": query.strip(),
        "results": result["results"],
        "insufficient_evidence": result["insufficient_evidence"],
    }
