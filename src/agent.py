"""Answer rep questions with Groq tool calling, or a rule-based mock fallback."""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import Groq

try:
    import tools
except ImportError:
    from src import tools

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
CLINIC_ID_RE = re.compile(r"C-\d{3}")
TOOL_CALL_LOG: list[dict[str, Any]] = []

SYSTEM_PROMPT = (
    "You are Clinic Pulse, an assistant for animal-health sales reps using synthetic data. "
    "Answer ONLY from tool results. Cite policy sources by file name and section. "
    "Say 'I don't know' when evidence is missing. Never invent numbers. "
    "Never recommend automatic actions; a human decides. Keep answers short."
)

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_at_risk_clinics",
            "description": "Rank clinics at risk of reducing orders for a rep.",
            "parameters": {
                "type": "object",
                "properties": {
                    "rep_id": {"type": "string", "description": "Rep id like REP-01"},
                    "top_n": {"type": "integer", "description": "Number of clinics"},
                },
                "required": ["rep_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_clinic_history",
            "description": "Get twelve months of orders and visits for a clinic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "clinic_id": {"type": "string", "description": "Clinic id like C-017"}
                },
                "required": ["clinic_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_risk",
            "description": "Get top three SHAP reasons for a clinic's risk score.",
            "parameters": {
                "type": "object",
                "properties": {
                    "clinic_id": {"type": "string", "description": "Clinic id like C-017"}
                },
                "required": ["clinic_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "Search synthetic policy documents with citations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Policy question"},
                    "top_k": {"type": "integer", "description": "Number of chunks"},
                },
                "required": ["query"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "get_at_risk_clinics": tools.get_at_risk_clinics,
    "get_clinic_history": tools.get_clinic_history,
    "explain_risk": tools.explain_risk,
    "search_policy": tools.search_policy,
}


def active_mode() -> str:
    """Return the configured mode before any request is made."""
    return "Groq" if API_KEY else "Mock"


def get_tool_call_log(limit: int = 20) -> list[dict[str, Any]]:
    """Return the most recent tool calls for display in the UI."""
    return TOOL_CALL_LOG[-limit:]


def _call_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Execute one tool and record the call with a timestamp."""
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    TOOL_CALL_LOG.append({"name": name, "args": args, "timestamp": timestamp})
    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        return {"ok": False, "error": f"Unknown tool: {name}"}
    try:
        return function(**args)
    except TypeError as exc:
        return {"ok": False, "error": f"Bad arguments for {name}: {exc}"}


def _citations_from(result: dict[str, Any]) -> list[dict[str, str]]:
    """Pull policy citations out of a tool result."""
    citations = []
    for chunk in result.get("results", []):
        citations.append(
            {
                "file": chunk.get("source_file", ""),
                "section": chunk.get("section", ""),
                "version": chunk.get("version", ""),
                "effective_date": chunk.get("effective_date", ""),
            }
        )
    return citations


def _format_risk(result: dict[str, Any], rep_id: str) -> str:
    """Format a ranked clinic list."""
    if not result.get("ok"):
        return f"I don't know. {result.get('error', 'Tool error')}"
    lines = [f"Top at-risk clinics for {rep_id} (synthetic scores):"]
    for clinic in result["clinics"][:5]:
        reasons = "; ".join(clinic["reasons"][:2])
        lines.append(
            f"{clinic['clinic_id']} - score {clinic['risk_score']:.3f} - {reasons}"
        )
    lines.append("A human decides whether any action is taken.")
    return "\n".join(lines)


def _format_explain(result: dict[str, Any], clinic_id: str) -> str:
    """Format one clinic explanation."""
    if not result.get("ok"):
        return f"I don't know. {result.get('error', 'Tool error')}"
    reasons = "; ".join(result["reasons"])
    return (
        f"{clinic_id} has a synthetic risk score of {result['risk_score']:.3f}. "
        f"Top reasons: {reasons}. These are model explanations, not a decision."
    )


def _format_history(result: dict[str, Any], clinic_id: str) -> str:
    """Format recent clinic history."""
    if not result.get("ok"):
        return f"I don't know. {result.get('error', 'Tool error')}"
    months = result["months"][-3:]
    summary = ", ".join(
        f"{month['month']}: value {month['order_value']:.0f}, "
        f"orders {month['order_count']}, visits {month['visits']}"
        for month in months
    )
    return f"{clinic_id} recent synthetic history - {summary}."


def _format_policy(result: dict[str, Any]) -> str:
    """Format the top policy chunk with its citation."""
    if not result.get("ok") or result.get("insufficient_evidence"):
        return "I don't know. The policy documents do not contain evidence for that question."
    chunk = result["results"][0]
    return (
        f"Policy guidance from {chunk['source_file']}, section '{chunk['section']}' "
        f"(version {chunk['version']}, effective {chunk['effective_date']}): "
        f"{chunk['text']} A human manager must approve any action."
    )


def _mock_answer(
    question: str, rep_id: str, mode: str = "Mock", note: str = ""
) -> dict[str, Any]:
    """Route common questions to tools using simple keyword rules."""
    text = question.lower()
    match = CLINIC_ID_RE.search(question.upper())
    clinic_id = match.group(0) if match else "C-017"
    tool_calls: list[dict[str, Any]] = []
    citations: list[dict[str, str]] = []

    if any(word in text for word in ["why", "reason", "explain", "flag"]):
        if not match:
            answer = "I don't know. Please include a clinic id such as C-017."
        else:
            result = _call_tool("explain_risk", {"clinic_id": clinic_id})
            tool_calls.append({"name": "explain_risk", "args": {"clinic_id": clinic_id}})
            answer = _format_explain(result, clinic_id)
    elif any(word in text for word in ["offer", "promotion", "discount", "policy", "eligible"]):
        if "discount" in text:
            query = "discount limits by tier maximum discounts approval rules"
        elif any(word in text for word in ["offer", "promotion", "eligible"]):
            query = question
        else:
            query = question
        result = _call_tool("search_policy", {"query": query, "top_k": 3})
        tool_calls.append({"name": "search_policy", "args": {"query": query, "top_k": 3}})
        citations = _citations_from(result)
        answer = _format_policy(result)
    elif any(word in text for word in ["risk", "at risk", "which clinics", "ranking"]):
        result = _call_tool("get_at_risk_clinics", {"rep_id": rep_id, "top_n": 5})
        tool_calls.append(
            {"name": "get_at_risk_clinics", "args": {"rep_id": rep_id, "top_n": 5}}
        )
        answer = _format_risk(result, rep_id)
    elif any(word in text for word in ["history", "orders", "trend", "visits"]):
        if not match:
            answer = "I don't know. Please include a clinic id such as C-017."
        else:
            result = _call_tool("get_clinic_history", {"clinic_id": clinic_id})
            tool_calls.append({"name": "get_clinic_history", "args": {"clinic_id": clinic_id}})
            answer = _format_history(result, clinic_id)
    else:
        answer = (
            "I don't know. This demo only answers questions about synthetic clinic risk, "
            "clinic history, and the three policy documents."
        )

    if note:
        answer = f"{note} {answer}"
    return {
        "answer": answer,
        "tool_calls": tool_calls,
        "citations": citations,
        "mode": mode,
    }


def _groq_answer(question: str, rep_id: str) -> dict[str, Any]:
    """Run the Groq tool-calling loop for at most four iterations."""
    client = Groq(api_key=API_KEY, timeout=20.0, max_retries=0)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Sales rep: {rep_id}\nQuestion: {question}"},
    ]
    tool_calls: list[dict[str, Any]] = []
    citations: list[dict[str, str]] = []
    for _ in range(4):
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.1,
            max_completion_tokens=800,
        )
        message = response.choices[0].message
        if not message.tool_calls:
            return {
                "answer": message.content or "I don't know.",
                "tool_calls": tool_calls,
                "citations": citations,
                "mode": "Groq",
            }
        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [call.model_dump() for call in message.tool_calls],
            }
        )
        for call in message.tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = _call_tool(name, args)
            tool_calls.append({"name": name, "args": args})
            for citation in _citations_from(result):
                if citation not in citations:
                    citations.append(citation)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": name,
                    "content": json.dumps(result),
                }
            )
    return {
        "answer": "I don't have enough tool evidence to answer that question.",
        "tool_calls": tool_calls,
        "citations": citations,
        "mode": "Groq",
    }


def answer(question: str, rep_id: str) -> dict[str, Any]:
    """Answer a question in Groq mode, with one retry then mock fallback."""
    if not isinstance(question, str) or not question.strip():
        return _mock_answer("", rep_id, mode=active_mode())
    if not isinstance(rep_id, str) or not rep_id.strip():
        rep_id = "REP-01"
    if not API_KEY:
        return _mock_answer(question.strip(), rep_id)
    for attempt in range(2):
        try:
            return _groq_answer(question.strip(), rep_id)
        except Exception as exc:
            if attempt == 0:
                time.sleep(1)
                continue
            return _mock_answer(
                question.strip(),
                rep_id,
                mode="Fallback",
                note=f"Fallback mode: Groq request failed ({type(exc).__name__}).",
            )
    return _mock_answer(question.strip(), rep_id, mode="Fallback")
