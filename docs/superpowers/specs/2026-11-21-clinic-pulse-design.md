# Clinic Pulse — Design Spec

Date: 2026-11-21

## Purpose

A small, offline-first prototype that helps animal-health sales reps identify vet clinics at risk of cutting orders in the next 90 days, explain why, answer policy questions with citations, and produce a weekly briefing. All data is synthetic. All tools are read-only. A human makes every decision.

## Scope

Eight stages, built and verified in order: synthetic data + model, core tools, RAG, MCP server, agent, API/UI/briefing, evaluation/tests, README. Cut scope rather than leave a stage broken. No features beyond the requested list.

## Key decisions

- Python 3.11 virtual environment (`.venv`), created with `py -3.11 -m venv .venv`.
- Default `GROQ_MODEL=openai/gpt-oss-120b`. The originally requested `llama-3.3-70b-versatile` was deprecated and shut down on 2026-08-16 per Groq's changelog. `GROQ_MODEL` remains configurable.
- Mock mode is automatic when `GROQ_API_KEY` is unset. Any Groq timeout, HTTP 429, or API error gets one retry with a short delay, then falls back to mock and reports mode `Fallback`.
- Data artifacts (generated CSVs, SQLite DB, trained model, SHAP reasons, briefings, UI feedback) live under `data/`.
- Policies live as three synthetic markdown files with explicit version and effective date.
- The RAG layer uses scikit-learn TF-IDF only: no embedding downloads, no network.
- No commit is made: the workspace is not a git repository and committing was not requested.

## Architecture

```text
generate_data.py -> data/*.csv, data/clinic_pulse.db
        |                    |
        v                    v
train_model.py -> data/model.json + data/risk_reasons.json
        |
        v
tools.py (get_at_risk_clinics, get_clinic_history, explain_risk, search_policy)
   |        |          |
   |        |          +-> rag.py -> docs/policies/*.md
   |        +-> mcp_server.py (FastMCP, 4 tools)
   +-> agent.py (Groq tool loop / mock router + retry/fallback)
          |         |            |
          v         v            v
       api.py    ui/app.py    briefing.py
```

## Components

- `generate_data.py`: 300 clinics across regions/reps/sizes/species mix; 12 months of order value, order count, three product lines, and visits; fixed seed; ~20% declining in the last months; label from next-3-month value drop of 20%+; past-only features.
- `train_model.py`: time-aware train/test split, XGBoost, precision/recall/ROC-AUC, rule baseline ("value down >15% over 3 months"), persisted model plus plain-English top-3 SHAP reasons per clinic.
- `tools.py`: pure JSON-serializable functions with input validation and error dicts, never exceptions escaping to callers.
- `rag.py`: heading-based chunks, TF-IDF index, cosine scores with source file, section, version, date; low best score returns `insufficient evidence`.
- `mcp_server.py`: FastMCP exposing the same four tools; one-command launch documented with MCP Inspector invocation.
- `agent.py`: `answer(question, rep_id)` returns `{answer, tool_calls, citations, mode}`; tool calls logged with name, args, timestamp; system prompt forbids invented numbers, automatic actions, and answers outside tool evidence.
- `api.py`: `/health`, `/clinics/at-risk`, `/clinics/{id}`, `/ask`.
- `ui/app.py`: synthetic-data banner, mode label, rep selector, chat, at-risk table, expandable tool calls and citations, thumbs feedback appended to a local file.
- `briefing.py`: per-rep markdown briefing in `data/briefings/`, one command, cron-friendly.
- `eval/run_eval.py` plus `eval/questions.json`: 10–15 mock-mode cases including refusal cases; runs Groq mode when a key is present; prints pass/fail.
- `tests/`: leakage check, bad-input tool handling, RAG refusal for out-of-scope questions.

## Data flow and error handling

1. Generation and training are offline batch steps. All other components read persisted artifacts.
2. The agent only sees tool results. Missing data yields `insufficient evidence` or `I don't know`, never a guess.
3. Every tool validates IDs and numeric arguments and returns `{"error": "..."}` on bad input.
4. Groq failures never break the demo: retry once, then mock fallback with mode `Fallback`.

## Testing and verification

After each stage: run the stage's script/command and show its output, then fix failures before the next stage. Final verification: install from `requirements.txt`, run data generation, training, tests, eval, API smoke check, MCP import check, briefing, and a Streamlit import check. If XGBoost does not beat the rule baseline, the README states that explicitly.
