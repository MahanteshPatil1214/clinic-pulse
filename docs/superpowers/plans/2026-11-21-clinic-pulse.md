# Clinic Pulse Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working offline-first synthetic-data prototype for vet clinic risk prediction, policy Q&A, and weekly briefings.

**Architecture:** Batch steps generate synthetic data and train an XGBoost model with SHAP reasons persisted to `data/`. Four validated read-only tools in `tools.py` are the single source of truth, reused by RAG, MCP, the Groq/mock agent, FastAPI, Streamlit, and the briefing generator. The agent falls back to a rule-based mock on any Groq error.

**Tech Stack:** Python 3.11, FastAPI, XGBoost, SHAP, pandas, numpy, scikit-learn, MCP FastMCP SDK, groq, python-dotenv, Streamlit, pytest, SQLite.

---

### Task 1: Project skeleton and Stage 1 data + model

**Files:** `requirements.txt`, `.env.example`, `.gitignore`, `src/generate_data.py`, `src/train_model.py`, `data/` artifacts.

- [ ] Create `.venv` with `py -3.11 -m venv .venv` and install pinned dependencies.
- [ ] Generate 300 clinics, 12 months of order/visit history, declining cohort (~20%), 20% forward-drop label, leakage-free features; write CSVs and SQLite.
- [ ] Train XGBoost on months 1–9 and test on 10–12; print precision/recall/ROC-AUC plus the rule baseline; save model and top-3 plain-English SHAP reasons per clinic.
- [ ] Verify: `python src/generate_data.py` then `python src/train_model.py`; show printed metrics and confirm `data/` files exist.

### Task 2: Stage 2 core tools

**Files:** `src/tools.py`.

- [ ] Implement `get_at_risk_clinics(rep_id, top_n=10)`, `get_clinic_history(clinic_id)`, `explain_risk(clinic_id)`, `search_policy(query, top_k=3)` returning JSON-serializable dicts.
- [ ] Validate all inputs; return `{"error": ..., "ok": False}` instead of raising.
- [ ] Verify: a short Python smoke command prints one result for each function, including a bad-input case.

### Task 3: Stage 3 RAG

**Files:** `docs/policies/*.md` (3 files), `src/rag.py`.

- [ ] Write three short synthetic policy docs with version and effective date.
- [ ] Chunk by heading, build TF-IDF index, return top-k with file/section/version/date and score; return `insufficient evidence` below threshold.
- [ ] Verify: query "discount limits by tier" returns citations; query "weather forecast" returns insufficient evidence.

### Task 4: Stage 4 MCP server

**Files:** `src/mcp_server.py`.

- [ ] Expose the four tools through FastMCP with the same names and schemas.
- [ ] Verify: import check plus documented `mcp dev src/mcp_server.py` command for Inspector testing.

### Task 5: Stage 5 agent

**Files:** `src/agent.py`.

- [ ] `answer(question, rep_id)` returns `{answer, tool_calls, citations, mode}`.
- [ ] Groq path: four JSON-schema tools, temperature 0.1, max 4 iterations, one retry on timeout/429/API error, then mock fallback with mode `Fallback`.
- [ ] Mock path: keyword router over the same tools; short answers with citations and refusals.
- [ ] Verify: mock answers for the four demo questions and one out-of-scope refusal; inspect logged tool calls.

### Task 6: Stage 6 API, UI, briefing

**Files:** `src/api.py`, `ui/app.py`, `src/briefing.py`.

- [ ] FastAPI endpoints `/health`, `/clinics/at-risk`, `/clinics/{id}`, `/ask`.
- [ ] Streamlit UI with banner, mode label, rep selector, chat, at-risk table, tool/citation expander, thumbs feedback file.
- [ ] Briefing CLI writing `data/briefings/<rep_id>.md` with top 5, reasons, and policy citations.
- [ ] Verify: API import + TestClient smoke checks, Streamlit import check, briefing file generated and inspected.

### Task 7: Stage 7 evaluation and tests

**Files:** `eval/questions.json`, `eval/run_eval.py`, `tests/*.py`.

- [ ] 10–15 eval questions with expected tool/source and refusal cases; print pass/fail; exercise Groq when key present.
- [ ] Pytest tests for no leakage, bad tool inputs, and RAG refusal.
- [ ] Verify: `pytest -q` and `python eval/run_eval.py` both pass.

### Task 8: Stage 8 README and final verification

**Files:** `README.md`.

- [ ] Document quick start, all run commands, env setup, model choice, modes/fallback, limitations, production next steps, 3-minute demo script, and 10 interview questions.
- [ ] Verify: clean reinstall into `.venv`, rerun generation/training/tests/eval, and report whether XGBoost beat the baseline.

---

Execution notes: no git commits (workspace is not a git repo and commits were not requested). After each task, run its verification command, show the output, fix any failure, then continue.
