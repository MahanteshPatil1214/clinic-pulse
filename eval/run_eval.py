"""Run the eval questions in mock mode, and in Groq mode when a key is set."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import agent  # noqa: E402

QUESTIONS_FILE = ROOT / "eval" / "questions.json"


def is_pass(result: dict, case: dict) -> bool:
    """Check whether one answer satisfies the eval case."""
    answer = result["answer"].lower()
    if case["should_refuse"]:
        return "i don't know" in answer or "insufficient evidence" in answer
    tool_names = [call["name"] for call in result["tool_calls"]]
    if case["expected_tool"] not in tool_names:
        return False
    if case["expected_source"]:
        sources = [citation["file"] for citation in result["citations"]]
        return case["expected_source"] in sources
    return True


def main() -> None:
    """Run all eval cases and print pass/fail for each mode."""
    cases = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    saved_key = agent.API_KEY
    agent.API_KEY = ""
    passed = 0
    print("Mode: Mock")
    for case in cases:
        result = agent.answer(case["question"], case["rep_id"])
        ok = is_pass(result, case)
        passed += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {case['question']}")
    print(f"Mock score: {passed}/{len(cases)}")
    agent.API_KEY = saved_key
    if saved_key:
        print("Mode: Groq")
        groq_passed = 0
        for case in cases:
            result = agent.answer(case["question"], case["rep_id"])
            ok = is_pass(result, case)
            groq_passed += int(ok)
            print(f"[{'PASS' if ok else 'FAIL'}] ({result['mode']}) {case['question']}")
        print(f"Groq score: {groq_passed}/{len(cases)}")


if __name__ == "__main__":
    main()
