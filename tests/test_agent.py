"""Tests for the Groq tool loop's exhaustion behaviour."""

from types import SimpleNamespace

import agent


class _AlwaysTools:
    """Fake Groq client that only issues tool calls, never a final answer."""

    def __init__(self, *args, **kwargs):
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        call = SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(
                name="get_at_risk_clinics",
                arguments='{"rep_id": "REP-01", "top_n": 5}',
            ),
            model_dump=lambda: {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "get_at_risk_clinics",
                    "arguments": '{"rep_id": "REP-01", "top_n": 5}',
                },
            },
        )
        message = SimpleNamespace(content=None, tool_calls=[call])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_loop_exhaustion_answers_from_tool_results(monkeypatch) -> None:
    """Four tool-only iterations must still produce an evidence-based answer."""
    monkeypatch.setattr(agent, "API_KEY", "test-key")
    monkeypatch.setattr(agent, "Groq", _AlwaysTools)
    result = agent.answer("Which clinics are most at risk?", "REP-01")
    assert result["mode"] == "Fallback"
    assert "I don't have enough tool evidence" not in result["answer"]
    assert "C-008" in result["answer"]
    assert result["tool_calls"]
