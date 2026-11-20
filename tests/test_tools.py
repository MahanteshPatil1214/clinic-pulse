"""Tests for input validation in the core tools."""

import tools


def test_bad_inputs_return_errors() -> None:
    """Invalid arguments must produce error dictionaries, not exceptions."""
    assert tools.get_at_risk_clinics("REP-99")["ok"] is False
    assert tools.get_at_risk_clinics("REP-01", top_n=0)["ok"] is False
    assert tools.get_clinic_history("C-999")["ok"] is False
    assert tools.get_clinic_history(None)["ok"] is False
    assert tools.explain_risk("bad")["ok"] is False
    assert tools.search_policy("")["ok"] is False
    assert tools.search_policy("policy", top_k=99)["ok"] is False


def test_valid_tool_calls_return_data() -> None:
    """Valid calls return the expected top-level fields."""
    ranked = tools.get_at_risk_clinics("REP-01", top_n=3)
    assert ranked["ok"] is True
    assert len(ranked["clinics"]) == 3
    assert ranked["clinics"][0]["reasons"]
    history = tools.get_clinic_history("C-017")
    assert history["ok"] is True
    assert len(history["months"]) == 12
    explanation = tools.explain_risk("C-017")
    assert explanation["ok"] is True
    assert len(explanation["reasons"]) == 3
