"""Tests for policy retrieval and refusal behaviour."""

import rag
import tools


def test_out_of_scope_question_refuses() -> None:
    """A question with no policy evidence must be refused."""
    assert rag.search("weather forecast for next week")["insufficient_evidence"] is True
    assert tools.search_policy("who won the match")["insufficient_evidence"] is True


def test_policy_answer_has_citation_metadata() -> None:
    """Relevant questions return file, section, version and date."""
    result = rag.search("what are the discount limits by tier", top_k=2)
    assert result["insufficient_evidence"] is False
    first = result["results"][0]
    assert first["source_file"] == "discount_limits.md"
    assert first["section"]
    assert first["version"] == "2.1"
    assert first["effective_date"] == "2026-03-01"
