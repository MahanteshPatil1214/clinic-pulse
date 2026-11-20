"""Regression tests for Streamlit UI dependencies and rendering."""

import numpy
from streamlit.testing.v1 import AppTest


def test_pyarrow_imports_with_pinned_numpy() -> None:
    """Streamlit dataframes need a pyarrow build compatible with pinned NumPy."""
    import pyarrow

    assert numpy.__version__.startswith("1.")
    assert pyarrow.__version__


def test_polished_ui_elements_render() -> None:
    """Modern clinical styling, status row, cards and pills render without errors."""
    at = AppTest.from_file("ui/app.py", default_timeout=60).run()
    assert len(at.exception) == 0
    markdown = "\n".join(element.value for element in at.markdown)
    assert "font-family:'Aptos'" in markdown
    assert "Demo uses synthetic data." in markdown
    assert "Active mode:" in markdown
    for label in ("Clinics reviewed", "Highest risk score", "Average risk score"):
        assert label in markdown
    assert "cp-pill" in markdown
