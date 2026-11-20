"""Regression test for Streamlit's Arrow dependency compatibility."""

import numpy


def test_pyarrow_imports_with_pinned_numpy() -> None:
    """Streamlit dataframes need a pyarrow build compatible with pinned NumPy."""
    import pyarrow

    assert numpy.__version__.startswith("1.")
    assert pyarrow.__version__
