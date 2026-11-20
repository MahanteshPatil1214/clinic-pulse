# Clinic Pulse UI Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the approved Modern clinical visual direction and three selected UI improvements to `ui/app.py` only.

**Architecture:** All changes stay in the Streamlit script. The existing injected CSS block gains typography and component classes; small helper functions render escaped HTML for status, cards, risk pills and the ranked table. No new dependencies or data changes.

**Tech Stack:** Python 3.11, Streamlit 1.41, standard library `html.escape`.

---

### Task 1: Add failing UI regression test

**Files:** Modify `tests/test_ui_deps.py`

- [ ] Add this test:

```python
from streamlit.testing.v1 import AppTest

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
```

- [ ] Run `python -m pytest -q tests/test_ui_deps.py` and confirm it FAILS on the missing strings.

### Task 2: Implement the UI changes

**Files:** Modify `ui/app.py`

- [ ] Replace `import pandas as pd` with `from html import escape`.
- [ ] Add helpers above `st.set_page_config`:

```python
def risk_pill(score: float) -> str:
    """Return a colour-coded score pill."""
    if score >= 0.8:
        background, colour = "#e6f7f4", "#0f6f66"
    elif score >= 0.5:
        background, colour = "#fff4d6", "#8a5b00"
    else:
        background, colour = "#ffe8e6", "#a3261d"
    return (
        f'<span class="cp-pill" style="background:{background};color:{colour}">'
        f"{score:.3f}</span>"
    )


def render_risk_table(clinics: list[dict]) -> str:
    """Render the ranked clinic table as escaped HTML."""
    rows = []
    for clinic in clinics:
        cells = [
            f"<td>{escape(str(clinic['clinic_id']))}</td>",
            f"<td>{escape(str(clinic['name']))}</td>",
            f"<td>{escape(str(clinic['region']))}</td>",
            f"<td>{risk_pill(float(clinic['risk_score']))}</td>",
            f"<td>{escape('; '.join(clinic['reasons']))}</td>",
        ]
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<table class="cp-table"><thead><tr>'
        "<th>Clinic</th><th>Name</th><th>Region</th><th>Risk</th><th>Reasons</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )
```

- [ ] Extend the existing CSS string with typography and component classes:

```css
body, .stApp { font-family:'Aptos','Segoe UI',sans-serif; }
.cp-status { display:flex; gap:.75rem; align-items:center; margin-bottom:1rem; }
.cp-banner { background:#e6f7f4; color:#0f4c4c; border:1px solid #c9e6e4; border-radius:.5rem; padding:.5rem .75rem; font-weight:600; }
.cp-mode { background:#f4fbfb; color:#0f6f66; border:1px solid #c9e6e4; border-radius:999px; padding:.4rem .75rem; font-size:.85rem; }
.cp-card { background:#fff; border:1px solid #c9e6e4; border-left:4px solid #0f9b8e; border-radius:.5rem; padding:.75rem; }
.cp-card-label { font-size:.75rem; text-transform:uppercase; letter-spacing:.05em; color:#39716f; }
.cp-card-value { font-size:1.4rem; font-weight:700; color:#0f4c4c; }
.cp-table { width:100%; border-collapse:collapse; background:#fff; }
.cp-table th { background:#0f4c4c; color:#fff; text-align:left; padding:.5rem; }
.cp-table td { padding:.5rem; border-bottom:1px solid #e3efee; vertical-align:top; }
.cp-pill { display:inline-block; padding:.15rem .5rem; border-radius:999px; font-weight:700; }
```

- [ ] Replace the warning/caption lines with one status markdown block:

```python
st.markdown(
    '<div class="cp-status">'
    '<div class="cp-banner">Demo uses synthetic data.</div>'
    f'<div class="cp-mode">Active mode: {escape(str(mode))}</div>'
    "</div>",
    unsafe_allow_html=True,
)
```

- [ ] Replace the `st.dataframe(...)` block with three `st.columns(3)` cards and `st.markdown(render_risk_table(risk["clinics"]), unsafe_allow_html=True)`.

### Task 3: Verify and commit

- [ ] Run `python -m pytest -q` — expected 9 passed.
- [ ] Run an `AppTest` smoke check — expected 0 exceptions, CSS/banner/cards/pills present.
- [ ] Commit and push with message `style: modern clinical UI with cards and risk pills`.
