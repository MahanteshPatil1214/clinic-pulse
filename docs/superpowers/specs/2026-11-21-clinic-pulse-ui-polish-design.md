# Clinic Pulse UI Polish — Design Spec

Date: 2026-11-21

## Purpose

Improve the Streamlit UI's look and scannability without changing any data, tools, agent behaviour, or dependencies. Keep the full-width layout added previously.

## Decisions from brainstorming

- Visual direction: **A. Modern clinical** — Aptos/Segoe UI font, teal accent, soft cards.
- Selected improvements: summary cards, colour-coded risk scores, styled status row.
- Not selected: suggested-question chips.
- Approach: extend the existing injected CSS and render the at-risk table as escaped HTML so score pills and card styling are fully controllable offline. This deliberately gives up native dataframe sorting.

## Scope

Only `ui/app.py` changes. No new dependencies, no new files, no changes to the banner wording ("Demo uses synthetic data."), mode reporting, chat flow, feedback, or expander content.

## Design

1. **Global style:** in the existing `st.markdown(..., unsafe_allow_html=True)` block, set `font-family: 'Aptos', 'Segoe UI', sans-serif` on body-like selectors, define teal accent colours, and add reusable classes for cards, pills, banner and mode badge. Keep the existing full-width rules for `.block-container`, `.stChatMessage` and the bottom container.
2. **Status row:** replace `st.warning(...)` and `st.caption(...)` with one HTML block containing the synthetic-data banner and a mode badge showing `Active mode: <mode>`. Keep the same visible text so existing UI checks still pass.
3. **Summary cards:** after the rep selector, use `st.columns(3)` with HTML cards showing number of clinics in the ranked list, highest risk score, and average risk score. Values come only from the existing `risk["clinics"]` result; format scores to three decimals.
4. **Risk table:** replace `st.dataframe` with an escaped HTML table. Columns remain clinic id, name, region, risk score, reasons. The score cell is a pill: teal for `>= 0.8`, amber for `0.5-0.79`, red for `< 0.5`. Escape all data with `html.escape` before inserting into markup.
5. **Error path:** if `get_at_risk_clinics` fails, show `st.error(...)` and skip cards/table as today.

## Verification

- `AppTest.from_file('ui/app.py').run()` reports 0 exceptions, contains the injected CSS, the synthetic-data banner text, three summary-card labels, and the risk pill markup.
- `pytest -q` passes all existing tests.
- Commit and push the change.
