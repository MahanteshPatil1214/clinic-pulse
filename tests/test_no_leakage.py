"""Tests for leakage-free feature construction."""

import pandas as pd

import generate_data as gd


def test_features_use_only_past_months() -> None:
    """Truncating future months must not change any feature value."""
    clinics = pd.read_csv(gd.DATA_DIR / "clinics.csv")
    orders = pd.read_csv(gd.DATA_DIR / "orders.csv")
    snapshots = pd.read_csv(gd.DATA_DIR / "snapshots.csv")
    for as_of_month in (8, 9):
        truncated = orders[orders["month_index"] <= as_of_month]
        recomputed = gd.build_features(truncated, clinics, as_of_month)
        saved = snapshots[snapshots["as_of_month"] == as_of_month]
        pd.testing.assert_frame_equal(
            recomputed[gd.FEATURE_COLUMNS].reset_index(drop=True),
            saved[gd.FEATURE_COLUMNS].reset_index(drop=True),
            rtol=1e-5,
            atol=1e-5,
        )


def test_label_uses_future_months() -> None:
    """The label must come from the three months after the feature cutoff."""
    clinics = pd.read_csv(gd.DATA_DIR / "clinics.csv")
    orders = pd.read_csv(gd.DATA_DIR / "orders.csv")
    snapshots = pd.read_csv(gd.DATA_DIR / "snapshots.csv")
    computed = gd.build_features(orders, clinics, 9)
    saved = snapshots[snapshots["as_of_month"] == 9]
    assert (
        computed["reduced_orders_next_90d"].to_numpy()
        == saved["reduced_orders_next_90d"].to_numpy()
    ).all()
