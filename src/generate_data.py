"""Generate fully synthetic clinic and monthly order data."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SEED = 42
N_CLINICS = 300
MONTHS = [f"2025-{month:02d}" for month in range(1, 13)]
REGIONS = ["North", "South", "East", "West", "Central"]
REPS = [f"REP-{number:02d}" for number in range(1, 5)]
SIZES = ["small", "medium", "large"]
SPECIES_MIXES = ["companion-heavy", "mixed", "livestock-heavy"]
PRODUCTS = ["vaccines", "diagnostics", "nutrition"]
BASE_VALUE = {"small": 2200.0, "medium": 5200.0, "large": 9500.0}
BASE_COUNT = {"small": 3, "medium": 6, "large": 10}
SIZE_ORDINAL = {"small": 0, "medium": 1, "large": 2}
DECLINE_START_MONTH = 7
DECLINING_SHARE = 0.20

FEATURE_COLUMNS = [
    "value_trend",
    "count_trend",
    "visit_trend",
    "days_since_last_order",
    "vaccines_share_change",
    "diagnostics_share_change",
    "nutrition_share_change",
    "size_ordinal",
]


def make_clinics(rng: np.random.Generator) -> pd.DataFrame:
    """Create the clinic master table."""
    adjectives = ["Cedar", "Bright", "Willow", "Harbor", "Pine", "Meadow", "River", "Oak"]
    animals = ["Paws", "Hooves", "Tails", "Care", "Wellness", "Family"]
    rows = []
    for index in range(N_CLINICS):
        size = str(rng.choice(SIZES, p=[0.4, 0.4, 0.2]))
        rows.append(
            {
                "clinic_id": f"C-{index + 1:03d}",
                "name": f"{rng.choice(adjectives)} {rng.choice(animals)} Vet Clinic",
                "region": str(rng.choice(REGIONS)),
                "rep_id": str(rng.choice(REPS)),
                "size": size,
                "size_ordinal": SIZE_ORDINAL[size],
                "species_mix": str(rng.choice(SPECIES_MIXES)),
                "is_declining_synthetic": bool(rng.random() < DECLINING_SHARE),
            }
        )
    return pd.DataFrame(rows)


def make_orders(clinics: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Create twelve months of orders and visits per clinic."""
    rows = []
    for clinic in clinics.itertuples(index=False):
        base_value = BASE_VALUE[clinic.size] * rng.uniform(0.85, 1.15)
        base_count = BASE_COUNT[clinic.size]
        decline_rate = rng.uniform(0.10, 0.16) if clinic.is_declining_synthetic else 0.0
        base_shares = rng.dirichlet([3.0, 2.0, 2.0])
        for month_index, month in enumerate(MONTHS, start=1):
            elapsed = max(0, month_index - DECLINE_START_MONTH + 1)
            value_factor = (1.0 - decline_rate) ** elapsed
            count_factor = (1.0 - 0.6 * decline_rate) ** elapsed
            value = max(200.0, base_value * value_factor * rng.normal(1.0, 0.06))
            count = max(1, int(round(base_count * count_factor * rng.normal(1.0, 0.12))))
            visits = max(0, int(round(count * rng.uniform(0.6, 1.1) * value_factor)))
            shares = base_shares * rng.dirichlet([8.0, 8.0, 8.0])
            shares = shares / shares.sum()
            rows.append(
                {
                    "clinic_id": clinic.clinic_id,
                    "month": month,
                    "month_index": month_index,
                    "order_value": round(float(value), 2),
                    "order_count": count,
                    "visits": visits,
                    "vaccines_value": round(float(value * shares[0]), 2),
                    "diagnostics_value": round(float(value * shares[1]), 2),
                    "nutrition_value": round(float(value * shares[2]), 2),
                }
            )
    return pd.DataFrame(rows)


def _matrix(orders: pd.DataFrame, column: str, clinic_ids: list[str]) -> np.ndarray:
    """Return a clinics-by-months matrix for one order column."""
    pivot = orders.pivot(index="clinic_id", columns="month_index", values=column)
    return pivot.reindex(index=clinic_ids, columns=range(1, 13), fill_value=0.0).to_numpy()


def build_features(
    orders: pd.DataFrame, clinics: pd.DataFrame, as_of_month: int
) -> pd.DataFrame:
    """Compute features using only months up to and including as_of_month."""
    if as_of_month < 6 or as_of_month > 9:
        raise ValueError("as_of_month must be between 6 and 9")
    clinic_ids = clinics["clinic_id"].tolist()
    values = _matrix(orders, "order_value", clinic_ids)
    counts = _matrix(orders, "order_count", clinic_ids)
    visits = _matrix(orders, "visits", clinic_ids)

    recent_values = values[:, as_of_month - 3 : as_of_month]
    prior_values = values[:, as_of_month - 6 : as_of_month - 3]
    recent_counts = counts[:, as_of_month - 3 : as_of_month]
    prior_counts = counts[:, as_of_month - 6 : as_of_month - 3]
    recent_visits = visits[:, as_of_month - 3 : as_of_month]
    prior_visits = visits[:, as_of_month - 6 : as_of_month - 3]

    recent_total = np.maximum(recent_values.sum(axis=1), 1.0)
    prior_total = np.maximum(prior_values.sum(axis=1), 1.0)
    future_total = values[:, as_of_month : as_of_month + 3].sum(axis=1)

    features = pd.DataFrame(
        {
            "clinic_id": clinic_ids,
            "as_of_month": as_of_month,
            "value_trend": recent_total / prior_total - 1.0,
            "count_trend": recent_counts.sum(axis=1)
            / np.maximum(prior_counts.sum(axis=1), 1.0)
            - 1.0,
            "visit_trend": recent_visits.sum(axis=1)
            / np.maximum(prior_visits.sum(axis=1), 1.0)
            - 1.0,
            "days_since_last_order": 30.0
            / np.maximum(counts[:, as_of_month - 1], 1.0),
            "reduced_orders_next_90d": (
                future_total < 0.8 * recent_total
            ).astype(int),
        }
    )
    for product in PRODUCTS:
        product_values = _matrix(orders, f"{product}_value", clinic_ids)
        recent_share = product_values[:, as_of_month - 3 : as_of_month].sum(
            axis=1
        ) / recent_total
        prior_share = product_values[:, as_of_month - 6 : as_of_month - 3].sum(
            axis=1
        ) / prior_total
        features[f"{product}_share_change"] = recent_share - prior_share
    features = features.merge(
        clinics[["clinic_id", "size_ordinal"]], on="clinic_id", how="left"
    )
    return features


def make_snapshots(orders: pd.DataFrame, clinics: pd.DataFrame) -> pd.DataFrame:
    """Build one labelled snapshot per clinic for each valid as-of month."""
    frames = [build_features(orders, clinics, month) for month in range(8, 10)]
    return pd.concat(frames, ignore_index=True)


def save_sqlite(clinics: pd.DataFrame, orders: pd.DataFrame, snapshots: pd.DataFrame) -> Path:
    """Persist the synthetic tables to SQLite."""
    path = DATA_DIR / "clinic_pulse.db"
    with sqlite3.connect(path) as connection:
        clinics.to_sql("clinics", connection, if_exists="replace", index=False)
        orders.to_sql("orders", connection, if_exists="replace", index=False)
        snapshots.to_sql("snapshots", connection, if_exists="replace", index=False)
    return path


def main() -> None:
    """Generate and save all synthetic data files."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    clinics = make_clinics(rng)
    orders = make_orders(clinics, rng)
    snapshots = make_snapshots(orders, clinics)
    clinics.to_csv(DATA_DIR / "clinics.csv", index=False)
    orders.to_csv(DATA_DIR / "orders.csv", index=False)
    snapshots.to_csv(DATA_DIR / "snapshots.csv", index=False)
    database = save_sqlite(clinics, orders, snapshots)
    print(f"Clinics: {len(clinics)}")
    print(f"Monthly order rows: {len(orders)}")
    print(f"Labelled snapshots: {len(snapshots)}")
    print(f"Positive label rate: {snapshots['reduced_orders_next_90d'].mean():.1%}")
    print(f"SQLite: {database}")


if __name__ == "__main__":
    main()
