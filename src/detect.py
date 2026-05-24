"""
Trains an Isolation Forest on behavioral features, scores every event,
and validates against seeded ground-truth labels.

Output: data/processed/scored_events.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

IN_PATH = Path(__file__).parent.parent / "data" / "processed" / "features.parquet"
OUT_DIR = Path(__file__).parent.parent / "data" / "processed"

FEATURE_COLS = [
    "hour_deviation",
    "access_count_ratio",
    "distinct_resources_ratio",
    "bytes_ratio",
    "geo_change_score",
    "failed_attempt_count",
    "is_first_time_resource",
    "is_rare_action",
]

# contamination matched to observed anomaly rate (~1%)
CONTAMINATION = 0.01


def run():
    df = pd.read_parquet(IN_PATH)

    X = df[FEATURE_COLS].fillna(0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        max_samples="auto",
        random_state=42,
    )
    model.fit(X_scaled)

    # score_samples returns negative anomaly scores: more negative = more anomalous
    raw_scores = model.score_samples(X_scaled)
    # invert so higher = more suspicious
    df["anomaly_score"] = -raw_scores
    df["predicted_anomaly"] = model.predict(X_scaled) == -1

    df = df.sort_values("anomaly_score", ascending=False).reset_index(drop=True)

    out_path = OUT_DIR / "scored_events.parquet"
    df.to_parquet(out_path, index=False)

    # ------------------------------------------------------------------
    # Validation against seeded labels
    # ------------------------------------------------------------------
    print("=" * 60)
    print("DETECTION VALIDATION")
    print("=" * 60)

    n_seeded = df["is_anomaly"].sum()
    print(f"\nSeeded anomalies : {n_seeded}")
    print(f"Model flagged    : {df['predicted_anomaly'].sum()}")

    # Detection rate at top-N (ranked by anomaly_score)
    for top_n in [50, 100, 200]:
        top = df.head(top_n)
        caught = top["is_anomaly"].sum()
        precision = caught / top_n
        recall = caught / n_seeded if n_seeded else 0
        print(f"\nTop-{top_n:>3}  |  caught {caught}/{n_seeded}  "
              f"|  precision {precision:.2f}  |  recall {recall:.2f}")

    # Per-pattern breakdown in top-200
    top200 = df.head(200)
    print("\nAnomaly-type breakdown in top-200:")
    breakdown = (
        top200[top200["is_anomaly"]]
        .groupby("anomaly_type")
        .agg(caught=("event_id", "count"))
        .join(
            df[df["is_anomaly"]].groupby("anomaly_type").agg(total=("event_id", "count"))
        )
    )
    breakdown["detection_rate"] = (breakdown["caught"] / breakdown["total"]).round(2)
    print(breakdown.to_string())

    # Overall confusion at model threshold
    tp = int((df["predicted_anomaly"] & df["is_anomaly"]).sum())
    fp = int((df["predicted_anomaly"] & ~df["is_anomaly"]).sum())
    fn = int((~df["predicted_anomaly"] & df["is_anomaly"]).sum())
    tn = int((~df["predicted_anomaly"] & ~df["is_anomaly"]).sum())
    print(f"\nConfusion matrix (model threshold):")
    print(f"  TP={tp}  FP={fp}")
    print(f"  FN={fn}  TN={tn}")
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0
    print(f"  Precision={prec:.2f}  Recall={rec:.2f}")

    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    run()
