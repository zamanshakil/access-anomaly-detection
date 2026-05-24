"""
Builds per-user behavioral baselines and computes 8 features per event.
All baselines are computed using only data *before* the event (no leakage).

Output: data/processed/features.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd

IN_PATH = Path(__file__).parent.parent / "data" / "processed" / "events.parquet"
OUT_DIR = Path(__file__).parent.parent / "data" / "processed"

# Rolling window for "recent" context (in hours)
RECENT_WINDOW_HOURS = 24
BASELINE_MIN_EVENTS = 5  # min events before we trust a user's baseline


# ---------------------------------------------------------------------------
# Feature helpers — each takes the event row + all prior events for that user
# ---------------------------------------------------------------------------

def hour_deviation(event_hour: int, prior: pd.DataFrame) -> float:
    """Z-score of this event's hour vs. the user's historical hour distribution."""
    if len(prior) < BASELINE_MIN_EVENTS:
        return 0.0
    hours = prior["hour"]
    mu, sigma = hours.mean(), hours.std()
    if sigma < 0.5:
        sigma = 0.5
    return abs(event_hour - mu) / sigma


def access_count_ratio(ts: pd.Timestamp, prior: pd.DataFrame) -> float:
    """Events in the last 24h vs. the user's rolling daily average."""
    if len(prior) < BASELINE_MIN_EVENTS:
        return 1.0
    window_start = ts - pd.Timedelta(hours=RECENT_WINDOW_HOURS)
    recent_count = (prior["timestamp"] >= window_start).sum()

    # rolling daily average: total prior events / days of history
    days = max(1, (ts - prior["timestamp"].min()).total_seconds() / 86400)
    daily_avg = len(prior) / days
    if daily_avg < 1:
        daily_avg = 1.0
    return recent_count / daily_avg


def distinct_resources_ratio(ts: pd.Timestamp, prior: pd.DataFrame) -> float:
    """Distinct resources touched today vs. user's average distinct resources/day."""
    if len(prior) < BASELINE_MIN_EVENTS:
        return 1.0
    today_start = ts.normalize()
    today_resources = prior.loc[prior["timestamp"] >= today_start, "resource"].dropna().nunique()

    days = max(1, (ts - prior["timestamp"].min()).total_seconds() / 86400)
    avg_daily_resources = prior["resource"].dropna().nunique() / days
    if avg_daily_resources < 1:
        avg_daily_resources = 1.0
    return today_resources / avg_daily_resources


def bytes_ratio(event_bytes: float, prior: pd.DataFrame) -> float:
    """This event's bytes vs. the user's mean bytes (file events only)."""
    file_prior = prior[prior["source"] == "file"]["bytes"].dropna()
    if len(file_prior) < BASELINE_MIN_EVENTS:
        return 1.0
    mu = file_prior.mean()
    if mu < 1:
        mu = 1.0
    return event_bytes / mu


def geo_change_score(ts: pd.Timestamp, country: str, prior: pd.DataFrame) -> float:
    """
    Returns a score 0–2:
      +1 if country differs from user's most recent VPN country
      +1 if time gap since last VPN login is < 120 min (impossible travel proxy)
    """
    vpn_prior = prior[prior["source"] == "vpn"].sort_values("timestamp")
    if vpn_prior.empty:
        return 0.0
    last = vpn_prior.iloc[-1]
    score = 0.0
    if pd.notna(last["country"]) and last["country"] != country:
        score += 1.0
        gap_minutes = (ts - last["timestamp"]).total_seconds() / 60
        if gap_minutes < 120:
            score += 1.0
    return score


def failed_attempt_count(ts: pd.Timestamp, prior: pd.DataFrame) -> int:
    """Failed auth events in the last 24h."""
    window_start = ts - pd.Timedelta(hours=RECENT_WINDOW_HOURS)
    return int(
        ((prior["timestamp"] >= window_start) &
         (prior["source"] == "app") &
         (prior["status"] == "fail")).sum()
    )


def is_first_time_resource(resource: str, prior: pd.DataFrame) -> int:
    """1 if the user has never accessed this resource before."""
    if pd.isna(resource):
        return 0
    return int(resource not in prior["resource"].values)


def is_rare_action(action: str, prior: pd.DataFrame) -> float:
    """
    Fraction of prior events that used this action (inverted, so 0=common, ~1=never seen).
    """
    if pd.isna(action) or len(prior) == 0:
        return 0.0
    freq = (prior["action"] == action).mean()
    return 1.0 - freq


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    df = pd.read_parquet(IN_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["hour"] = df["timestamp"].dt.hour
    df = df.sort_values("timestamp").reset_index(drop=True)

    records = []

    # group by user once; iterate events in timestamp order
    for user_id, group in df.groupby("user_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)

        for i, row in group.iterrows():
            prior = group.iloc[:i]  # strictly before this event — no leakage

            ts = row["timestamp"]
            event_hour = row["hour"]

            # Only compute geo/bytes features when relevant
            geo = 0.0
            if row["source"] == "vpn" and pd.notna(row["country"]):
                vpn_prior = prior[prior["source"] == "vpn"]
                geo = geo_change_score(ts, row["country"], vpn_prior.assign(
                    # pass full prior so helper can look at vpn subset
                    **{}
                ) if not vpn_prior.empty else prior)
                # simpler: pass full prior, helper filters internally
                geo = geo_change_score(ts, row["country"], prior)

            b_ratio = 1.0
            if row["source"] == "file" and pd.notna(row["bytes"]):
                b_ratio = bytes_ratio(float(row["bytes"]), prior)

            records.append({
                "event_id": row["event_id"],
                "user_id": user_id,
                "timestamp": ts,
                "source": row["source"],
                # raw context
                "hour": event_hour,
                "bytes": row["bytes"],
                "resource": row["resource"],
                "action": row["action"],
                "status": row["status"],
                "country": row["country"],
                # features
                "hour_deviation":            hour_deviation(event_hour, prior),
                "access_count_ratio":        access_count_ratio(ts, prior),
                "distinct_resources_ratio":  distinct_resources_ratio(ts, prior),
                "bytes_ratio":               b_ratio,
                "geo_change_score":          geo,
                "failed_attempt_count":      failed_attempt_count(ts, prior),
                "is_first_time_resource":    is_first_time_resource(row["resource"], prior),
                "is_rare_action":            is_rare_action(row["action"], prior),
                # ground truth (held aside — not fed to model)
                "is_anomaly":    row["is_anomaly"],
                "anomaly_type":  row["anomaly_type"],
            })

    features_df = pd.DataFrame(records)

    FEATURE_COLS = [
        "hour_deviation", "access_count_ratio", "distinct_resources_ratio",
        "bytes_ratio", "geo_change_score", "failed_attempt_count",
        "is_first_time_resource", "is_rare_action",
    ]

    out_path = OUT_DIR / "features.parquet"
    features_df.to_parquet(out_path, index=False)

    print(f"Feature rows: {len(features_df):,}")
    print(f"\nFeature summary:")
    print(features_df[FEATURE_COLS].describe().round(3).to_string())
    print(f"\nAnomaly breakdown:")
    print(features_df.groupby("anomaly_type")["event_id"].count().sort_values(ascending=False).to_string())
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    run()
