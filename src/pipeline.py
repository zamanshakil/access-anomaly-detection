"""
Ingests three raw log CSVs (VPN, app auth, file access), normalizes them to a
unified schema, and writes data/processed/events.parquet.

Unified schema:
  event_id, user_id, timestamp (UTC datetime), source (vpn|app|file),
  country, ip, resource, action, status, bytes, session_minutes,
  is_anomaly, anomaly_type
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).parent.parent / "data" / "processed"


def load_vpn(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={
        "user": "user_id",
        "timestamp": "timestamp",
        "source_ip": "ip",
        "country": "country",
        "session_minutes": "session_minutes",
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S", utc=True)
    df["source"] = "vpn"
    df["resource"] = None
    df["action"] = None
    df["status"] = None
    df["bytes"] = None
    return df


def load_app(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={
        "username": "user_id",
        "event_time": "timestamp",
        "app_name": "resource",
        "status": "status",
        "role": "_role",
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%m/%d/%Y %H:%M:%S", utc=True)
    df["source"] = "app"
    df["ip"] = None
    df["country"] = None
    df["action"] = "auth"
    df["bytes"] = None
    df["session_minutes"] = None
    df = df.drop(columns=["_role"])
    return df


def load_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={
        "userId": "user_id",
        "ts": "timestamp",
        "resource_path": "resource",
        "bytes_transferred": "bytes",
        "action": "action",
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%dT%H%M%SZ", utc=True)
    df["source"] = "file"
    df["ip"] = None
    df["country"] = None
    df["status"] = None
    df["session_minutes"] = None
    return df


UNIFIED_COLS = [
    "event_id", "user_id", "timestamp", "source",
    "country", "ip", "resource", "action", "status",
    "bytes", "session_minutes", "is_anomaly", "anomaly_type",
]


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    vpn = load_vpn(RAW_DIR / "vpn_logs.csv")
    app = load_app(RAW_DIR / "app_logs.csv")
    fil = load_file(RAW_DIR / "file_logs.csv")

    combined = pd.concat([vpn, app, fil], ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    combined.insert(0, "event_id", combined.index.astype(str).str.zfill(6))

    # enforce column order and drop any extras
    combined = combined[UNIFIED_COLS]

    out_path = OUT_DIR / "events.parquet"
    combined.to_parquet(out_path, index=False)

    anomalies = combined["is_anomaly"].sum()
    print(f"Unified events: {len(combined):,}  |  anomalies: {anomalies} ({anomalies/len(combined)*100:.2f}%)")
    print(f"Date range: {combined['timestamp'].min().date()} → {combined['timestamp'].max().date()}")
    print(f"Users: {combined['user_id'].nunique()}")
    print(f"Written to {out_path}")


if __name__ == "__main__":
    run()
