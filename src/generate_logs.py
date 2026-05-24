"""
Generates three synthetic log sources (VPN, app auth, file access) with seeded
labeled anomalies. Output: data/raw/vpn_logs.csv, app_logs.csv, file_logs.csv
"""

import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
rng = np.random.default_rng(42)
random.seed(42)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SIM_DAYS = 60
START_DATE = datetime(2026, 1, 1)
END_DATE = START_DATE + timedelta(days=SIM_DAYS)
OUT_DIR = Path(__file__).parent.parent / "data" / "raw"

COUNTRIES_COMMON = ["US", "US", "US", "CA", "GB", "DE"]  # weighted toward US
APPS = ["Jira", "Confluence", "GitHub", "Salesforce", "DataWarehouse", "AdminPortal"]
RESOURCES = [f"/data/project_{i}" for i in range(1, 20)] + \
            [f"/reports/q{q}" for q in range(1, 9)] + \
            [f"/admin/config_{i}" for i in range(1, 6)]
ACTIONS = ["read", "read", "read", "write", "delete"]  # weighted toward read

# ---------------------------------------------------------------------------
# User personas
# ---------------------------------------------------------------------------
USERS = [
    # (user_id, persona, typical_hour_mean, typical_hour_std, daily_events_mean,
    #  bytes_mean, bytes_std, home_country)
    {"id": "alice",    "persona": "office_worker",  "hour_mean": 10,  "hour_std": 2,  "daily_events": 12, "bytes_mean": 5_000,    "bytes_std": 2_000,   "country": "US"},
    {"id": "bob",      "persona": "office_worker",  "hour_mean": 9,   "hour_std": 2,  "daily_events": 8,  "bytes_mean": 4_000,    "bytes_std": 1_500,   "country": "US"},
    {"id": "carol",    "persona": "office_worker",  "hour_mean": 11,  "hour_std": 2,  "daily_events": 10, "bytes_mean": 6_000,    "bytes_std": 2_500,   "country": "CA"},
    {"id": "nightadm", "persona": "night_admin",    "hour_mean": 2,   "hour_std": 1,  "daily_events": 15, "bytes_mean": 8_000,    "bytes_std": 3_000,   "country": "US"},
    {"id": "analyst1", "persona": "data_analyst",   "hour_mean": 10,  "hour_std": 3,  "daily_events": 40, "bytes_mean": 500_000,  "bytes_std": 200_000, "country": "US"},
    {"id": "analyst2", "persona": "data_analyst",   "hour_mean": 9,   "hour_std": 2,  "daily_events": 35, "bytes_mean": 400_000,  "bytes_std": 150_000, "country": "GB"},
    {"id": "lowuser1", "persona": "low_activity",   "hour_mean": 10,  "hour_std": 3,  "daily_events": 3,  "bytes_mean": 1_000,    "bytes_std": 500,     "country": "US"},
    {"id": "lowuser2", "persona": "low_activity",   "hour_mean": 14,  "hour_std": 3,  "daily_events": 2,  "bytes_mean": 800,      "bytes_std": 300,     "country": "US"},
    {"id": "lowuser3", "persona": "low_activity",   "hour_mean": 11,  "hour_std": 2,  "daily_events": 4,  "bytes_mean": 1_200,    "bytes_std": 600,     "country": "DE"},
    {"id": "dev1",     "persona": "developer",      "hour_mean": 10,  "hour_std": 3,  "daily_events": 20, "bytes_mean": 20_000,   "bytes_std": 8_000,   "country": "US"},
    {"id": "dev2",     "persona": "developer",      "hour_mean": 11,  "hour_std": 3,  "daily_events": 18, "bytes_mean": 18_000,   "bytes_std": 7_000,   "country": "US"},
]
USER_MAP = {u["id"]: u for u in USERS}

# Each user has a stable preferred resource pool (subset of RESOURCES)
USER_RESOURCES = {
    u["id"]: random.sample(RESOURCES, k=random.randint(4, 10))
    for u in USERS
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rand_ts(date: datetime, hour_mean: float, hour_std: float) -> datetime:
    hour = int(np.clip(rng.normal(hour_mean, hour_std), 0, 23))
    minute = rng.integers(0, 60)
    second = rng.integers(0, 60)
    return date.replace(hour=hour, minute=int(minute), second=int(second), microsecond=0)


def rand_ip(country: str) -> str:
    prefixes = {"US": "10", "CA": "172", "GB": "192", "DE": "185"}
    p = prefixes.get(country, "10")
    return f"{p}.{rng.integers(0,255)}.{rng.integers(0,255)}.{rng.integers(1,254)}"


# ---------------------------------------------------------------------------
# Normal event generators (one row per call)
# ---------------------------------------------------------------------------

def vpn_row(user: dict, ts: datetime, country: Optional[str] = None) -> dict:
    c = country or user["country"]
    return {
        "user": user["id"],
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "source_ip": rand_ip(c),
        "country": c,
        "session_minutes": max(1, int(rng.normal(45, 20))),
    }


def app_row(user: dict, ts: datetime, status: str = "success") -> dict:
    app = random.choice(APPS)
    return {
        "username": user["id"],
        "event_time": ts.strftime("%m/%d/%Y %H:%M:%S"),  # intentionally different format
        "app_name": app,
        "status": status,
        "role": user["persona"],
    }


def file_row(user: dict, ts: datetime, resource: Optional[str] = None,
             action: Optional[str] = None, bytes_val: Optional[int] = None) -> dict:
    res = resource or random.choice(USER_RESOURCES[user["id"]])
    act = action or random.choices(["read", "write", "delete"], weights=[7, 2, 1])[0]
    bv = bytes_val if bytes_val is not None else max(
        100, int(rng.normal(user["bytes_mean"], user["bytes_std"]))
    )
    return {
        "userId": user["id"],
        "ts": ts.strftime("%Y%m%dT%H%M%SZ"),  # yet another format
        "resource_path": res,
        "bytes_transferred": bv,
        "action": act,
        "is_anomaly": False,
        "anomaly_type": "none",
    }


# ---------------------------------------------------------------------------
# Anomaly injectors
# ---------------------------------------------------------------------------

def inject_off_hours(user: dict, day: datetime) -> tuple[dict, dict]:
    """Office-hours user logs in at 3am."""
    ts = day.replace(hour=3, minute=rng.integers(0, 30).item(), second=0)
    return (
        vpn_row(user, ts) | {"is_anomaly": True, "anomaly_type": "off_hours"},
        app_row(user, ts + timedelta(minutes=2)) | {"is_anomaly": True, "anomaly_type": "off_hours"},
    )


def inject_impossible_travel(user: dict, day: datetime) -> tuple[dict, dict]:
    """Two VPN logins from different countries 10 minutes apart."""
    ts1 = rand_ts(day, user["hour_mean"], 1)
    ts2 = ts1 + timedelta(minutes=10)
    foreign = random.choice([c for c in ["JP", "BR", "AU", "ZA"] if c != user["country"]])
    r1 = vpn_row(user, ts1) | {"is_anomaly": True, "anomaly_type": "impossible_travel"}
    r2 = vpn_row(user, ts2, country=foreign) | {"is_anomaly": True, "anomaly_type": "impossible_travel"}
    return r1, r2


def inject_priv_escalation(user: dict, day: datetime) -> list[dict]:
    """Burst of deletes on resources the user never touches."""
    unusual_resources = [r for r in RESOURCES if r not in USER_RESOURCES[user["id"]]]
    rows = []
    base_ts = rand_ts(day, user["hour_mean"], 1)
    for i in range(8):
        ts = base_ts + timedelta(minutes=i * 2)
        res = random.choice(unusual_resources)
        rows.append(
            file_row(user, ts, resource=res, action="delete") |
            {"is_anomaly": True, "anomaly_type": "priv_escalation"}
        )
    return rows


def inject_volume_exfil(user: dict, day: datetime) -> list[dict]:
    """Bytes 50× normal average."""
    rows = []
    base_ts = rand_ts(day, user["hour_mean"], 1)
    for i in range(5):
        ts = base_ts + timedelta(minutes=i * 3)
        big_bytes = int(user["bytes_mean"] * rng.uniform(40, 60))
        rows.append(
            file_row(user, ts, bytes_val=big_bytes) |
            {"is_anomaly": True, "anomaly_type": "volume_exfil"}
        )
    return rows


def inject_failed_storm(user: dict, day: datetime) -> list[dict]:
    """Many failed auth attempts followed by a success."""
    rows = []
    base_ts = rand_ts(day, user["hour_mean"], 1)
    for i in range(12):
        ts = base_ts + timedelta(minutes=i)
        rows.append(
            app_row(user, ts, status="fail") |
            {"is_anomaly": True, "anomaly_type": "failed_storm"}
        )
    # final success
    rows.append(
        app_row(user, base_ts + timedelta(minutes=13), status="success") |
        {"is_anomaly": True, "anomaly_type": "failed_storm"}
    )
    return rows


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    vpn_rows, app_rows, file_rows = [], [], []

    days = [START_DATE + timedelta(days=d) for d in range(SIM_DAYS)]

    # --- Normal activity ---
    for user in USERS:
        for day in days:
            if day.weekday() >= 5 and user["persona"] not in ("night_admin", "data_analyst"):
                if rng.random() > 0.15:
                    continue  # most personas skip weekends

            n_events = max(1, int(rng.poisson(user["daily_events"])))
            for _ in range(n_events):
                ts = rand_ts(day, user["hour_mean"], user["hour_std"])
                source = rng.choice(["vpn", "app", "file"], p=[0.25, 0.35, 0.40])
                if source == "vpn":
                    vpn_rows.append(vpn_row(user, ts) | {"is_anomaly": False, "anomaly_type": "none"})
                elif source == "app":
                    status = "fail" if rng.random() < 0.05 else "success"
                    app_rows.append(app_row(user, ts, status=status) | {"is_anomaly": False, "anomaly_type": "none"})
                else:
                    file_rows.append(file_row(user, ts))

    # --- Anomaly injection (~1.5% of events) ---
    anomaly_schedule = [
        # (pattern, user_id, day_offset)
        ("off_hours",          "alice",    10),
        ("off_hours",          "bob",      25),
        ("off_hours",          "carol",    40),
        ("impossible_travel",  "alice",    15),
        ("impossible_travel",  "dev1",     30),
        ("impossible_travel",  "lowuser1", 45),
        ("priv_escalation",    "bob",      20),
        ("priv_escalation",    "carol",    35),
        ("volume_exfil",       "analyst1", 18),
        ("volume_exfil",       "lowuser2", 33),
        ("volume_exfil",       "dev2",     50),
        ("failed_storm",       "alice",    22),
        ("failed_storm",       "lowuser3", 38),
        ("failed_storm",       "bob",      55),
    ]

    for pattern, uid, day_offset in anomaly_schedule:
        if day_offset >= SIM_DAYS:
            continue
        user = USER_MAP[uid]
        day = START_DATE + timedelta(days=day_offset)

        if pattern == "off_hours":
            r1, r2 = inject_off_hours(user, day)
            vpn_rows.append(r1)
            app_rows.append(r2)
        elif pattern == "impossible_travel":
            r1, r2 = inject_impossible_travel(user, day)
            vpn_rows.extend([r1, r2])
        elif pattern == "priv_escalation":
            file_rows.extend(inject_priv_escalation(user, day))
        elif pattern == "volume_exfil":
            file_rows.extend(inject_volume_exfil(user, day))
        elif pattern == "failed_storm":
            app_rows.extend(inject_failed_storm(user, day))

    # --- Write CSVs ---
    vpn_df = pd.DataFrame(vpn_rows)
    app_df = pd.DataFrame(app_rows)
    file_df = pd.DataFrame(file_rows)

    vpn_df.to_csv(OUT_DIR / "vpn_logs.csv", index=False)
    app_df.to_csv(OUT_DIR / "app_logs.csv", index=False)
    file_df.to_csv(OUT_DIR / "file_logs.csv", index=False)

    total = len(vpn_df) + len(app_df) + len(file_df)
    anomalies = (
        vpn_df["is_anomaly"].sum() +
        app_df["is_anomaly"].sum() +
        file_df["is_anomaly"].sum()
    )
    print(f"Generated {total:,} events  |  anomalies: {anomalies} ({anomalies/total*100:.2f}%)")
    print(f"  VPN rows : {len(vpn_df):,}")
    print(f"  App rows : {len(app_df):,}")
    print(f"  File rows: {len(file_df):,}")
    print(f"Written to {OUT_DIR}/")


if __name__ == "__main__":
    generate()
