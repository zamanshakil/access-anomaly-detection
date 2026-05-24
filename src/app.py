"""
Streamlit dashboard — ranked anomaly table, filters, and per-event "why" panel.
Run: streamlit run src/app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "scored_events.parquet"

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

FEATURE_LABELS = {
    "hour_deviation":           "Hour deviation (z-score)",
    "access_count_ratio":       "Access count ratio vs. baseline",
    "distinct_resources_ratio": "Distinct resources ratio",
    "bytes_ratio":              "Bytes ratio vs. baseline",
    "geo_change_score":         "Geo change score (0–2)",
    "failed_attempt_count":     "Failed auth attempts (24h)",
    "is_first_time_resource":   "First-time resource (0/1)",
    "is_rare_action":           "Rare action score (0–1)",
}

# Approximate "normal" upper bound per feature for the bar chart reference line
FEATURE_NORMAL = {
    "hour_deviation":           1.0,
    "access_count_ratio":       1.5,
    "distinct_resources_ratio": 6.0,
    "bytes_ratio":              2.0,
    "geo_change_score":         0.0,
    "failed_attempt_count":     1.0,
    "is_first_time_resource":   0.0,
    "is_rare_action":           0.5,
}

ANOMALY_TYPE_COLORS = {
    "none":              "#d1d5db",
    "off_hours":         "#fbbf24",
    "impossible_travel": "#f87171",
    "priv_escalation":   "#c084fc",
    "volume_exfil":      "#60a5fa",
    "failed_storm":      "#34d399",
}


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["date"] = df["timestamp"].dt.date
    return df


def score_bar(score: float, max_score: float = 0.9) -> str:
    pct = min(int(score / max_score * 100), 100)
    color = "#ef4444" if pct > 66 else "#f59e0b" if pct > 33 else "#22c55e"
    return (
        f'<div style="background:#e5e7eb;border-radius:4px;height:10px;width:100%">'
        f'<div style="background:{color};width:{pct}%;height:10px;border-radius:4px"></div>'
        f'</div>'
    )


def main():
    st.set_page_config(
        page_title="Access Anomaly Detection",
        page_icon="🔍",
        layout="wide",
    )

    st.title("Access Anomaly Detection Dashboard")
    st.caption("Isolation Forest · Behavioral baselines · 8 features per event")

    df = load_data()

    # ------------------------------------------------------------------
    # Sidebar filters
    # ------------------------------------------------------------------
    st.sidebar.header("Filters")

    users = ["All"] + sorted(df["user_id"].unique().tolist())
    sel_user = st.sidebar.selectbox("User", users)

    sources = ["All"] + sorted(df["source"].unique().tolist())
    sel_source = st.sidebar.selectbox("Source", sources)

    min_date = df["date"].min()
    max_date = df["date"].max()
    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    score_min = float(df["anomaly_score"].min())
    score_max = float(df["anomaly_score"].max())
    score_thresh = st.sidebar.slider(
        "Min anomaly score",
        min_value=round(score_min, 3),
        max_value=round(score_max, 3),
        value=round(score_min, 3),
        step=0.001,
    )

    show_only_flagged = st.sidebar.checkbox("Show only model-flagged events", value=False)
    show_ground_truth = st.sidebar.checkbox("Show ground-truth labels (eval mode)", value=True)

    # ------------------------------------------------------------------
    # Apply filters
    # ------------------------------------------------------------------
    filtered = df.copy()
    if sel_user != "All":
        filtered = filtered[filtered["user_id"] == sel_user]
    if sel_source != "All":
        filtered = filtered[filtered["source"] == sel_source]
    if len(date_range) == 2:
        filtered = filtered[
            (filtered["date"] >= date_range[0]) &
            (filtered["date"] <= date_range[1])
        ]
    filtered = filtered[filtered["anomaly_score"] >= score_thresh]
    if show_only_flagged:
        filtered = filtered[filtered["predicted_anomaly"]]

    filtered = filtered.sort_values("anomaly_score", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # KPI row
    # ------------------------------------------------------------------
    total_events = len(df)
    flagged = int(df["predicted_anomaly"].sum())
    seeded = int(df["is_anomaly"].sum())
    top100_caught = int(df.head(100)["is_anomaly"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total events", f"{total_events:,}")
    c2.metric("Model-flagged", flagged)
    c3.metric("Seeded anomalies", seeded)
    c4.metric("Caught in top-100", f"{top100_caught}/{seeded}", f"{top100_caught/seeded*100:.0f}%")

    st.divider()

    # ------------------------------------------------------------------
    # Main table + detail panel side by side
    # ------------------------------------------------------------------
    left, right = st.columns([3, 2])

    with left:
        st.subheader(f"Ranked events ({len(filtered):,} shown)")

        display_cols = ["anomaly_score", "user_id", "timestamp", "source", "resource", "action"]
        if show_ground_truth:
            display_cols += ["anomaly_type"]

        table_df = filtered[display_cols].copy()
        table_df["timestamp"] = table_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        table_df["anomaly_score"] = table_df["anomaly_score"].round(4)

        # highlight predicted anomalies
        def row_style(row):
            if show_ground_truth and row.get("anomaly_type", "none") != "none":
                return ["background-color: #fef9c3"] * len(row)
            return [""] * len(row)

        styled = table_df.head(500).style.apply(row_style, axis=1)
        event_selection = st.dataframe(
            styled,
            use_container_width=True,
            height=480,
            on_select="rerun",
            selection_mode="single-row",
        )

    with right:
        st.subheader("Why is this anomalous?")

        selected_rows = event_selection.selection.get("rows", [])
        if selected_rows:
            idx = selected_rows[0]
            event = filtered.iloc[idx]

            # header info
            st.markdown(f"**User:** `{event['user_id']}`")
            st.markdown(f"**Time:** {event['timestamp'].strftime('%Y-%m-%d %H:%M:%S %Z')}")
            st.markdown(f"**Source:** {event['source']} &nbsp;|&nbsp; **Score:** `{event['anomaly_score']:.4f}`")
            if pd.notna(event.get("resource")):
                st.markdown(f"**Resource:** `{event['resource']}`")
            if pd.notna(event.get("action")):
                st.markdown(f"**Action:** `{event['action']}`")
            if show_ground_truth and event["anomaly_type"] != "none":
                color = ANOMALY_TYPE_COLORS.get(event["anomaly_type"], "#gray")
                st.markdown(
                    f'<span style="background:{color};padding:2px 8px;border-radius:4px;'
                    f'font-size:0.85em">🚩 {event["anomaly_type"]}</span>',
                    unsafe_allow_html=True,
                )

            st.markdown("---")
            st.markdown("**Feature deviations vs. normal baseline**")

            # build feature comparison chart data
            chart_data = []
            for feat in FEATURE_COLS:
                val = float(event[feat])
                normal = FEATURE_NORMAL[feat]
                chart_data.append({
                    "Feature": FEATURE_LABELS[feat],
                    "This event": val,
                    "Normal upper bound": normal,
                })
            chart_df = pd.DataFrame(chart_data).set_index("Feature")
            st.bar_chart(chart_df, height=320)

            # plain-text summary (template-based; v2 will use an LLM here)
            flags = []
            if event["hour_deviation"] > 2:
                flags.append(f"login hour is **{event['hour_deviation']:.1f}σ** from their norm")
            if event["bytes_ratio"] > 5:
                flags.append(f"transferred **{event['bytes_ratio']:.0f}×** their usual bytes")
            if event["geo_change_score"] >= 2:
                flags.append("logged in from a **new country** within 2h of a previous login")
            if event["geo_change_score"] == 1:
                flags.append("logged in from an **unusual country**")
            if event["failed_attempt_count"] >= 5:
                flags.append(f"**{int(event['failed_attempt_count'])} failed auth** attempts in 24h")
            if event["is_first_time_resource"] == 1:
                flags.append("accessed a **resource they've never touched before**")
            if event["is_rare_action"] > 0.9:
                flags.append("performed an action they **almost never do**")
            if event["access_count_ratio"] > 3:
                flags.append(f"**{event['access_count_ratio']:.1f}×** their normal activity rate")

            if flags:
                st.markdown("**Summary:** This event was flagged because the user " + "; ".join(flags) + ".")
            else:
                st.markdown("*No single feature dominates — flagged by the combination.*")

        else:
            st.info("Click a row in the table to see why it was flagged.")

    # ------------------------------------------------------------------
    # Bottom: anomaly score distribution
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Score distribution")
    score_dist = st.columns([2, 1])
    with score_dist[0]:
        hist_df = filtered[["anomaly_score"]].copy()
        st.bar_chart(hist_df["anomaly_score"].value_counts(bins=40).sort_index())
    with score_dist[1]:
        if show_ground_truth:
            st.markdown("**Anomaly types in view**")
            type_counts = filtered[filtered["is_anomaly"]]["anomaly_type"].value_counts()
            if not type_counts.empty:
                st.dataframe(type_counts.rename("count"), use_container_width=True)
            else:
                st.write("No seeded anomalies in current filter.")


if __name__ == "__main__":
    main()
