# Access Log Anomaly Detection & AI Investigation System

**A self-contained blueprint.** If you've been away from this project for days or weeks, read this top to bottom and you'll know exactly where you are and what to do next. It's also written so you (or Claude Code) can pick up any section and implement it without re-deriving the design.

---

## 0. TL;DR — What this is

A system that ingests access logs from multiple sources, engineers *behavioral* features for each user, scores every access event for "weirdness" using an unsupervised Isolation Forest model, and surfaces the most suspicious events on an interactive dashboard.

**One-line data flow:**

```
Raw multi-source logs  →  normalize to unified schema  →  aggregate into
per-user behavioral features  →  Isolation Forest scores each event  →
dashboard ranks & visualizes the suspicious ones
```

**v1 scope (this document):** synthetic data with seeded labeled anomalies, pipeline, features, detection, dashboard.
**Deferred to v2:** the LLM explanation layer (intentionally skipped for v1 — see §9).

---

## 1. Why this project exists (and why these choices)

This is a portfolio project demonstrating data-engineering + ML skills: multi-source ingestion, feature engineering, unsupervised anomaly detection, and a presentation layer. The design choices below are deliberate, and each one answers a question a technical interviewer is likely to ask. **If you can explain the "why" in each section, you can defend the whole project.**

The guiding principle behind the whole thing: **anomaly is relative to a user's own baseline, not to a global rule.** "Logged in at 3am" is normal for a night-shift admin; "logged in at 3am when *this user* never has before" is the anomaly. Most of the cleverness lives in encoding that idea as features.

---

## 2. Architecture overview

Four stages, each a separate module so the repo stays maintainable and each piece is independently testable.

| Stage | Module | Responsibility | Key interview question it answers |
|-------|--------|----------------|-----------------------------------|
| 1. Ingestion | `pipeline.py` | Read multiple log formats, normalize to one schema | "What made this *data engineering*, not just a notebook?" |
| 2. Features | `features.py` | Turn raw events into per-user behavioral features | "How did you represent 'suspicious' numerically?" |
| 3. Detection | `detect.py` | Score events with Isolation Forest | "Why this algorithm? How did you tune it?" |
| 4. Dashboard | `app.py` | Rank, filter, and visualize anomalies | "How would a human actually use this?" |
| (support) | `generate_logs.py` | Produce synthetic logs with seeded anomalies | "How did you validate detection?" |

---

## 3. Repo structure

```
access-anomaly-detection/
├── README.md                  # public-facing: what it is, how to run, screenshots
├── PROJECT_BLUEPRINT.md       # this file (your private working doc)
├── requirements.txt
├── .gitignore                 # ignore data/raw, data/processed, __pycache__, venv
├── data/
│   ├── raw/                   # generated multi-source logs land here
│   └── processed/             # unified schema + feature tables
├── src/
│   ├── generate_logs.py       # synthetic data with seeded labeled anomalies
│   ├── pipeline.py            # ingestion + normalization
│   ├── features.py            # behavioral feature engineering
│   ├── detect.py              # Isolation Forest training + scoring
│   └── app.py                 # Streamlit dashboard
└── notebooks/                 # optional: exploration / validation scratch
```

**Why modular:** when asked "how was it structured," separate files signal you think about maintainability. Each module reads its input from disk and writes its output to disk, so you can run and debug stages independently.

---

## 4. Stage 1 — Synthetic data generation (`generate_logs.py`)

You can't build anything without data, and generating it well *forces* you to define what normal vs. anomalous looks like — which is the feature engineering in disguise.

### Approach: labeled injection

Generate mostly-normal behavior, then inject a handful of **known** anomaly patterns. Keep a ground-truth label column so you can later measure whether the detector catches what you planted. This is the single most valuable thing for the interview: *"I seeded known anomalies and measured detection rate"* beats *"it flagged some stuff."*

### Three simulated sources (deliberately different shapes)

The point of three sources is to create a realistic normalization problem in Stage 2.

1. **VPN log** — columns like `user`, `timestamp`, `source_ip`, `country`, `session_minutes`.
2. **Application auth log** — columns like `username`, `event_time`, `app_name`, `status` (success/fail), `role`.
3. **File-access log** — columns like `userId`, `ts`, `resource_path`, `bytes_transferred`, `action` (read/write/delete).

Note the intentional inconsistencies: the user field is named differently in each (`user` / `username` / `userId`), and timestamps use different formats. This is the realism that makes Stage 1 (normalization) a real task.

### The cast of users

Create ~20–50 synthetic users with stable personalities so "normal" is consistent per user:
- A typical 9–5 office worker (predictable hours, few resources).
- A night-shift admin (works 11pm–7am normally — the trap for naive rules).
- A high-volume data analyst (touches many resources, large transfers — normal for them).
- A handful of low-activity users.

Generate, say, 30–90 days of activity so per-user baselines are meaningful.

### Anomaly patterns to seed (label each with the pattern name)

| Pattern | What it looks like | Which features should catch it |
|---------|--------------------|--------------------------------|
| Off-hours access | A 9–5 user logs in at 3am | login-hour-vs-baseline |
| Impossible travel | Two logins from different countries minutes apart | geo/IP change vs. last login |
| Privilege escalation burst | Sudden spike in distinct resources / a `delete` they never do | distinct-resource count, first-time-action |
| Volume exfiltration | `bytes_transferred` 50× their norm | data-volume-vs-baseline |
| Failed-attempt storm | Many failed auths then a success | failed-attempt count |

Aim for anomalies to be **~1–2% of all events** — rare, like reality. Store the label as `is_anomaly` + `anomaly_type` (use `none` for normal). The detector never sees these columns; they're only for evaluation.

**Output:** three raw CSVs in `data/raw/`, each in its own format.

---

## 5. Stage 2 — Pipeline / normalization (`pipeline.py`)

### Job
Read the three differently-shaped raw logs and emit one **unified event table** with a consistent schema.

### Target unified schema
```
event_id, user_id, timestamp (parsed, UTC), source (vpn|app|file),
country, ip, resource, action, status, bytes, session_minutes
```
Fields not present in a given source are null (e.g. VPN rows have no `resource`). That's expected and fine.

### Steps
1. Load each CSV.
2. Rename source-specific columns to the unified names (`username`/`userId` → `user_id`, etc.).
3. Parse the differing timestamp formats into real datetime objects.
4. Add the `source` column.
5. Concatenate, sort by `timestamp`, assign `event_id`.
6. Write to `data/processed/events.parquet` (or CSV).

### Interview hook
This is your answer to *"what made it data engineering?"* — the schema you designed and the normalization of heterogeneous sources. Be ready to say why you chose your unified schema and how you handled missing fields.

---

## 6. Stage 3 — Feature engineering (`features.py`)

**The heart of the project. Spend your best thinking time here.**

Isolation Forest sees numbers, not "logins." You translate each event into behavioral features that *encode* suspiciousness. **The key move: features are relative to each user's own baseline.**

### Two-pass approach
1. **Build per-user baselines** from history: typical login hours (mean/std or distribution), rolling average of daily access count, average bytes transferred, set of previously-seen resources, set of previously-seen countries.
2. **Compute features per event** relative to that baseline.

### Candidate features (start here, justify each)
- `hour_deviation` — how far this event's hour is from the user's typical hours (e.g. z-score against their hour distribution).
- `access_count_ratio` — events in the surrounding window vs. the user's rolling average.
- `distinct_resources_today` — count vs. baseline.
- `bytes_ratio` — this event's bytes vs. the user's average.
- `geo_change` — boolean/score: is `country` different from recent logins, and is the time gap physically implausible (impossible travel)?
- `failed_attempt_count` — failures in the recent window.
- `is_first_time_resource` — has this user ever touched this resource before?
- `is_rare_action` — does this user normally do this action (e.g. `delete`)?

### Watch-outs (good things to mention in interview)
- **Cold start:** new users have no baseline. Decide how to handle (flag as low-confidence, or use a global baseline until enough history accrues).
- **Leakage:** when computing a baseline for an event, only use data *before* that event — otherwise the anomaly contaminates its own baseline.
- **Scaling:** Isolation Forest is fairly robust to scale, but standardizing features keeps things clean.

**Output:** `data/processed/features.parquet` — one row per event, feature columns + carried-through `event_id`, `user_id`, `timestamp`, and the held-aside `is_anomaly`/`anomaly_type` for evaluation only.

---

## 7. Stage 4 — Detection (`detect.py`)

### Algorithm: Isolation Forest — and why
- **Unsupervised**, which is the *honest* reason: you have no labeled breach data in the real world. (You seeded labels only to validate.)
- **How it works (one line you can say):** it randomly partitions the feature space; anomalies get isolated in fewer splits because they sit far from the dense normal region, so they're "easy to cut off."
- **Scales well** and needs little tuning vs. alternatives.

### Alternatives to be able to contrast
- **Z-score / statistical threshold:** too rigid, single-variable, misses multivariate weirdness.
- **DBSCAN / clustering:** viable, but more parameter-sensitive and scales worse.

### Knobs to have opinions on
- `contamination` ≈ 0.01–0.02 — set low because real anomalies are rare. Be ready to explain you matched it to your seeded rate.
- `n_estimators`, `max_samples` — defaults are fine for v1; mention you'd tune if scaling up.
- **Feature set** — the biggest lever is Stage 6, not hyperparameters.

### Steps
1. Load `features.parquet`, drop the label columns before fitting.
2. Fit Isolation Forest, get an anomaly score per event (use `decision_function` / `score_samples` for a continuous score, not just the binary label — you want a *ranking*).
3. Attach scores back to events.
4. **Validate against seeded labels:** since you planted ground truth, compute detection rate / precision at top-N, and a confusion-style breakdown by `anomaly_type`. This is your money slide.
5. Write `data/processed/scored_events.parquet`.

### Interview hook
"I seeded ~1.5% known anomalies across five patterns, set contamination to match, and measured that the model caught X% of them in the top-N ranked events, with impossible-travel and volume-exfil being easiest and off-hours hardest." That sentence alone signals you actually did this.

---

## 8. Stage 5 — Dashboard (`app.py`, Streamlit)

### Purpose
Turn scores into something a human analyst would trust and act on.

### Minimum viable views
1. **Ranked anomaly table** — most suspicious events first, with score, user, time, source, and key feature values.
2. **Filters** — by user, time range, source, anomaly score threshold.
3. **"Why is this anomalous?" panel** — for a selected event, show its feature values against the user's normal range (e.g. a small bar/range chart per feature). This is what makes the score explainable.
4. *(Optional)* a timeline / activity overview per user.

### Note for later
The "why" panel is the natural seam where the **v2 LLM explanation layer** plugs in — it would take the feature deviations and write a human-readable sentence ("This event was flagged because the user accessed 12× their normal data volume from a new country at an unusual hour"). Build the panel so that swap is easy.

---

## 9. v2 — LLM explanation layer (DEFERRED, do not build in v1)

Intentionally skipped for v1 because it adds the least learning per hour and is trivial to bolt on. When you return to it:
- Input: the top feature deviations for a flagged event.
- Output: a plain-English explanation.
- Start with a **template-based stub** (string formatting from the feature deltas) so the system is fully functional, then swap in a real model call behind the same interface.

---

## 10. Tech stack & setup

- **Python 3.11+**
- **pandas** — data manipulation
- **scikit-learn** — Isolation Forest
- **numpy** — numerics
- **streamlit** — dashboard
- **pyarrow** — parquet I/O (optional; CSV works too)
- **faker** *(optional)* — nicer synthetic identities

**requirements.txt** (starting point):
```
pandas
numpy
scikit-learn
streamlit
pyarrow
faker
```

**First-time setup:**
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Run order:**
```bash
python src/generate_logs.py     # writes data/raw/*.csv
python src/pipeline.py          # writes data/processed/events.parquet
python src/features.py          # writes data/processed/features.parquet
python src/detect.py            # writes data/processed/scored_events.parquet
streamlit run src/app.py        # opens the dashboard
```

---

## 11. Build plan — two weeks of evenings

| Phase | Evenings | What you finish |
|-------|----------|-----------------|
| A | 1–3 | `generate_logs.py` + `pipeline.py`: realistic seeded data, unified schema. You can *see* the data. |
| B | 4–7 | `features.py` + `detect.py`: the meaty part. Baselines, features, scoring, validation against seeded labels. |
| C | 8–10 | `app.py`: ranked table, filters, the "why" panel. |
| Buffer | 11–14 | README, screenshots, polish, GitHub publish. LLM layer only if time. |

**Definition of done for v1:** all five run-order commands work end to end; the dashboard shows ranked anomalies; you can state your detection rate against seeded labels; README explains it to a stranger.

---

## 12. Publishing to GitHub

1. `git init`, add `.gitignore` (ignore `venv/`, `__pycache__/`, `data/raw/`, `data/processed/` — keep the repo light; people regenerate data by running the scripts).
2. Write the **README** for a human who's never seen it: one-paragraph what-it-is, the data-flow diagram from §0, setup + run order, a screenshot or two of the dashboard, and a short "design decisions" section (lift the "why" points from §4–§7 — this is what makes a recruiter linger).
3. Commit in logical chunks (one per module) so the history tells a story.
4. Push to a public repo; put the link on your resume and LinkedIn.

---

## 13. Interview prep — the questions this project must let you answer

Rehearse these out loud; if you can answer them, the project is doing its job:
1. Why Isolation Forest over a simple threshold or clustering?
2. How did you represent "suspicious" — what features, and why *relative to the user*?
3. What made this a data-engineering project? (normalization, multi-source schema)
4. How did you know it actually worked? (seeded labels, detection rate by anomaly type)
5. What's the contamination parameter and how did you set it?
6. How would you handle a brand-new user with no history? (cold start)
7. How would you avoid the anomaly contaminating its own baseline? (temporal leakage)
8. What would you do next / at production scale? (streaming, the LLM layer, retraining cadence)

---

## 14. Where am I right now? (update this when you stop)

> **Status:** Blueprint complete. No code written yet. Next action: build `generate_logs.py` (Phase A, evening 1).
>
> _Update this line each time you stop working so future-you knows where to resume._
