# Login Anomaly Detector

Detects three login attack patterns in a synthetic auth-log dataset:
`impossible_travel`, `new_device_odd_hour`, and `credential_stuffing`.

## How it works

An IsolationForest scores each login on six behavioral features: geo-velocity
between consecutive logins, device novelty, per-user hour-of-day deviation,
a novelty/hour interaction term, an explicit "novel device + unusual hour"
conjunction feature, and account-wide login burst rate.

`hour_deviation` and the conjunction feature are computed against each
user's own login history (median hour, circular standard deviation) rather
than a single global baseline, so users with legitimately unusual but
consistent schedules aren't flagged on every login.

An event is alerted on if either the IsolationForest score crosses the
trained threshold, or the novel-device-and-unusual-hour conjunction fires
on its own — that feature is near-zero for legitimate traffic by
construction, so a single shared anomaly-score threshold shouldn't be the
only thing standing between it and an alert.

## Usage

```
make install   # create .venv and install requirements.txt
make data      # generate data/logs.csv and data/labels.csv
make eval      # train + evaluate, report precision/recall/F1 per attack type
make serve     # run the FastAPI scoring endpoint (see serve.py)
```

## Layout

- `src/generate_logs.py` — synthetic login/attack data generator
- `src/models.py` — `LoginEvent` and `UserHistory`
- `src/features.py` — feature extraction functions
- `evals/run_eval.py` — training + evaluation harness
- `serve.py` — FastAPI scoring endpoint (note: no persisted per-user
  history across requests yet — see the note in `serve.py`)
