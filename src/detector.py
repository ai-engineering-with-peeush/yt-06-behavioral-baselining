"""
detector.py — trains and saves the IsolationForest login anomaly detector.

Builds the same 6-feature vector as evals/run_eval.py (geo-velocity, device
novelty, per-user hour deviation, their interaction, the explicit
novel-device-and-unusual-hour conjunction, and burst rate), fits an
IsolationForest on clean training-window events, and persists the model
plus standardization stats to data/model.pkl.
"""

import bisect
import csv
import json
import logging
import pickle
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import LoginEvent, UserHistory
from src.features import (
    geo_velocity_kmh,
    device_novelty,
    hour_deviation,
    burst_rate,
    novel_device_and_unusual_hour,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

DATA_DIR      = Path(__file__).parent.parent / "data"
MODEL_PATH    = DATA_DIR / "model.pkl"
MODEL_VERSION = "v1"
TS_FMT        = "%Y-%m-%dT%H:%M:%S"

N_ESTIMATORS  = 200
CONTAMINATION = 0.002
RANDOM_STATE  = 42
TRAIN_DAYS    = 20
START_DATE    = datetime(2026, 5, 1)


def load_events(path: Path = DATA_DIR / "logs.csv") -> List[LoginEvent]:
    events: List[LoginEvent] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            events.append(LoginEvent(
                event_id  = row["event_id"],
                user_id   = row["user_id"],
                ts        = datetime.strptime(row["ts"], TS_FMT),
                device_id = row["device_id"],
                country   = row["country"],
                lat       = float(row["lat"]),
                lon       = float(row["lon"]),
                success   = row["success"].lower() == "true",
            ))
    return events


def build_histories(events: List[LoginEvent]) -> Dict[str, UserHistory]:
    by_user: Dict[str, List[LoginEvent]] = defaultdict(list)
    for e in events:
        by_user[e.user_id].append(e)
    return {uid: UserHistory(evts) for uid, evts in by_user.items()}


def extract_feature_row(
    event: LoginEvent,
    window_events: List[LoginEvent],
    histories: Dict[str, UserHistory],
) -> Optional[List[float]]:
    history = histories.get(event.user_id)
    if history is None:
        return None

    prior = history.events_before(event.ts)
    if not prior:
        return None

    prev = prior[-1]

    return [
        geo_velocity_kmh(prev, event),
        device_novelty(history, event),
        hour_deviation(history, event),
        device_novelty(history, event) * hour_deviation(history, event),
        novel_device_and_unusual_hour(history, event),
        burst_rate(event, window_events),
    ]


def build_feature_matrix(
    events: List[LoginEvent],
    histories: Dict[str, UserHistory],
) -> Tuple[np.ndarray, List[str]]:
    train_cutoff = START_DATE + timedelta(days=TRAIN_DAYS)

    sorted_events = sorted(events, key=lambda e: e.ts)
    ts_list = [e.ts for e in sorted_events]

    X_train, eids = [], []
    for event in sorted_events:
        if event.ts >= train_cutoff:
            break
        w_start = event.ts - timedelta(minutes=10)
        lo = bisect.bisect_left(ts_list, w_start)
        hi = bisect.bisect_left(ts_list, event.ts)

        row = extract_feature_row(event, sorted_events[lo:hi], histories)
        if row is None:
            continue
        X_train.append(row)
        eids.append(event.event_id)

    return np.array(X_train), eids


def score_event(
    event: LoginEvent,
    all_events: List[LoginEvent],
    histories: Dict[str, UserHistory],
    detector: IsolationForest,
    threshold: float,
    mu: Optional[np.ndarray] = None,
    sigma: Optional[np.ndarray] = None,
) -> dict:
    row = extract_feature_row(event, all_events, histories)

    if row is None:
        decision = {
            "event_id":      event.id,
            "user_id":       event.user_id,
            "score":         None,
            "alert":         False,
            "features":      None,
            "model_version": MODEL_VERSION,
        }
        logger.info(json.dumps(decision))
        return decision

    arr = np.array(row, dtype=float)
    if mu is not None and sigma is not None:
        arr = (arr - mu) / sigma
    raw_score = float(detector.score_samples([arr])[0])
    # Matches the alerting rule in evals/run_eval.py: the IsolationForest
    # score alone, OR'd with the novel_device_and_unusual_hour conjunction.
    is_alert  = bool(raw_score < threshold or row[4] >= 1.0)

    decision = {
        "event_id": event.id,
        "user_id":  event.user_id,
        "score":    round(raw_score, 4),
        "alert":    is_alert,
        "features": {
            "geo_velocity_kmh": round(row[0], 2),
            "device_novelty":   round(row[1], 2),
            "hour_deviation":   round(row[2], 2),
            "device_hour_interaction": round(row[3], 2),
            "novel_device_and_unusual_hour": round(row[4], 2),
            "burst_rate":       round(row[5], 2),
        },
        "model_version": MODEL_VERSION,
    }
    logger.info(json.dumps(decision))
    return decision


def train_and_save() -> None:
    print("Loading events...")
    events = load_events()
    print(f"  {len(events):,} events loaded")

    print("Building user histories...")
    histories = build_histories(events)

    print("Extracting training feature matrix...")
    X_train, _ = build_feature_matrix(events, histories)
    print(f"  {len(X_train):,} training rows")

    print(f"Fitting IsolationForest (n_estimators={N_ESTIMATORS}, contamination={CONTAMINATION})...")

    # Standardize features (fit on training-window clean events only)
    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0)
    sigma[sigma == 0.0] = 1.0

    X_train_std = (X_train - mu) / sigma

    detector = IsolationForest(
        n_estimators  = N_ESTIMATORS,
        contamination = CONTAMINATION,
        random_state  = RANDOM_STATE,
    )
    detector.fit(X_train_std)

    train_scores = detector.score_samples(X_train_std)
    threshold    = float(np.percentile(train_scores, CONTAMINATION * 100))
    print(f"  Alert threshold: {threshold:.4f}")

    geo_vels = X_train[:, 0]
    z_mu     = float(np.mean(geo_vels))
    z_std    = float(np.std(geo_vels))
    print(f"  Geo-velocity baseline: mu={z_mu:.1f} km/h  std={z_std:.1f} km/h")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    payload = {
        "detector":  detector,
        "threshold": threshold,
        "z_mu":      z_mu,
        "z_std":     z_std,
        "mu":        mu,
        "sigma":     sigma,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)
    print(f"Model saved -> {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save()
