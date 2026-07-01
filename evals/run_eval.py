"""
run_eval.py — evaluates the login anomaly detector against synthetic,
labeled attack data (impossible_travel, new_device_odd_hour,
credential_stuffing) and reports precision/recall/F1 per attack type,
alongside a geo-velocity z-score baseline for comparison.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import pickle
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import LoginEvent, UserHistory
from src.features import (
    geo_velocity_kmh,
    device_novelty,
    hour_deviation,
    burst_rate,
    novel_device_and_unusual_hour,
)

logging.basicConfig(level=logging.WARNING)

DATA_DIR   = Path(__file__).parent.parent / "data"
TS_FMT     = "%Y-%m-%dT%H:%M:%S"
TRAIN_DAYS = 20
START_DATE = datetime(2026, 5, 1)

N_ESTIMATORS  = 200
CONTAMINATION = 0.002
RANDOM_STATE  = 42


def load_events(path: Path) -> List[LoginEvent]:
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


def load_labels(path: Path) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            labels[row["event_id"]] = row["attack_type"]
    return labels


def build_histories(events: List[LoginEvent]) -> Dict[str, UserHistory]:
    by_user: Dict[str, List[LoginEvent]] = defaultdict(list)
    for e in events:
        by_user[e.user_id].append(e)
    return {uid: UserHistory(evts) for uid, evts in by_user.items()}


def extract_row(
    event: LoginEvent,
    all_events: List[LoginEvent],
    histories: Dict[str, UserHistory],
) -> Optional[List[float]]:
    """Build the 6-feature vector for one login event.

    Returns None for events with no prior history for the user (nothing to
    compare against yet), so the IsolationForest only ever trains and scores
    on events that have a real per-user baseline behind them.
    """
    history = histories.get(event.user_id)
    if history is None:
        return None

    prior = history.events_before(event.ts)
    if not prior:
        return None

    prev = prior[-1]

    vel   = geo_velocity_kmh(prev, event)
    nov   = device_novelty(history, event)
    hdev  = hour_deviation(history, event)
    inter = nov * hdev
    conj  = novel_device_and_unusual_hour(history, event)
    burst = burst_rate(event, all_events)

    return [vel, nov, hdev, inter, conj, burst]


def train(
    events: List[LoginEvent],
    labels: Dict[str, str],
    all_events: List[LoginEvent],
    histories: Dict[str, UserHistory],
) -> Tuple[IsolationForest, float, np.ndarray, List[str], np.ndarray, np.ndarray, np.ndarray]:
    train_cutoff = START_DATE + timedelta(days=TRAIN_DAYS)

    X_train, X_all, eids = [], [], []
    for event in events:
        row = extract_row(event, all_events, histories)
        if row is None:
            continue
        X_all.append(row)
        eids.append(event.event_id)
        if event.ts < train_cutoff and event.event_id not in labels:
            X_train.append(row)

    X_train_arr = np.array(X_train)
    X_all_arr   = np.array(X_all)

    # Standardize features based on training-window clean events
    mu = np.mean(X_train_arr, axis=0)
    sigma = np.std(X_train_arr, axis=0)
    sigma[sigma == 0.0] = 1.0

    X_train_std = (X_train_arr - mu) / sigma
    X_all_std = (X_all_arr - mu) / sigma

    detector = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
    )
    detector.fit(X_train_std)

    train_scores = detector.score_samples(X_train_std)
    threshold = float(np.percentile(train_scores, CONTAMINATION * 100))

    # Raw (unstandardized) novel_device_and_unusual_hour column, returned
    # separately so callers can use it as a standalone alert trigger rather
    # than only as one input among several to the IsolationForest score.
    conj_raw = X_all_arr[:, 4]

    return detector, threshold, X_all_std, eids, mu, sigma, conj_raw


def baseline_alerts(
    events: List[LoginEvent],
    labels: Dict[str, str],
    all_events: List[LoginEvent],
    histories: Dict[str, UserHistory],
    z_threshold: float = 3.0,
) -> Dict[str, bool]:
    train_cutoff = START_DATE + timedelta(days=TRAIN_DAYS)

    train_vels = []
    for event in events:
        if event.ts >= train_cutoff or event.event_id in labels:
            continue
        history = histories.get(event.user_id)
        if not history:
            continue
        prior = history.events_before(event.ts)
        if not prior:
            continue
        train_vels.append(geo_velocity_kmh(prior[-1], event))

    if not train_vels:
        return {}
    mu    = float(np.mean(train_vels))
    sigma = float(np.std(train_vels)) or 1.0

    vels, eids = [], []
    for event in events:
        history = histories.get(event.user_id)
        if not history:
            continue
        prior = history.events_before(event.ts)
        if not prior:
            continue
        vels.append(geo_velocity_kmh(prior[-1], event))
        eids.append(event.event_id)

    return {eid: bool(abs(vels[i] - mu) / sigma > z_threshold) for i, eid in enumerate(eids)}


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def print_report(
    all_events: List[LoginEvent],
    labels: Dict[str, str],
    scored_eids: List[str],
    if_alerts: Dict[str, bool],
    baseline: Dict[str, bool],
) -> None:
    attack_types = ["impossible_travel", "new_device_odd_hour", "credential_stuffing"]

    print("\n" + "=" * 65)
    print("  Login Anomaly Detector — Evaluation Report")
    print("=" * 65)

    print(f"\n{'Attack':<30} {'P':>6} {'R':>6} {'F1':>6}   {'TP':>4} {'FP':>4} {'FN':>4}")
    print("-" * 65)

    total_alerts = sum(1 for a in if_alerts.values() if a)

    for attack in attack_types:
        attack_eids = {eid for eid, t in labels.items() if t == attack}
        tp = sum(1 for eid in scored_eids if eid in attack_eids and if_alerts.get(eid, False))
        fp = sum(1 for eid in scored_eids if eid not in labels and if_alerts.get(eid, False))
        fn = sum(1 for eid in attack_eids if not if_alerts.get(eid, False))
        p, r, f = _prf(tp, fp, fn)
        print(f"  {attack:<28} {p:>5.0%} {r:>6.0%} {f:>6.0%}   {tp:>4} {fp:>4} {fn:>4}")

    total_scored = len(scored_eids)
    total_labeled = len(labels)
    total_normal = total_scored - total_labeled
    total_fp = sum(1 for eid in scored_eids if eid not in labels and if_alerts.get(eid, False))
    total_tp = total_alerts - total_fp
    alerts_per_10k = total_alerts / total_scored * 10_000
    attack_prevalence_per_10k = total_labeled / total_scored * 10_000
    fp_rate_of_normal = total_fp / total_normal * 10_000 if total_normal else 0.0
    overall_precision = total_tp / total_alerts if total_alerts else 0.0
    print(f"\n  Total events scored : {total_scored:,}")
    print(f"  Total alerts        : {total_alerts:,}  (tp={total_tp:,}  fp={total_fp:,})")
    print(f"  Overall precision   : {overall_precision:.0%}")
    print(f"  Alerts per 10k      : {alerts_per_10k:.1f}")
    print(f"  Attack prevalence   : {attack_prevalence_per_10k:.1f}/10k"
          f"  (contamination={CONTAMINATION} assumes only {CONTAMINATION*10_000:.0f}/10k --"
          f" real prevalence here is ~{attack_prevalence_per_10k/max(CONTAMINATION*10_000,1e-9):.0f}x that)")
    print(f"  False-alert rate    : {fp_rate_of_normal:.1f}/10k normal events"
          f"  ({total_fp:,} noisy alerts out of {total_normal:,} legitimate logins)")

    print(f"\n{'Baseline (geo_velocity z-score)':<30} {'P':>6} {'R':>6} {'F1':>6}")
    print("-" * 50)
    for attack in attack_types:
        attack_eids = {eid for eid, t in labels.items() if t == attack}
        b_scored = set(baseline.keys())
        tp = sum(1 for eid in b_scored if eid in attack_eids and baseline.get(eid, False))
        fp = sum(1 for eid in b_scored if eid not in labels and baseline.get(eid, False))
        fn = sum(1 for eid in attack_eids if not baseline.get(eid, False))
        p, r, f = _prf(tp, fp, fn)
        print(f"  {attack:<28} {p:>5.0%} {r:>6.0%} {f:>6.0%}")

    print("\n" + "=" * 65)
    print("  Alerts fire on IsolationForest score < threshold, OR on the")
    print("  novel_device_and_unusual_hour conjunction feature alone.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    print("Loading data...")
    events = load_events(DATA_DIR / "logs.csv")
    labels = load_labels(DATA_DIR / "labels.csv")
    print(f"  {len(events):,} events  |  {len(labels):,} labeled attacks")

    print("Building user histories...")
    histories = build_histories(events)

    print("Training IsolationForest on normal events (days 0-19)...")
    detector, threshold, X_all_std, scored_eids, mu, sigma, conj_raw = train(events, labels, events, histories)
    print(f"  Alert threshold: {threshold:.4f}")

    print("Scoring all events...")
    raw_scores = detector.score_samples(X_all_std)
    # An event alerts if either the IsolationForest flags it, or the
    # novel_device_and_unusual_hour conjunction fires on its own. That
    # feature is near-zero for legitimate traffic by construction, so it's
    # treated as an independent, high-confidence trigger rather than just
    # one more input blended into the shared anomaly score.
    if_alerts = {
        eid: bool(raw_scores[i] < threshold or conj_raw[i] >= 1.0)
        for i, eid in enumerate(scored_eids)
    }

    print("Running z-score baseline...")
    baseline = baseline_alerts(events, labels, events, histories)

    print_report(events, labels, scored_eids, if_alerts, baseline)

    model_payload = {"detector": detector, "threshold": threshold, "mu": mu, "sigma": sigma}
    with open(DATA_DIR / "model.pkl", "wb") as f:
        pickle.dump(model_payload, f)
    print(f"Model saved -> {DATA_DIR / 'model.pkl'}")
