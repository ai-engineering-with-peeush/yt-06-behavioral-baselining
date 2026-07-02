"""
features.py — behavioral feature extraction for login anomaly detection.

Each function derives one signal from a login event and the requesting
user's history: travel speed between consecutive logins, device novelty,
per-user hour-of-day deviation, and account-wide login burst rate.

STARTER STATE (A2 pre-recording): everything below is already fully typed —
including novel_device_and_unusual_hour, carried over from A1. The only
line that differs from src/features.py is hour_deviation's max_std clamp,
still at 4.0 here (tightened to 3.5 live on camera). Diff this file against
../src/features.py to see exactly that one change.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import List

from src.models import LoginEvent, UserHistory


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points (km)."""
    R = 6_371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


def burst_rate(
    event: LoginEvent,
    all_events: List[LoginEvent],
    window_minutes: int = 10,
) -> float:
    """Number of distinct user accounts with login activity in the last window_minutes."""
    window_start = event.ts - timedelta(minutes=window_minutes)
    active_users = {e.user_id for e in all_events if window_start <= e.ts < event.ts}
    return float(len(active_users))


def geo_velocity_kmh(prev_login: LoginEvent, curr_login: LoginEvent) -> float:
    """Speed (km/h) required to travel between two consecutive login locations."""
    distance_km = haversine(
        prev_login.lat, prev_login.lon, curr_login.lat, curr_login.lon
    )
    hours = (curr_login.ts - prev_login.ts).total_seconds() / 3600
    return distance_km / max(hours, 1e-6)


def device_novelty(user_history: UserHistory, curr_login: LoginEvent) -> float:
    """1.0 if this device has never been seen for this user, 0.0 otherwise."""
    seen = user_history.devices_before(curr_login.ts)
    return 0.0 if curr_login.device_id in seen else 1.0


def _circular_distance_hours(a: float, b: float) -> float:
    """Circular distance between two hours in [0,24). Returns value in [0,12]."""
    diff = abs(a - b)
    return min(diff, 24.0 - diff)


def hour_deviation(user_history: UserHistory, curr_login: LoginEvent) -> float:
    """Per-user normalized hour deviation (z-score-like).

    Computes the circular distance between `curr_login`'s hour and the user's
    median login hour, normalized by the user's own circular standard
    deviation. `min_std` prevents exploding values for users with very
    consistent schedules; `max_std` caps how much dispersion credit a user
    with a wide, irregular schedule can claim, so their baseline can't fully
    absorb a login at an hour they've genuinely never used.

    Returns a unitless normalized deviation (higher => more unusual for the user).
    """
    median_hour = user_history.median_login_hour(curr_login.ts)
    diff = _circular_distance_hours(
        curr_login.ts.hour + curr_login.ts.minute / 60.0, median_hour
    )

    user_std = user_history.login_hour_std(curr_login.ts)
    min_std = 0.5
    max_std = 4.0
    std = min(max(user_std, min_std), max_std)

    return diff / std


def novel_device_and_unusual_hour(
    user_history: UserHistory, curr_login: LoginEvent, k: float = 2.0
) -> float:
    """Binary feature: 1.0 if device is novel AND hour_deviation > k, else 0.0.

    `k` is the normalized-hour threshold (z-score-like). This engineered
    conjunction makes the contextual pattern explicit to downstream models.
    """
    nov = device_novelty(user_history, curr_login)
    hdev = hour_deviation(user_history, curr_login)
    return 1.0 if (nov == 1.0 and hdev > k) else 0.0
