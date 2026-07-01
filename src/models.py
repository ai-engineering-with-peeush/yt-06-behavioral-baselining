"""
models.py — core data structures for login events and per-user history.

`UserHistory` indexes a user's past logins and derives the per-user
statistics (median login hour, circular dispersion, known devices) that
the feature extractors in features.py normalize against.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Set
import math


@dataclass
class LoginEvent:
    event_id: str
    user_id: str
    ts: datetime
    device_id: str
    country: str
    lat: float
    lon: float
    success: bool

    @property
    def id(self) -> str:
        """Alias used by score_event for cleaner log output."""
        return self.event_id


class UserHistory:
    """Per-user event index for feature extraction.

    All lookups are *strictly before* the given timestamp to prevent leaking
    future data into features.
    """

    def __init__(self, events: List[LoginEvent]) -> None:
        self._events: List[LoginEvent] = sorted(events, key=lambda e: e.ts)

    def events_before(self, ts: datetime) -> List[LoginEvent]:
        """All events strictly before ts."""
        return [e for e in self._events if e.ts < ts]

    def devices_before(self, ts: datetime) -> Set[str]:
        """Set of device IDs seen strictly before ts."""
        return {e.device_id for e in self.events_before(ts)}

    def median_login_hour(self, ts: datetime) -> float:
        """Median hour-of-day (0–23) across logins before ts.

        Returns 12.0 (noon) when there is no prior history — a neutral default.
        """
        prior = self.events_before(ts)
        if not prior:
            return 12.0
        hours = sorted(e.ts.hour + e.ts.minute / 60.0 for e in prior)
        mid = len(hours) // 2
        if len(hours) % 2 == 0:
            return (hours[mid - 1] + hours[mid]) / 2.0
        return hours[mid]

    def login_hour_std(self, ts: datetime) -> float:
        """Robust circular dispersion (MAD scaled to std) in hours.

        Computes angular distances to the circular mean, takes the median
        absolute deviation (MAD), and scales by 1.4826 to estimate standard
        deviation. Returns 6.0 when history is empty or unstable.
        """
        prior = self.events_before(ts)
        if not prior:
            return 6.0

        thetas = [((e.ts.hour + e.ts.minute / 60.0) * (2 * math.pi / 24.0)) for e in prior]
        sin_sum = sum(math.sin(t) for t in thetas)
        cos_sum = sum(math.cos(t) for t in thetas)
        n = len(thetas)
        if n == 0:
            return 6.0

        # circular mean angle
        mean_angle = math.atan2(sin_sum, cos_sum)

        def ang_dist(a: float, b: float) -> float:
            d = abs(a - b) % (2 * math.pi)
            return min(d, 2 * math.pi - d)

        dists = [ang_dist(mean_angle, t) for t in thetas]
        # convert radians -> hours
        dists_hours = [d * 24.0 / (2 * math.pi) for d in dists]

        # median absolute deviation
        dists_hours_sorted = sorted(dists_hours)
        m = dists_hours_sorted[len(dists_hours_sorted) // 2]

        # scale MAD to approximate std for normal-like distributions
        mad_to_std = 1.4826
        est_std = m * mad_to_std
        if est_std <= 1e-6:
            return 6.0
        return est_std
