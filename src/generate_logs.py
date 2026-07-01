"""
generate_logs.py — generates synthetic login events for a fixed set of
users, then injects three labeled attack patterns (impossible_travel,
new_device_odd_hour, credential_stuffing) into the test window for
evaluating the anomaly detector.
"""

from __future__ import annotations

import csv
import math
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

random.seed(42)

START_DATE = datetime(2026, 5, 1, 0, 0, 0)
DAYS = 30
EVENTS_PER_USER_PER_DAY_MU = 22
DATA_DIR = Path(__file__).parent.parent / "data"

CITIES: List[Tuple[str, float, float]] = [
    ("Germany",      52.52,  13.40),
    ("USA",          37.77, -122.41),
    ("India",        28.61,   77.20),
    ("UK",           51.51,   -0.12),
    ("Japan",        35.68,  139.69),
    ("Australia",   -33.87,  151.20),
    ("France",       48.85,    2.35),
    ("Brazil",      -23.55,  -46.63),
    ("Canada",       43.65,  -79.38),
    ("Netherlands",  52.37,    4.89),
]

DISTANT_CITIES: List[Tuple[str, float, float]] = [
    ("Singapore",   1.35,  103.82),
    ("South Africa", -26.20,  28.04),
    ("Argentina",  -34.60,  -58.38),
    ("South Korea", 37.57,  126.97),
    ("UAE",         25.20,   55.27),
]


def _build_users() -> Dict[str, dict]:
    users: Dict[str, dict] = {}
    uid = 1
    for city, lat, lon in CITIES:
        for slot in range(4):
            user_id = f"user_{uid:03d}"
            irregular = (slot == 3)
            work_start = random.choice([7, 8, 9]) if not irregular else 0
            work_end   = work_start + random.choice([8, 9]) if not irregular else 23
            n_devices  = random.choice([1, 2])
            devices    = [f"dev-{user_id}-{chr(97+i)}" for i in range(n_devices)]
            users[user_id] = {
                "country":     city,
                "lat":         lat,
                "lon":         lon,
                "work_start":  work_start,
                "work_end":    work_end,
                "devices":     devices,
                "irregular":   irregular,
            }
            uid += 1
    return users


USERS = _build_users()


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi   = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _random_ts_in_window(date: datetime, start_h: int, end_h: int) -> datetime:
    hour = random.uniform(start_h, end_h - 0.01)
    return date + timedelta(hours=hour)


def _jitter_location(lat: float, lon: float, sigma_km: float = 5.0) -> Tuple[float, float]:
    deg_per_km = 1.0 / 111.0
    return (
        lat + random.gauss(0, sigma_km * deg_per_km),
        lon + random.gauss(0, sigma_km * deg_per_km),
    )


def generate_normal_events() -> List[dict]:
    events: List[dict] = []
    for user_id, profile in USERS.items():
        for day_offset in range(DAYS):
            date = START_DATE + timedelta(days=day_offset)
            n = max(1, int(random.gauss(EVENTS_PER_USER_PER_DAY_MU, 4)))
            for _ in range(n):
                ts = _random_ts_in_window(date, profile["work_start"], profile["work_end"])
                lat, lon = _jitter_location(profile["lat"], profile["lon"])
                events.append({
                    "event_id":  str(uuid.uuid4()),
                    "user_id":   user_id,
                    "ts":        ts,
                    "device_id": random.choice(profile["devices"]),
                    "country":   profile["country"],
                    "lat":       round(lat, 5),
                    "lon":       round(lon, 5),
                    "success":   True,
                    "attack":    "",
                })
    return events


def inject_impossible_travel(events: List[dict], n: int = 12) -> List[dict]:
    injected: List[dict] = []
    users = list(USERS.keys())
    random.shuffle(users)
    targets = users[:n]

    test_start = START_DATE + timedelta(days=20)
    test_events = [e for e in events if e["ts"] >= test_start]

    for user_id in targets:
        user_evts = [e for e in test_events if e["user_id"] == user_id and e["success"]]
        if not user_evts:
            continue
        anchor = random.choice(user_evts)
        distant = random.choice(DISTANT_CITIES)
        gap_minutes = random.uniform(30, 50)
        attack_ts = anchor["ts"] + timedelta(minutes=gap_minutes)
        if attack_ts.date() != anchor["ts"].date():
            continue
        injected.append({
            "event_id":  str(uuid.uuid4()),
            "user_id":   user_id,
            "ts":        attack_ts,
            "device_id": anchor["device_id"],
            "country":   distant[0],
            "lat":       round(distant[1] + random.gauss(0, 0.05), 5),
            "lon":       round(distant[2] + random.gauss(0, 0.05), 5),
            "success":   True,
            "attack":    "impossible_travel",
        })
    return injected


def inject_new_device_odd_hour(events: List[dict], n: int = 20) -> List[dict]:
    injected: List[dict] = []
    regular   = [uid for uid, p in USERS.items() if not p["irregular"]]
    irregular = [uid for uid, p in USERS.items() if     p["irregular"]]
    random.shuffle(regular)
    random.shuffle(irregular)

    targets = regular[: n // 2] + irregular[: n // 2]
    test_start = START_DATE + timedelta(days=20)

    for user_id in targets:
        profile = USERS[user_id]
        day_offset = random.randint(20, DAYS - 1)
        date = START_DATE + timedelta(days=day_offset)
        ts = date + timedelta(hours=random.uniform(1.0, 4.0))
        lat, lon = _jitter_location(profile["lat"], profile["lon"], sigma_km=2)
        injected.append({
            "event_id":  str(uuid.uuid4()),
            "user_id":   user_id,
            "ts":        ts,
            "device_id": f"unknown-{uuid.uuid4().hex[:8]}",
            "country":   profile["country"],
            "lat":       round(lat, 5),
            "lon":       round(lon, 5),
            "success":   True,
            "attack":    "new_device_odd_hour",
        })
    return injected


def inject_credential_stuffing(n_attempts: int = 500) -> List[dict]:
    injected: List[dict] = []
    burst_start = START_DATE + timedelta(days=25, hours=3, minutes=14)
    users = list(USERS.keys())

    for i in range(n_attempts):
        user_id = random.choice(users)
        profile = USERS[user_id]
        offset_sec = random.uniform(0, 600)
        ts = burst_start + timedelta(seconds=offset_sec)
        lat = random.uniform(-60, 70)
        lon = random.uniform(-180, 180)
        injected.append({
            "event_id":  str(uuid.uuid4()),
            "user_id":   user_id,
            "ts":        ts,
            "device_id": f"bot-{uuid.uuid4().hex[:8]}",
            "country":   "unknown",
            "lat":       round(lat, 5),
            "lon":       round(lon, 5),
            "success":   random.random() < 0.02,
            "attack":    "credential_stuffing",
        })
    return injected


FIELDNAMES = ["event_id", "user_id", "ts", "device_id", "country", "lat", "lon", "success"]
TS_FMT = "%Y-%m-%dT%H:%M:%S"


def write_logs(all_events: List[dict]) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / "logs.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for e in all_events:
            writer.writerow({
                "event_id":  e["event_id"],
                "user_id":   e["user_id"],
                "ts":        e["ts"].strftime(TS_FMT),
                "device_id": e["device_id"],
                "country":   e["country"],
                "lat":       e["lat"],
                "lon":       e["lon"],
                "success":   e["success"],
            })
    print(f"  logs.csv        -> {len(all_events):,} events")


def write_labels(all_events: List[dict]) -> None:
    path = DATA_DIR / "labels.csv"
    labeled = [e for e in all_events if e["attack"]]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["event_id", "attack_type"])
        writer.writeheader()
        for e in labeled:
            writer.writerow({"event_id": e["event_id"], "attack_type": e["attack"]})
    counts = {}
    for e in labeled:
        counts[e["attack"]] = counts.get(e["attack"], 0) + 1
    print(f"  labels.csv      -> {len(labeled):,} labeled events")
    for attack, count in sorted(counts.items()):
        print(f"    {attack:<30} {count:>4}")


if __name__ == "__main__":
    print("Generating synthetic auth logs...")

    normal   = generate_normal_events()
    travel   = inject_impossible_travel(normal)
    device   = inject_new_device_odd_hour(normal)
    stuffing = inject_credential_stuffing()

    all_events = normal + travel + device + stuffing
    all_events.sort(key=lambda e: e["ts"])

    write_logs(all_events)
    write_labels(all_events)
    print("Done. Run 'make eval' to score.")
