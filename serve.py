"""
serve.py — FastAPI scoring service for real-time login event scoring.

Loads the IsolationForest trained by evals/run_eval.py and reproduces its
feature extraction and alerting rule exactly, so live scores match the
offline evaluation.
"""

from __future__ import annotations

import json
import logging
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.models import LoginEvent, UserHistory
from src.features import (
    device_novelty,
    hour_deviation,
    burst_rate,
    novel_device_and_unusual_hour,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = DATA_DIR / "model.pkl"
MODEL_VERSION = "v1"
TS_FMT = "%Y-%m-%dT%H:%M:%S"

if not MODEL_PATH.exists():
    raise RuntimeError("model.pkl not found — run 'make eval' first.")

with open(MODEL_PATH, "rb") as _f:
    _payload = pickle.load(_f)

_detector = _payload["detector"]
_threshold = _payload["threshold"]
_mu = _payload.get("mu")
_sigma = _payload.get("sigma")

if _mu is not None:
    _mu = np.array(_mu)
if _sigma is not None:
    _sigma = np.array(_sigma)


class LoginRequest(BaseModel):
    event_id: str
    user_id: str
    ts: str
    device_id: str
    country: str
    lat: float
    lon: float
    success: bool = True


app = FastAPI(title="Login Anomaly Detector", version=MODEL_VERSION)


@app.post("/score")
def score(req: LoginRequest) -> dict:
    try:
        ts = datetime.strptime(req.ts, TS_FMT)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"ts must be {TS_FMT}")

    event = LoginEvent(
        event_id=req.event_id,
        user_id=req.user_id,
        ts=ts,
        device_id=req.device_id,
        country=req.country,
        lat=req.lat,
        lon=req.lon,
        success=req.success,
    )

    # NOTE: this endpoint has no persisted cross-request user history, so
    # every call scores against an empty history and device_novelty /
    # hour_deviation fall back to their neutral defaults. Wiring in a
    # persistent per-user store is required before this can score real
    # traffic meaningfully.
    history = UserHistory([event])
    device_nov = device_novelty(history, event)
    hour_dev = hour_deviation(history, event)
    conj = novel_device_and_unusual_hour(history, event)
    features = [
        0.0,
        device_nov,
        hour_dev,
        device_nov * hour_dev,
        conj,
        burst_rate(event, [event]),
    ]

    arr = np.array(features, dtype=float)
    if _mu is not None and _sigma is not None:
        arr = (arr - _mu) / _sigma
    raw_score = float(_detector.score_samples([arr])[0])
    # Matches the alerting rule in evals/run_eval.py: the IsolationForest
    # score alone, OR'd with the novel_device_and_unusual_hour conjunction.
    is_alert = bool(raw_score < _threshold or conj >= 1.0)

    decision = {
        "event_id": event.event_id,
        "user_id": event.user_id,
        "score": round(raw_score, 4),
        "alert": is_alert,
        "features": {
            "geo_velocity_kmh": round(features[0], 2),
            "device_novelty": round(features[1], 2),
            "hour_deviation": round(features[2], 2),
            "device_hour_interaction": round(features[3], 2),
            "novel_device_and_unusual_hour": round(features[4], 2),
            "burst_rate": round(features[5], 2),
        },
        "model_version": MODEL_VERSION,
    }
    logger.info(json.dumps(decision))
    return decision


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": MODEL_VERSION}
