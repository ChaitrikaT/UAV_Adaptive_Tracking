"""Small ByteTrack-style multi-object tracker (numpy + scipy only).

- constant-velocity Kalman filter per track (state: cx, cy, w, h, vx, vy)
- two-stage association: high-confidence detections first, then low-confidence
  detections for tracks that are still unmatched
- association cost is centre distance gated by box size. IoU alone is brittle
  for tiny air-to-air targets that move several box-widths per frame.
- IDs are handed out only when a track is confirmed, so single-frame false
  positives do not burn IDs.
Predicted IDs come from this tracker; ground-truth IDs are never an input.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_xyxy(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]) + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


@dataclass
class TrackerConfig:
    high_conf: float = 0.30      # detections >= this start / strongly update tracks
    low_conf: float = 0.10       # detections in [low, high) may only extend existing tracks
    min_hits: int = 2            # detections needed before a track gets an ID
    max_age: int = 30            # frames a track may coast without a detection
    gate_scale: float = 3.0      # gate = max(gate_min_px, gate_scale * max(box w/h))
    gate_min_px: float = 40.0
    emit_coasting: bool = False  # also output Kalman-predicted boxes while coasting
    max_emit_coast: int = 5      # ...for at most this many missed frames
    state_mode: str = "primary"  # policy-state summary: "primary" (most established track) or "worst" (hardest track)


class _Track:
    F = np.eye(6)
    F[0, 4] = F[1, 5] = 1.0
    H = np.zeros((4, 6))
    H[0, 0] = H[1, 1] = H[2, 2] = H[3, 3] = 1.0
    Q = np.diag([1.0, 1.0, 0.5, 0.5, 0.5, 0.5])
    R = np.diag([4.0, 4.0, 4.0, 4.0])

    def __init__(self, box, conf):
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        w, h = max(1.0, box[2] - box[0]), max(1.0, box[3] - box[1])
        self.x = np.array([cx, cy, w, h, 0.0, 0.0])
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 100.0, 100.0])
        self.hits = 1
        self.time_since_update = 0
        self.last_conf = float(conf)
        self.track_id: int | None = None  # assigned on confirmation

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.x[2:4] = np.maximum(self.x[2:4], 1.0)
        self.time_since_update += 1

    def update(self, box, conf):
        z = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2,
                      max(1.0, box[2] - box[0]), max(1.0, box[3] - box[1])])
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(6) - K @ self.H) @ self.P
        self.hits += 1
        self.time_since_update = 0
        self.last_conf = float(conf)

    @property
    def xyxy(self):
        cx, cy, w, h = self.x[:4]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


class Tracker:
    def __init__(self, config: TrackerConfig | None = None):
        self.cfg = config or TrackerConfig()
        self.tracks: list[_Track] = []
        self._next_id = 1

    # ---- association -------------------------------------------------
    def _match(self, tracks, dets):
        """Return (matches[(ti, di)], unmatched_track_idx, unmatched_det_idx)."""
        if not tracks or len(dets) == 0:
            return [], list(range(len(tracks))), list(range(len(dets)))
        cost = np.full((len(tracks), len(dets)), 1e6)
        for i, t in enumerate(tracks):
            tb = t.xyxy
            tc = np.array([(tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2])
            for j, d in enumerate(dets):
                dc = np.array([(d[0] + d[2]) / 2, (d[1] + d[3]) / 2])
                size = max(tb[2] - tb[0], tb[3] - tb[1], d[2] - d[0], d[3] - d[1])
                gate = max(self.cfg.gate_min_px, self.cfg.gate_scale * size)
                dist = float(np.linalg.norm(tc - dc))
                if dist <= gate:
                    cost[i, j] = dist / gate - 0.5 * iou_xyxy(tb, d)
        rows, cols = linear_sum_assignment(cost)
        matches = [(int(r), int(c)) for r, c in zip(rows, cols) if cost[r, c] < 1e5]
        mt = {r for r, _ in matches}
        md = {c for _, c in matches}
        return (matches,
                [i for i in range(len(tracks)) if i not in mt],
                [j for j in range(len(dets)) if j not in md])

    # ---- main step ---------------------------------------------------
    def update(self, detections) -> list[dict]:
        """detections: array-like of [x1, y1, x2, y2, conf]. Returns output rows."""
        dets = np.asarray(detections, dtype=float).reshape(-1, 5)
        for t in self.tracks:
            t.predict()
        high = dets[dets[:, 4] >= self.cfg.high_conf]
        low = dets[(dets[:, 4] >= self.cfg.low_conf) & (dets[:, 4] < self.cfg.high_conf)]
        observed: set[int] = set()

        # stage 1: all tracks vs high-confidence detections
        matches, unmatched_tracks, unmatched_high = self._match(self.tracks, high)
        for ti, di in matches:
            self.tracks[ti].update(high[di, :4], high[di, 4])
            observed.add(id(self.tracks[ti]))
        # stage 2: leftover *confirmed* tracks vs low-confidence detections
        leftover = [self.tracks[i] for i in unmatched_tracks if self.tracks[i].track_id is not None]
        m2, _, _ = self._match(leftover, low)
        for ti, di in m2:
            leftover[ti].update(low[di, :4], low[di, 4])
            observed.add(id(leftover[ti]))
        # new tentative tracks from unmatched high-confidence detections
        for j in unmatched_high:
            self.tracks.append(_Track(high[j, :4], high[j, 4]))
            observed.add(id(self.tracks[-1]))

        # confirm + prune
        survivors = []
        for t in self.tracks:
            if t.track_id is None and t.hits >= self.cfg.min_hits:
                t.track_id = self._next_id
                self._next_id += 1
            if t.track_id is None and t.time_since_update > 0:
                continue  # tentative track that missed once: drop
            if t.time_since_update > self.cfg.max_age:
                continue
            survivors.append(t)
        self.tracks = survivors

        out = []
        for t in self.tracks:
            if t.track_id is None:
                continue
            seen = t.time_since_update == 0
            if seen or (self.cfg.emit_coasting and t.time_since_update <= self.cfg.max_emit_coast):
                x1, y1, x2, y2 = t.xyxy
                out.append({"track_id": t.track_id, "x1": float(x1), "y1": float(y1),
                            "x2": float(x2), "y2": float(y2),
                            "conf": t.last_conf, "observed": int(seen)})
        return out

    def primary_state(self) -> dict:
        """Summary of the most established confirmed track, for condition features.

        Uses only information available BEFORE the next frame is processed.
        """
        confirmed = [t for t in self.tracks if t.track_id is not None]
        if not confirmed:
            return {"has_track": 0, "frames_since_seen": 999, "target_size": 0.0,
                    "target_speed": 0.0, "last_conf": 0.0}
        if self.cfg.state_mode == "worst":
            # hardest confirmed track decides: any track being lost / tiny / low-confidence is a reason
            # to search harder. Suited to several targets; on single-target clips spurious coasting
            # tracks can keep this pessimistic, so the default stays "primary".
            return {"has_track": 1,
                    "frames_since_seen": int(max(t.time_since_update for t in confirmed)),
                    "target_size": float(min(np.sqrt(t.x[2] * t.x[3]) for t in confirmed)),
                    "target_speed": float(max(np.hypot(t.x[4], t.x[5]) for t in confirmed)),
                    "last_conf": float(min(t.last_conf for t in confirmed))}
        t = max(confirmed, key=lambda k: (k.hits, -k.time_since_update))
        return {"has_track": 1, "frames_since_seen": int(t.time_since_update),
                "target_size": float(np.sqrt(t.x[2] * t.x[3])),
                "target_speed": float(np.hypot(t.x[4], t.x[5])), "last_conf": t.last_conf}
