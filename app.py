"""Interactive demo: upload a video, watch how the condition-aware pipeline detects and tracks UAVs.

Run:   streamlit run app.py        (or double-click start.bat on Windows)

Uses the project's own `uavtrack` package: features -> policy (FAST 640 / ROBUST 960) -> YOLO -> tracker.
There is no ground truth here, so the app shows boxes, track IDs, the mode timeline and timing only.
It does NOT compute recall / IDF1 / MOTA (those need annotated clips, see the paper).
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from uavtrack.difficulty import DifficultyPolicy
from uavtrack.pipeline import YoloDetector, draw, run_video, summarize_timing
from uavtrack.policy import MODES, make_policy
from uavtrack.tracker import TrackerConfig
from uavtrack.video_io import transcode_h264

import subprocess, imageio_ffmpeg

def to_browser_mp4(src_path):
    src = str(src_path)
    dst = src.replace(".mp4", "_h264.mp4")
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", src, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
    )
    return dst

ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"
RUNS_DIR = ROOT / "runs"
FROZEN = dict(high_conf=0.4, max_age=60, emit_coasting=False)   # frozen on clip 7, as in the paper
VIDEO_TYPES = ["mp4", "mov", "avi", "mkv", "m4v", "webm"]

st.set_page_config(page_title="Condition-aware UAV tracking", layout="wide")
st.title("Condition-aware adaptive perception for air-to-air UAV tracking")
st.caption("Upload a video, choose policies, and compare detections. No ground truth: this is a qualitative demo. "
           "The paper's numbers come from annotated Purdue clips.")


# ----------------------------------------------------------------------------- sidebar
def _pt_files():
    return sorted(p.name for p in WEIGHTS_DIR.glob("*.pt")) if WEIGHTS_DIR.is_dir() else []


def _joblib_files():
    return sorted(p.name for p in WEIGHTS_DIR.glob("*.joblib")) if WEIGHTS_DIR.is_dir() else []


with st.sidebar:
    st.header("Detector")
    pts = _pt_files()
    weights_name = st.selectbox("Weights (.pt in ./weights)", pts or ["(none found)"])
    device_choice = st.selectbox("Device", ["auto (GPU if available)", "cpu"],
                                 help="Choose cpu to see how FAST and ROBUST timings differ without a GPU.")
    conf = st.slider("Confidence floor", 0.01, 0.9, 0.10, 0.01,
                     help="Paper setting: 0.10. The tracker's own gate (high_conf=0.4) is applied on top.")
    st.header("Policies to run")
    run_fast = st.checkbox("Always FAST (640 px)", True)
    run_robust = st.checkbox("Always ROBUST (960 px)", True)
    run_rule = st.checkbox("Rule policy", True)
    jl = _joblib_files()
    diff_models = st.multiselect("Difficulty policy (learned model files)", jl,
                                 default=[j for j in jl if "train7_8" in j] or jl[:1])
    st.header("Tracker (frozen in the paper)")
    high_conf = st.slider("high_conf", 0.1, 0.9, FROZEN["high_conf"], 0.05)
    max_age = st.slider("max_age (frames)", 5, 120, FROZEN["max_age"], 5)
    show_raw = st.checkbox("Also draw raw detections (thin blue boxes, before the tracker gate)", True,
                           help="The tracker only starts tracks from detections >= high_conf, so most boxes on hard "
                                "clips never become tracks. Raw boxes show what the detector actually found.")
    max_frames = st.number_input("Max frames (0 = whole video)", 0, 100000, 150, 10,
                                 help="Keep small on CPU: ROBUST is slow there.")

# ----------------------------------------------------------------------------- input
col_a, col_b = st.columns(2)
with col_a:
    up = st.file_uploader("Upload a video", type=VIDEO_TYPES)
with col_b:
    path_in = st.text_input("...or a video path on this computer", "",
                            help="Useful for large files, e.g. D:\\clips\\clip7.mp4")


def _resolve_video():
    if up is not None:
        RUNS_DIR.mkdir(exist_ok=True)
        dst = RUNS_DIR / f"upload_{int(time.time())}_{Path(up.name).name}"
        dst.write_bytes(up.getbuffer())
        return dst
    if path_in.strip():
        p = Path(path_in.strip().strip('"'))
        return p if p.is_file() else None
    return None


@st.cache_resource(show_spinner="Loading detector...")
def _detector(weights: str, device: str, conf_floor: float):
    import torch
    dev = "cpu" if device == "cpu" else (0 if torch.cuda.is_available() else "cpu")
    return YoloDetector(Path(weights), conf=conf_floor, device=dev)


class _Tap:
    """Wraps the detector so the app can also draw raw detections (before the tracker's confidence gate)."""

    def __init__(self, det):
        self.det, self.last = det, np.zeros((0, 5))

    def sync(self):
        self.det.sync()

    def detect(self, frame, imgsz):
        self.last = self.det.detect(frame, imgsz)
        return self.last


def _make_policies():
    out = []
    if run_fast:
        out.append(("always_fast", make_policy("fast")))
    if run_robust:
        out.append(("always_robust", make_policy("robust")))
    if run_rule:
        out.append(("rule", make_policy("rule")))
    for name in diff_models:
        out.append((f"difficulty ({name.replace('.joblib', '')})", DifficultyPolicy.load(WEIGHTS_DIR / name)[0]))
    return out


# ----------------------------------------------------------------------------- run
go = st.button("Run", type="primary")
if go:
    video = _resolve_video()
    if video is None:
        st.error("Upload a video or give a valid path first.")
        st.stop()
    if not pts:
        st.error("No .pt weights found in ./weights. Copy your trained detector there.")
        st.stop()
    policies = _make_policies()
    if not policies:
        st.error("Select at least one policy.")
        st.stop()

    cap = cv2.VideoCapture(str(video))
    ok, first = cap.read()
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not ok:
        st.error("OpenCV could not read this video (try mp4/H.264).")
        st.stop()
    n_run = n_total if max_frames == 0 else min(n_total, int(max_frames))

    base_det = _detector(str(WEIGHTS_DIR / weights_name), device_choice, conf)
    base_det.warmup(first)
    det = _Tap(base_det)          # remove the cold-start effect so run order does not bias timings
    run_dir = RUNS_DIR / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = TrackerConfig(high_conf=high_conf, max_age=int(max_age), emit_coasting=False)

    results = {}
    overall = st.progress(0.0, text="Starting...")
    for i, (name, policy) in enumerate(policies):
        raw = run_dir / f"{name.split()[0]}_{i}_raw.mp4"
        final = run_dir / f"{name.split()[0]}_{i}.mp4"
        scores = []

        def _draw(frame, tracks, mode, frame_no, _p=policy, _n=name, _i=i):
            scores.append(getattr(_p, "last_score", None))
            overall.progress(min(1.0, (_i + len(scores) / max(1, n_run)) / len(policies)),
                             text=f"{_n}: frame {len(scores)}/{n_run}")
            out = frame.copy()
            if show_raw:
                for x1, y1, x2, y2, c in det.last:
                    cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 120, 0), 1)
                    cv2.putText(out, f"{c:.2f}", (int(x1), max(12, int(y1) - 3)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (255, 120, 0), 1)
            return draw(out, tracks, mode, frame_no, f"{_n} | no ground truth")

        frames, tracks = run_video(video, det, policy, start=1, end=n_run, tracker_cfg=cfg,
                                   video_out=raw, log_every=0, draw_fn=_draw)
        transcode_h264(raw, final)
        raw.unlink(missing_ok=True)
        s = summarize_timing(frames)
        fdf, tdf = pd.DataFrame(frames), pd.DataFrame(tracks)
        if scores and any(v is not None for v in scores):
            fdf["difficulty_score"] = scores
        results[name] = dict(summary=s, frames=fdf, tracks=tdf, video=final,
                             det_per_frame=round(float(fdf["n_det"].mean()), 3),
                             ids=int(tdf["track_id"].nunique()) if len(tdf) else 0,
                             with_track=int(tdf["frame"].nunique()) if len(tdf) else 0)
    overall.progress(1.0, text="Done")
    st.session_state["results"] = results
    st.session_state["device_note"] = "CPU" if (device_choice == "cpu") else "auto"

# ----------------------------------------------------------------------------- results
res = st.session_state.get("results")
if res:
    st.subheader("Summary")
    rows = []
    for name, r in res.items():
        s = r["summary"]
        rows.append({"policy": name, "frames": s["frames"], "ROBUST share": s["robust_fraction"],
                     "detector ms/frame": s["mean_detect_ms"], "total ms/frame": s["mean_total_ms_incl_decode"],
                     "end-to-end FPS": s["end_to_end_fps"], "distinct track IDs": r["ids"],
                     "frames with a track": r["with_track"], "raw detections/frame": r["det_per_frame"]})
    table = pd.DataFrame(rows)
    st.dataframe(table, hide_index=True)
    st.download_button("Download summary CSV", table.to_csv(index=False), "summary.csv", "text/csv")
    st.caption("Timings depend on the machine and device. Run on CPU and GPU to see how the FAST/ROBUST gap changes. "
               "'Distinct track IDs' is not identity accuracy: there is no ground truth here.")

    st.subheader("Annotated videos")
    names = list(res.keys())
    cols = st.columns(min(3, len(names)))
    for k, name in enumerate(names):
        with cols[k % len(cols)]:
            st.markdown(f"**{name}**")
            st.video(to_browser_mp4(res[name]["video"]))
            with open(res[name]["video"], "rb") as fh:
                st.download_button("Download video", fh.read(), Path(res[name]["video"]).name, "video/mp4",
                                   key=f"dl_{k}")

    st.subheader("Mode timeline (0 = FAST, 1 = ROBUST)")
    tl = pd.DataFrame({n: (r["frames"]["mode"] == "ROBUST").astype(int).values for n, r in res.items()
                       if len(r["frames"])})
    if len(tl):
        st.line_chart(tl)
    for name, r in res.items():
        if "difficulty_score" in r["frames"].columns:
            st.markdown(f"**Difficulty score, {name}**")
            st.line_chart(r["frames"]["difficulty_score"])
    with st.expander("Per-frame data"):
        pick = st.selectbox("Policy", names)
        st.dataframe(res[pick]["frames"])
        st.download_button("Download per-frame CSV", res[pick]["frames"].to_csv(index=False), "frames.csv", "text/csv")
        if len(res[pick]["tracks"]):
            st.download_button("Download tracks CSV", res[pick]["tracks"].to_csv(index=False), "tracks.csv", "text/csv")

# ----------------------------------------------------------------------------- latency probe
with st.expander("Detector latency probe: 640 px vs 960 px on this machine"):
    st.write("Runs the detector alone on one frame size at a time, interleaved, so run order does not bias the result.")
    probe_video = st.text_input("Video for the probe (path)", "")
    n_probe = st.number_input("Frames per size", 5, 200, 30, 5)
    if st.button("Measure latency") and probe_video.strip() and pts:
        det = _detector(str(WEIGHTS_DIR / weights_name), device_choice, conf)
        cap = cv2.VideoCapture(probe_video.strip().strip('"'))
        fr = []
        while len(fr) < int(n_probe):
            ok, f = cap.read()
            if not ok:
                break
            fr.append(f)
        cap.release()
        if not fr:
            st.error("Could not read frames.")
        else:
            det.warmup(fr[0])
            ms = {s: [] for s in MODES.values()}
            for order in (list(MODES.values()), list(MODES.values())[::-1]):
                for s in order:
                    for f in fr:
                        det.sync()
                        t = time.perf_counter()
                        det.detect(f, s)
                        det.sync()
                        ms[s].append((time.perf_counter() - t) * 1e3)
            out = pd.DataFrame({"imgsz": list(ms), "mean ms": [np.mean(v) for v in ms.values()],
                                "median ms": [np.median(v) for v in ms.values()], "n": [len(v) for v in ms.values()]})
            st.dataframe(out, hide_index=True)
            a, b = np.mean(ms[640]), np.mean(ms[960])
            st.write(f"ROBUST / FAST latency ratio: **{b / a:.2f}** (pixel-count ratio is 2.25).")
