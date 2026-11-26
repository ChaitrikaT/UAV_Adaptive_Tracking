"""Streamlit UI for Condition-Aware Adaptive Perception for Air-to-Air UAV Tracking.

Combines polished live tracking, multi-model adaptive selection (8 ML models + Rules),
10 causal condition features, ground-truth evaluation (HOTA/IDF1/MOTA/IDSW),
and online-replay equivalence verification.
"""

import json
import time
from collections import deque
from pathlib import Path

import altair as alt
import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.evaluation.ground_truth import discover_gt_for_video, load_mot_gt
from src.evaluation.metrics import evaluate_against_ground_truth
from src.evaluation.protocol import verify_online_replay_equivalence
from src.selector.selector import (
    LEARNED_SELECTOR_KINDS,
    SELECTOR_LABELS,
    available_selectors,
    selector_model_path,
)
from src.utils.config import load_config

UPLOAD_DIR = Path("data/uploads")
GT_UPLOAD_DIR = Path("data/gt")

MODE_HELP = {
    "ADAPTIVE": "A selector watches 10 causal features every frame and switches between FAST and ROBUST.",
    "FAST": "Small model (YOLOv8n), 640px, skips frames. Quickest, least careful.",
    "ROBUST": "Larger model (YOLOv8s), 1280px, every frame. Slowest, most careful.",
}
MODE_COLORS = {"ADAPTIVE": "#2563eb", "FAST": "#0d9488", "ROBUST": "#ea580c"}
BADGE_COLORS = {"ADAPTIVE": "blue", "FAST": "green", "ROBUST": "orange"}
SELECTOR_HELP = {
    "rules": "Hand-set thresholds on blur, contrast, motion, target size, and confidence (baseline).",
    "tree": "Decision Tree trained to predict when FAST would miss targets ROBUST finds.",
    "forest": "Random Forest ensemble of decision trees.",
    "gbm": "Gradient Boosting classifier capturing non-linear condition interactions.",
    "logreg": "Calibrated Logistic Regression difficulty estimator (from Chaitrika's protocol).",
    "svm": "RBF-kernel Support Vector Machine with probability calibration.",
    "mlp": "Multi-Layer Perceptron neural network on normalized causal features.",
    "xgb": "XGBoost gradient boosted trees.",
    "lgbm": "LightGBM histogram gradient boosting.",
}
RESULT_TABS = [
    ":material/movie: Video",
    ":material/moving: Targets & paths",
    ":material/tune: Conditions",
    ":material/compare_arrows: Compare runs",
]
LIVE_MAX_UPDATES = 12
LIVE_MAX_WIDTH = 960

st.set_page_config(page_title="UAV Tracker", page_icon=":material/flight:", layout="wide")
st.session_state.setdefault("runs", {})


@st.cache_data(ttl=60, show_spinner=False)
def app_config() -> dict:
    return load_config("config.yaml")


@st.cache_resource(show_spinner=False, max_entries=16)
def selector_report(kind: str, mtime: float) -> dict:
    return joblib.load(selector_model_path(kind, app_config())).get("report", {})


def trained_selectors() -> list:
    """Return available selectors with learned models ordered by priority, plus rules."""
    found = available_selectors(app_config())
    preferred = ["forest", "gbm", "xgb", "lgbm", "logreg", "svm", "mlp", "tree", "rules"]
    return [k for k in preferred if k in found]


@st.cache_resource(show_spinner=False)
def engine_status() -> dict:
    return {"ready": False}


def load_engine():
    status = engine_status()
    if status["ready"]:
        return

    screen = st.empty()
    with screen.container():
        st.space("large")
        _, middle, _ = st.columns([1, 2, 1])
        with middle.container(border=True, horizontal_alignment="center"):
            st.space("small")
            st.badge("Air-to-air tracking", icon=":material/flight:", color="primary")
            st.subheader("Getting the tracker ready", anchor=False, text_alignment="center")
            st.caption(
                "Loading detection libraries and warming up FAST and ROBUST models (~10 seconds on cold start).",
                text_alignment="center",
            )
            bar = st.progress(0.0, text="Step 1 of 3 · Loading detection libraries")

            try:
                with st.spinner("Importing torch and YOLO...", show_time=True):
                    from src.pipeline.pipeline import Mode, TrackingPipeline

                bar.progress(0.45, text="Step 2 of 3 · Warming up FAST model")
                with st.spinner("Running a first test frame...", show_time=True):
                    pipeline = TrackingPipeline(mode=Mode.ADAPTIVE, selector_type="rules")
                    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
                    pipeline.fast_detector.detect(blank)

                bar.progress(0.9, text="Step 3 of 3 · Warming up ROBUST model")
                with st.spinner("Running a first test frame...", show_time=True):
                    pipeline.robust_detector.detect(blank)
            except Exception as err:
                bar.empty()
                st.error(
                    f"{err}\n\nCheck that the model files in `weights/` and `config.yaml` are in place.",
                    title="The tracker couldn't start",
                    icon=":material/error:",
                )
                st.stop()

            bar.progress(1.0, text="Ready")
            time.sleep(0.25)

    status["ready"] = True
    screen.empty()


load_engine()


def sidebar_settings():
    with st.sidebar:
        st.markdown("### :material/flight: UAV Tracker")
        st.caption("Condition-aware adaptive perception for air-to-air tracking")

        st.markdown("**Settings**")
        conf = st.slider("Detection confidence", 0.05, 0.90, 0.25, 0.05)
        limit = st.number_input("Only process first N frames", 0, 20000, 0, 50, help="0 = full video.")
        live = st.toggle("Show tracking live", value=True)
        verify_replay = st.toggle("Verify online-replay equivalence", value=True)

        history_slot = st.container()

    return conf, int(limit) or None, live, verify_replay, history_slot


def draw_history(slot):
    runs = st.session_state["runs"]
    if not runs:
        return
    with slot:
        st.divider()
        st.markdown("**Saved runs**")
        for label, saved in runs.items():
            active = label == st.session_state.get("active_run")
            st.caption(
                f"{':material/check_circle:' if active else ':material/circle:'} {label}"
                f" · {saved['summary']['avg_fps']} fps"
            )
        if st.button("Clear saved runs", type="tertiary", icon=":material/delete:"):
            st.session_state["runs"] = {}
            st.rerun()


def setup_panel():
    with st.container(border=True):
        left, right = st.columns([3, 2], gap="large")
        with left:
            video = st.file_uploader(
                "Flight video",
                type=["mp4", "avi", "mov", "mkv"],
                help="Upload a Purdue or custom UAV flight clip.",
            )
            gt_file = st.file_uploader(
                "Optional Ground Truth (MOT .txt or .csv)",
                type=["txt", "csv"],
                help="Enables HOTA, IDF1, MOTA, and IDSW scoring. Auto-detected from data/gt/ if omitted.",
            )
        with right:
            mode = (
                st.segmented_control(
                    "Mode",
                    options=list(MODE_HELP),
                    default="ADAPTIVE",
                    selection_mode="single",
                    format_func=str.title,
                    width="stretch",
                )
                or "ADAPTIVE"
            )
            selector, threshold = None, None
            if mode == "ADAPTIVE":
                options = trained_selectors()
                selector = st.selectbox(
                    "Selector model",
                    options=options,
                    index=0,
                    format_func=lambda k: SELECTOR_LABELS.get(k, k),
                )
                st.caption(SELECTOR_HELP.get(selector, ""))
                if selector in LEARNED_SELECTOR_KINDS:
                    threshold = st.slider(
                        "Switch to ROBUST when P(ROBUST) >=",
                        0.10,
                        0.95,
                        float((app_config().get("selector") or {}).get("threshold", 0.5)),
                        0.05,
                        format="%.2f",
                    )
                if options == ["rules"]:
                    st.caption(
                        ":material/school: No trained ML selectors yet. Run `python -m src.training.train_selector`."
                    )
            else:
                st.caption(MODE_HELP[mode])

            run = st.button(
                "Run tracking",
                type="primary",
                width="stretch",
                icon=":material/play_arrow:",
                disabled=video is None,
            )
            compare = st.button(
                "Compare all modes",
                width="stretch",
                icon=":material/compare_arrows:",
                disabled=video is None,
                help="Runs FAST, ROBUST, and ADAPTIVE across all available selectors.",
            )
            if video is None:
                st.caption("Upload a video to enable the buttons.")
    return video, gt_file, mode, selector, threshold, run, compare


def mode_title(mode: str, selector) -> str:
    return f"Adaptive · {SELECTOR_LABELS.get(selector, selector)}" if mode == "ADAPTIVE" and selector else mode.title()


def run_label(settings) -> str:
    label = f"{settings['video_name']} · {mode_title(settings['mode'], settings.get('selector'))}"
    if settings.get("selector") in LEARNED_SELECTOR_KINDS and settings.get("threshold") not in (None, 0.5):
        label += f" @ {settings['threshold']:.2f}"
    if settings["conf"] != 0.25:
        label += f" · conf {settings['conf']:.2f}"
    if settings["max_frames"]:
        label += f" · first {settings['max_frames']}"
    return label


def save_upload(video) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / video.name
    path.write_bytes(video.getbuffer())
    return path


def resolve_gt(gt_file, video_path: Path):
    if gt_file is not None:
        GT_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        gt_path = GT_UPLOAD_DIR / gt_file.name
        gt_path.write_bytes(gt_file.getbuffer())
        return load_mot_gt(gt_path)
    discovered = discover_gt_for_video(video_path, app_config().get("paths", {}).get("gt_dir", "data/gt"))
    return load_mot_gt(discovered) if discovered else None


def live_view():
    frame_col, stats_col = st.columns([3, 1])
    with frame_col:
        image_slot = st.skeleton(height=360)
    with stats_col:
        mode_slot, fps_slot, tracks_slot = st.empty(), st.empty(), st.empty()

    recent_fps, recent_tracks = deque(maxlen=90), deque(maxlen=90)
    last_draw = [0.0]

    def on_frame(frame, info):
        recent_fps.append(info["fps"])
        recent_tracks.append(info["num_tracks"])
        now = time.perf_counter()
        is_last = 0 < info["total_frames"] <= info["frame"] + 1
        if now - last_draw[0] < 1 / LIVE_MAX_UPDATES and not is_last:
            return
        last_draw[0] = now

        h, w = frame.shape[:2]
        if w > LIVE_MAX_WIDTH:
            frame = cv2.resize(
                frame, (LIVE_MAX_WIDTH, int(h * LIVE_MAX_WIDTH / w)), interpolation=cv2.INTER_AREA
            )
        image_slot.image(
            frame,
            channels="BGR",
            output_format="JPEG",
            width="stretch",
            caption=f"Frame {info['frame'] + 1} of {info['total_frames']}",
        )

        mode = info["mode"]
        mode_slot.metric(
            "Mode now",
            mode.title(),
            border=True,
            icon=":material/bolt:" if mode == "FAST" else ":material/shield:",
        )
        fps_slot.metric("Speed", f"{info['fps']:.0f} fps", border=True, chart_data=list(recent_fps))
        tracks_slot.metric(
            "Targets in view", info["num_tracks"], border=True, chart_data=list(recent_tracks), chart_type="bar"
        )

    return on_frame


def run_pipeline(video, gt_file, settings, live: bool, verify_replay: bool):
    from src.pipeline.pipeline import Mode, TrackingPipeline

    label = run_label(settings)
    title = mode_title(settings["mode"], settings.get("selector"))
    with st.status(f"Tracking {settings['video_name']} · {title}...", expanded=True) as status:
        video_path = save_upload(video)
        gt_df = resolve_gt(gt_file, video_path)
        bar = st.progress(0.0, text="Preparing models...")
        on_frame = live_view() if live else None

        def on_progress(current, total, fps):
            bar.progress(min(1.0, current / max(1, total)), text=f"Frame {current} of {total} · {fps:.0f} fps")

        try:
            pipeline = TrackingPipeline(
                mode=Mode(settings["mode"]),
                custom_conf=settings["conf"],
                selector_type=settings.get("selector"),
                selector_threshold=settings.get("threshold"),
            )
            summary = pipeline.run_video(
                input_path=video_path,
                max_frames=settings["max_frames"],
                progress_callback=on_progress,
                frame_callback=on_frame,
            )
        except Exception as err:
            bar.empty()
            status.update(label="Tracking failed", state="error", expanded=True)
            st.error(str(err), title="The pipeline stopped with an error")
            st.stop()

        gt_scores = evaluate_against_ground_truth(summary["tracks_df"], gt_df) if gt_df is not None else None
        replay_check = None
        if verify_replay and settings["mode"] == "ADAPTIVE" and summary.get("selector"):
            replay_check = verify_online_replay_equivalence(
                summary["records_df"], summary["selector"], app_config(), settings.get("threshold")
            )

        bar.empty()
        status.update(
            label=f"Done: {summary['total_frames']} frames in {summary['total_time_seconds']} s",
            state="complete",
            expanded=False,
        )

    runs = st.session_state["runs"]
    runs.pop(label, None)
    runs[label] = {
        **settings,
        "label": label,
        "input": str(video_path),
        "summary": summary,
        "gt_scores": gt_scores,
        "replay_check": replay_check,
    }
    st.session_state["active_run"] = label


def sparkline(values: pd.Series, points: int = 60) -> list:
    s = pd.Series(values).astype(float).reset_index(drop=True)
    if s.empty:
        return []
    step = max(1, len(s) // points)
    return s.groupby(s.index // step).mean().round(3).tolist()


def run_stats(run) -> dict:
    summary, frames = run["summary"], run["summary"]["records_df"]
    robust = (frames["mode"] == "ROBUST").mean() if not frames.empty else 0.0
    row = {
        "run": run["label"],
        "mode": run["mode"],
        "decided_by": SELECTOR_LABELS.get(summary.get("selector"), "—"),
        "frames": summary["total_frames"],
        "seconds": summary["total_time_seconds"],
        "fps": summary["avg_fps"],
        "detector": summary["detector_call_ratio"],
        "robust": float(robust),
        "switches": summary["mode_switches"],
        "targets": summary.get("total_unique_targets", 0),
    }
    if run.get("gt_scores"):
        gt = run["gt_scores"]
        row.update({"HOTA": gt["HOTA"], "IDF1": gt["IDF1"], "MOTA": gt["MOTA"], "IDSW": gt["IDSW"]})
    return row


def stat_row(run):
    summary, frames = run["summary"], run["summary"]["records_df"]
    stats = run_stats(run)
    peak = int(frames["num_tracks"].max()) if not frames.empty else 0
    has = not frames.empty

    cols = st.columns(4)
    cols[0].metric("Speed", f"{stats['fps']} fps", border=True, height="stretch", chart_data=sparkline(frames["fps"]) if has else None)
    cols[1].metric(
        "Detector used on",
        f"{stats['detector'] * 100:.0f}% of frames",
        border=True,
        height="stretch",
        chart_type="area",
        chart_data=sparkline(frames["detector_invoked"]) if has else None,
    )
    cols[2].metric(
        "Time in robust mode",
        f"{stats['robust'] * 100:.0f}%",
        border=True,
        height="stretch",
        chart_type="area",
        chart_data=sparkline(frames["mode"] == "ROBUST") if has else None,
    )
    cols[3].metric(
        "Most targets at once",
        peak,
        border=True,
        height="stretch",
        chart_type="bar",
        chart_data=sparkline(frames["num_tracks"]) if has else None,
    )

    if run.get("gt_scores"):
        gt = run["gt_scores"]
        gcols = st.columns(4)
        gcols[0].metric("HOTA", f"{gt['HOTA']:.1f}%", border=True)
        gcols[1].metric("IDF1", f"{gt['IDF1']:.1f}%", border=True)
        gcols[2].metric("MOTA", f"{gt['MOTA']:.1f}%", border=True)
        gcols[3].metric("ID Switches", gt["IDSW"], border=True)


def tab_video(run):
    summary = run["summary"]
    left, right = st.columns(2)
    with left:
        st.caption("Original")
        if Path(run["input"]).is_file():
            st.video(run["input"])
    with right:
        st.caption(f"Tracked · {run['mode'].title()} mode")
        if Path(summary["output_video"]).is_file():
            st.video(summary["output_video"])

    with st.container(horizontal=True):
        for key, label, mime in [
            ("output_video", "Tracked video", "video/mp4"),
            ("tracks_csv", "Target positions (CSV)", "text/csv"),
            ("metrics_csv", "Per-frame data (CSV)", "text/csv"),
        ]:
            path = Path(summary.get(key, ""))
            if path.is_file():
                st.download_button(
                    label,
                    data=path.read_bytes if key == "output_video" else path.read_bytes(),
                    file_name=path.name,
                    mime=mime,
                    icon=":material/download:",
                    on_click="ignore",
                    key=f"dl_{key}",
                )


def tab_targets(run):
    tracks_df = run["summary"].get("tracks_df", pd.DataFrame())
    if tracks_df.empty:
        st.info("Try lowering the detection confidence in the sidebar.", title="No objects were tracked", icon=":material/search_off:")
        return

    ids = sorted(tracks_df["track_id"].unique())
    labels = [f"#{i}" for i in ids]
    st.markdown(f"**{len(ids)}** target(s) followed across **{len(tracks_df)}** observations.")

    picker = st.pills if len(ids) <= 15 else st.multiselect
    choice = picker(
        "Show targets",
        labels,
        default=labels[:6],
        key=f"targets_{run['label']}",
        **({"selection_mode": "multi"} if picker is st.pills else {}),
    )
    shown = tracks_df[tracks_df["track_id"].isin([int(c[1:]) for c in choice or []])].copy()
    shown["target"] = "#" + shown["track_id"].astype(str)

    if not shown.empty:
        path = (
            alt.Chart(shown)
            .mark_line(point=alt.OverlayMarkDef(size=18))
            .encode(
                x=alt.X("center_x:Q", title="x (px)"),
                y=alt.Y("center_y:Q", title="y (px)", scale=alt.Scale(reverse=True)),
                color=alt.Color("target:N", title="Target"),
                order="frame:Q",
                tooltip=["target", "frame", "center_x", "center_y", "confidence"],
            )
            .properties(height=380)
            .interactive()
        )
        st.altair_chart(path)

    st.markdown("**Last known position of each target**")
    last = tracks_df.sort_values("frame").groupby("track_id").tail(1)
    px = st.column_config.NumberColumn(format="%.0f px")
    st.dataframe(
        last[["track_id", "frame", "center_x", "center_y", "bbox_width", "bbox_height", "confidence"]],
        hide_index=True,
        column_config={
            "track_id": st.column_config.NumberColumn("Target", format="#%d"),
            "frame": "Last seen (frame)",
            "center_x": px,
            "center_y": px,
            "bbox_width": px,
            "bbox_height": px,
            "confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0, format="%.2f"),
        },
    )


def tab_conditions(run):
    frames_df = run["summary"]["records_df"]
    if frames_df.empty:
        st.info("No per-frame data available.")
        return

    stats = run_stats(run)
    st.markdown("**Which mode was running**")
    st.caption(f"{stats['switches']} switches · ROBUST for {stats['robust'] * 100:.0f}% of frames")
    modes = frames_df.assign(run=lambda d: (d["mode"] != d["mode"].shift()).cumsum())
    blocks = modes.groupby("run").agg(start=("frame", "min"), end=("frame", "max"), mode=("mode", "first"))
    blocks["end"] += 1
    timeline = (
        alt.Chart(blocks.reset_index())
        .mark_bar(height=26)
        .encode(
            x=alt.X("start:Q", title="frame"),
            x2="end:Q",
            color=alt.Color(
                "mode:N",
                title=None,
                scale=alt.Scale(domain=["FAST", "ROBUST"], range=[MODE_COLORS["FAST"], MODE_COLORS["ROBUST"]]),
            ),
            tooltip=["mode", "start", "end"],
        )
        .properties(height=60)
    )
    st.altair_chart(timeline)

    st.markdown("**Causal scene & tracker conditions (scaled 0-1)**")
    names = {
        "blur": "Sharpness",
        "contrast": "Contrast",
        "camera_motion": "Camera motion",
        "frame_change": "Frame change",
        "mean_confidence": "Detection confidence",
        "target_speed": "Target speed",
    }
    cols = [c for c in names if c in frames_df]
    if cols:
        scaled = frames_df[cols] / frames_df[cols].max().clip(lower=1e-9)
        scaled.index = frames_df["frame"]
        st.line_chart(scaled.rename(columns=names), height=280, x_label="frame")

    why_it_switched(run)
    with st.expander("Per-frame table", icon=":material/table_rows:"):
        st.dataframe(frames_df, hide_index=True)


def why_it_switched(run):
    kind = run["summary"].get("selector")
    if kind not in LEARNED_SELECTOR_KINDS:
        return
    frames_df = run["summary"]["records_df"]
    st.markdown(f"**Why it switched · {SELECTOR_LABELS.get(kind, kind)}**")
    left, right = st.columns([3, 2], gap="large")
    with left:
        if "robust_prob" in frames_df:
            thr = run.get("threshold") or 0.5
            st.caption(f"P(ROBUST needed) per frame (threshold = {thr:.2f}).")
            prob = frames_df.set_index("frame")[["robust_prob"]].rename(columns={"robust_prob": "Chance ROBUST is needed"})
            st.area_chart(prob, height=220, x_label="frame", y_label="probability", color=MODE_COLORS["ROBUST"])
    with right:
        path = selector_model_path(kind, app_config())
        if path.is_file():
            report = selector_report(kind, path.stat().st_mtime)
            importance = pd.DataFrame(
                list(report.get("feature_importance", {}).items())[:8], columns=["feature", "importance"]
            )
            if not importance.empty:
                st.caption("Top features from training:")
                st.altair_chart(
                    alt.Chart(importance)
                    .mark_bar(color=MODE_COLORS["ADAPTIVE"])
                    .encode(
                        x=alt.X("importance:Q", title=None),
                        y=alt.Y("feature:N", sort="-x", title=None),
                        tooltip=["feature", alt.Tooltip("importance:Q", format=".3f")],
                    )
                    .properties(height=alt.Step(24))
                )
            cv = report.get("cv") or {}
            if cv:
                auc_str = f" · ROC-AUC {cv['roc_auc']:.3f}" if "roc_auc" in cv else ""
                st.caption(
                    f"CV accuracy {cv['accuracy'] * 100:.1f}% · balanced {cv['balanced_accuracy'] * 100:.1f}%{auc_str}"
                )


def tab_compare(runs):
    if len(runs) >= 2:
        table = pd.DataFrame([run_stats(r) for r in runs.values()])
        st.markdown("**Speed of each run**")
        bars = (
            alt.Chart(table)
            .mark_bar()
            .encode(
                x=alt.X("fps:Q", title="average fps"),
                y=alt.Y("run:N", title=None, sort="-x"),
                color=alt.Color(
                    "mode:N",
                    title="Mode",
                    scale=alt.Scale(domain=list(MODE_COLORS), range=list(MODE_COLORS.values())),
                ),
                tooltip=["run", "fps", "seconds", "detector"],
            )
            .properties(height=alt.Step(34))
        )
        st.altair_chart(bars)
        st.dataframe(
            table,
            hide_index=True,
            column_config={
                "run": st.column_config.TextColumn("Run", width="large"),
                "detector": st.column_config.ProgressColumn("Detector used", min_value=0.0, max_value=1.0, format="percent"),
                "robust": st.column_config.ProgressColumn("Time in robust", min_value=0.0, max_value=1.0, format="percent"),
            },
        )
    else:
        st.info("Run the same video in another mode or press **Compare all modes** to populate side-by-side results.")

    report_file = Path("outputs/selector/report.json")
    if report_file.is_file():
        with st.expander("Trained Selector CV Leaderboard (from outputs/selector/report.json)", expanded=True):
            rep = json.loads(report_file.read_text(encoding="utf-8"))
            lb_rows = [{"Model": "Rules (Baseline)", **rep["rules_baseline"]}]
            for k, v in rep.get("models", {}).items():
                scores = v.get("cv") or v.get("train") or {}
                lb_rows.append({"Model": SELECTOR_LABELS.get(k, k), **scores})
            st.dataframe(pd.DataFrame(lb_rows), hide_index=True)


def show_results(runs, settings):
    labels = list(runs)
    if st.session_state.get("active_run") not in runs:
        st.session_state["active_run"] = labels[-1]

    if len(labels) > 1:
        st.selectbox("Showing run", labels, key="active_run")
    run = runs[st.session_state["active_run"]]

    with st.container(horizontal=True, vertical_alignment="center"):
        st.subheader(run["video_name"], anchor=False, width="content")
        st.badge(run["mode"].title(), color=BADGE_COLORS[run["mode"]])
        if run["summary"].get("selector"):
            st.badge(
                f"Decided by {SELECTOR_LABELS.get(run['summary']['selector'], run['summary']['selector']).lower()}",
                icon=":material/psychology:",
                color="violet",
            )
        if run.get("replay_check"):
            eq = run["replay_check"]
            st.badge(
                f"Replay {'100% equivalent' if eq['equivalent'] else 'mismatch'}",
                icon=":material/verified:",
                color="green" if eq["equivalent"] else "red",
            )
        st.badge(f"{run['summary']['total_frames']} frames", icon=":material/movie:", color="gray")
        st.badge(f"{run['summary']['total_time_seconds']} s", icon=":material/timer:", color="gray")

    stat_row(run)

    tabs = st.tabs(RESULT_TABS, key="results_tab", on_change="rerun")
    for tab, draw, arg in zip(tabs, [tab_video, tab_targets, tab_conditions, tab_compare], [run, run, run, runs]):
        if tab.open:
            with tab:
                draw(arg)


def show_empty_state():
    a, b, c = st.columns(3, border=True)
    a.markdown(":material/upload: **Upload**  \nDrop a drone flight video (and optional MOT ground truth).")
    b.markdown(":material/alt_route: **Choose a mode & model**  \nCompare Rules, Random Forest, GBM, Logistic Regression, SVM, MLP, XGBoost & LightGBM.")
    c.markdown(":material/compare_arrows: **Run and evaluate**  \nInspect FPS, detector savings, replay equivalence, and HOTA/IDF1/MOTA.")


conf, max_frames, live, verify_replay, history_slot = sidebar_settings()

st.badge("Air-to-air tracking", icon=":material/flight:", color="primary")
st.title("Condition-Aware Adaptive UAV Tracker", anchor=False)
st.caption("10 causal features + multi-model adaptive perception + ground-truth MOT evaluation.")

video, gt_file, mode, selector, threshold, run_one, run_all = setup_panel()
settings = {
    "video_name": video.name if video else None,
    "mode": mode,
    "selector": selector,
    "threshold": threshold,
    "conf": conf,
    "max_frames": max_frames,
}

if video is not None and run_one:
    run_pipeline(video, gt_file, settings, live, verify_replay)
elif video is not None and run_all:
    plan = [("FAST", None), ("ROBUST", None)]
    plan += [("ADAPTIVE", kind) for kind in reversed(trained_selectors())]
    for each_mode, each_selector in plan:
        run_pipeline(
            video,
            gt_file,
            {
                **settings,
                "mode": each_mode,
                "selector": each_selector,
                "threshold": threshold if each_selector in LEARNED_SELECTOR_KINDS else None,
            },
            live,
            verify_replay,
        )
    best = trained_selectors()[0]
    st.session_state["active_run"] = run_label(
        {
            **settings,
            "mode": "ADAPTIVE",
            "selector": best,
            "threshold": threshold if best in LEARNED_SELECTOR_KINDS else None,
        }
    )
    st.session_state["results_tab"] = RESULT_TABS[-1]

if st.session_state["runs"]:
    show_results(st.session_state["runs"], settings)
else:
    show_empty_state()

draw_history(history_slot)
