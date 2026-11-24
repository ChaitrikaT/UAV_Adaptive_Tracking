"""One command: build (if missing) and open the three clearly-labelled demo videos.

  python run_video_demo.py            # serve + open browser
  python run_video_demo.py --rebuild  # regenerate videos 2 and 3 first
Needs: demo/videos/clip_7_detector_review_h264.mp4 (already in the repo), numpy, opencv-python;
imageio-ffmpeg (in requirements.txt) for H.264 encoding. No model, dataset or GPU required.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VIDEOS = ROOT / "demo" / "videos"
PAGE = ROOT / "demo" / "final" / "video_demo.html"


def build(script: str):
    print(f"== running {script}", flush=True)
    subprocess.run([sys.executable, str(ROOT / script)], cwd=str(ROOT), check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    if not (VIDEOS / "clip_7_detector_review_h264.mp4").is_file():
        raise SystemExit("Missing demo/videos/clip_7_detector_review_h264.mp4 (the original detector-review video).")
    if not PAGE.is_file():
        raise SystemExit(f"Missing {PAGE}")
    if a.rebuild or not (VIDEOS / "clip_7_archived_results_replay_h264.mp4").is_file():
        build("demo_archived_review.py")
    if a.rebuild or not (VIDEOS / "synthetic_mechanism_demo_h264.mp4").is_file():
        build("make_synthetic_demo.py")
    handler = partial(SimpleHTTPRequestHandler, directory=str(ROOT / "demo"))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    url = f"http://127.0.0.1:{server.server_port}/final/video_demo.html"
    print(f"Video demo: {url}\nKeep this terminal open while viewing. Ctrl+C to stop.")
    if not a.no_browser:
        threading.Timer(0.3, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
