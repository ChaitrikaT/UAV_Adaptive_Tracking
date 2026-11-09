"""Open the already-generated results dashboard with one terminal.

No third-party dependencies, model inference, training, or dataset access.
Run from the project root: python run_demo.py
"""
from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import threading
import webbrowser


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve and open the saved offline results dashboard")
    parser.add_argument("--results-dir", default=None, help="Folder containing final/index.html and videos/")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    results = Path(args.results_dir).resolve() if args.results_dir else root / "results"
    page = results / "final" / "index.html"
    if not page.is_file():
        parser.error(f"Dashboard missing: {page}\nFirst generate the dashboard, or provide --results-dir.")

    html = page.read_text(encoding="utf-8")
    match = re.search(r'<video[^>]*\bsrc=["\']([^"\']+)', html, re.I)
    if match:
        src = match.group(1).split("?")[0]
        video_path = (page.parent / src).resolve()
        if not video_path.exists():
            print(f"WARNING: Dashboard points to missing video: {video_path}")
        elif video_path.suffix.lower() == ".mp4" and "h264" not in video_path.name.lower():
            print("NOTE: Browser video playback may require H.264; use your converted H.264 file if needed.")

    handler = partial(SimpleHTTPRequestHandler, directory=str(results))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    url = f"http://127.0.0.1:{server.server_port}/final/index.html"
    print(f"Dashboard: {url}")
    print("Keep this terminal open while viewing the dashboard. Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.3, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard server stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
