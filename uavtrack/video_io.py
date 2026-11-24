"""Small helpers for writing browser-playable (H.264) MP4 files."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2


def ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def transcode_h264(src: Path, dst: Path) -> str:
    exe = ffmpeg_exe()
    if exe:
        r = subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", str(dst)])
        if r.returncode == 0:
            return "h264"
    shutil.copyfile(src, dst)
    return "mp4v (no ffmpeg found: browser playback may fail; install imageio-ffmpeg)"


class H264Writer:
    """Writes with OpenCV (mp4v) to a temp file, then re-encodes to H.264 if ffmpeg is available."""

    def __init__(self, path: Path, fps: float, size: tuple[int, int]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp = self.path.with_suffix(".tmp.mp4")
        self.w = cv2.VideoWriter(str(self.tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not self.w.isOpened():
            raise RuntimeError(f"Could not open video writer for {self.tmp}")

    def write(self, frame):
        self.w.write(frame)

    def close(self) -> str:
        self.w.release()
        status = transcode_h264(self.tmp, self.path)
        self.tmp.unlink(missing_ok=True)
        return status
