"""Video input/output helpers: reading frames, writing video, and H.264 re-encoding."""

import os
from pathlib import Path
import subprocess
from typing import Dict, Generator, Optional, Tuple, Union
import warnings
import cv2
import imageio_ffmpeg
import numpy as np


def get_video_properties(video_path: Union[str, Path]) -> Dict[str, Union[int, float]]:
    """Retrieve video resolution, FPS, and total frame count.

    Args:
        video_path: Path to video file.

    Returns:
        Dict with 'width', 'height', 'fps', and 'total_frames'.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return {
        "width": width,
        "height": height,
        "fps": float(fps) if fps > 0 else 30.0,
        "total_frames": total_frames,
    }


def read_video_frames(
    video_path: Union[str, Path], max_frames: Optional[int] = None
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """Yield frames sequentially from a video file.

    Args:
        video_path: Path to video file.
        max_frames: Optional maximum number of frames to read.

    Yields:
        Tuple of (frame_index, frame_ndarray_bgr).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    frame_idx = 0
    try:
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            yield frame_idx, frame
            frame_idx += 1
    finally:
        cap.release()


def reencode_to_h264(raw_path: Union[str, Path], output_path: Union[str, Path]) -> str:
    """Re-encode an OpenCV output video into browser-compatible H.264 MP4 using imageio-ffmpeg.

    Args:
        raw_path: Path to raw OpenCV video.
        output_path: Destination path for H.264 video.

    Returns:
        Final playable video file path.
    """
    raw_p = Path(raw_path).resolve()
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i",
            str(raw_p),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            "-preset",
            "fast",
            "-movflags",
            "+faststart",
            str(out_p),
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=120,
        )
        if out_p.is_file() and out_p.stat().st_size > 0:
            if raw_p != out_p and raw_p.exists():
                try:
                    raw_p.unlink()
                except OSError:
                    pass
            return str(out_p)
    except Exception as e:
        warnings.warn(
            f"H.264 re-encoding failed ({e}). Falling back to raw OpenCV video at {raw_p}."
        )

    # Fallback to raw file
    if raw_p != out_p and raw_p.exists():
        if out_p.exists():
            out_p.unlink()
        raw_p.rename(out_p)
    return str(out_p)


class VideoWriterWrapper:
    """Convenient OpenCV video writer with automatic H.264 re-encoding upon release."""

    def __init__(
        self,
        output_path: Union[str, Path],
        fps: float,
        width: int,
        height: int,
    ) -> None:
        """Initialize writer with resolution and target output path.

        Args:
            output_path: Final target path for the video.
            fps: Video frames per second.
            width: Frame width in pixels.
            height: Frame height in pixels.
        """
        self.final_output_path = Path(output_path).resolve()
        self.final_output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write intermediate frames to a temporary raw file
        self.raw_path = self.final_output_path.with_name(
            f"temp_raw_{self.final_output_path.name}"
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(self.raw_path), fourcc, fps, (width, height)
        )
        if not self.writer.isOpened():
            # Try alternate MJPG codec if mp4v fails on platform
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.raw_path = self.raw_path.with_suffix(".avi")
            self.writer = cv2.VideoWriter(
                str(self.raw_path), fourcc, fps, (width, height)
            )

    def write_frame(self, frame: np.ndarray) -> None:
        """Append a frame to the video."""
        self.writer.write(frame)

    def release(self) -> str:
        """Close writer and re-encode to browser-playable H.264 MP4."""
        self.writer.release()
        return reencode_to_h264(self.raw_path, self.final_output_path)

