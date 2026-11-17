"""Utilities for sampling RGB frames from video segments."""

import cv2
import numpy as np


def _sample_frames_once(video_path, start_seconds, end_seconds, num_frames=16):
    """Read evenly spaced frames within a video segment.

    Args:
        video_path: Path to the video.
        start_seconds: Beginning of the segment in seconds.
        end_seconds: End of the segment in seconds, excluded.
        num_frames: Number of frames to select.

    Returns:
        RGB array with shape (num_frames, height, width, 3).
    """
    if start_seconds < 0 or end_seconds <= start_seconds:
        raise ValueError("The segment must have valid start and end times.")
    if num_frames < 1:
        raise ValueError("num_frames must be positive.")

    capture = cv2.VideoCapture(str(video_path))

    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise RuntimeError("Cannot determine the video frame rate.")

        # Sample inside each time interval, avoiding the segment boundaries.
        interval = (end_seconds - start_seconds) / num_frames
        times = start_seconds + (np.arange(num_frames) + 0.5) * interval

        frames = []
        for time_seconds in times:
            frame_index = int(np.floor(time_seconds * fps))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()

            if not success:
                raise RuntimeError(f"Cannot read frame {frame_index}.")

            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        return np.stack(frames)
    finally:
        capture.release()


def sample_frames(video_path, start_seconds, end_seconds, num_frames=16):
    """Retry transient decoder failures without skipping any clips."""
    import time

    for attempt in range(1, 4):
        try:
            return _sample_frames_once(
                video_path, start_seconds, end_seconds, num_frames
            )
        except RuntimeError as error:
            if attempt == 3:
                raise RuntimeError(
                    f"Video sampling failed after 3 attempts: "
                    f"{video_path}, interval {start_seconds}-{end_seconds}"
                ) from error
            print(
                f"Video read attempt {attempt}/3 failed: {error}. Retrying.",
                flush=True,
            )
            time.sleep(1)
