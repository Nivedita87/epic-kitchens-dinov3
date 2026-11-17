"""Load action-recognition segments using EPIC-KITCHENS annotations."""

from pathlib import Path

from src.video_loading import sample_frames


def timestamp_to_seconds(timestamp):
    """Convert a timestamp in HH:MM:SS format to seconds."""
    hours, minutes, seconds = map(float, timestamp.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def load_training_segment(annotation, video_root, num_frames=16):
    """Load frames and labels for one annotated training segment.

    Args:
        annotation: One row from the training annotation table.
        video_root: Folder containing participant video folders.
        num_frames: Number of frames to sample.

    Returns:
        Dictionary containing RGB frames, labels and segment ID.
    """
    video_path = (
        Path(video_root)
        / annotation["participant_id"]
        / f'{annotation["video_id"]}.MP4'
    )

    start = timestamp_to_seconds(annotation["start_timestamp"])
    end = timestamp_to_seconds(annotation["stop_timestamp"])

    frames = sample_frames(video_path, start, end, num_frames)

    return {
        "frames": frames,
        "verb_label": int(annotation["verb_class"]),
        "noun_label": int(annotation["noun_class"]),
        "narration_id": annotation["narration_id"],
    }
