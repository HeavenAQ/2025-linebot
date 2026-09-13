"""One pause-free video clock for coaching overlays and frontend cues."""

from typing import Any


def coaching_video_frame(
    issue: dict[str, Any], normalized_length: int, window_length: int
) -> int:
    if "video_frame_index" in issue:
        frame = issue["video_frame_index"]
        if type(frame) is not int or not 0 <= frame < window_length:
            raise ValueError("coaching video frame is outside the analysis window")
        return frame
    frame = int(issue["frame_index"])
    if normalized_length > 1 and window_length > 0:
        return round(
            min(max(frame, 0), normalized_length - 1)
            * (window_length - 1)
            / (normalized_length - 1)
        )
    return frame
