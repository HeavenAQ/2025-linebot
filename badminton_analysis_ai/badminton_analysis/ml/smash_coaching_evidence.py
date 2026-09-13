"""Build coaching evidence from scoring decisions, never rediscover checkpoints.

All inputs use decoded source-frame indices. Conversion to the cropped video's
clock happens once here. Missing lead-in/tail remains explicit; do not clamp a
missing source frame onto an unrelated visible image.
"""

from typing import Any

CRITERIA = (
    "preparation",
    "body_rotation",
    "arm_balance",
    "elbow_forward",
    "wrist_flick",
    "follow_through",
)


def build_checkpoint_evidence(
    *,
    intervals: dict[str, Any],
    window_start: int,
    window_end: int,
    initial_frame: int,
    balance_anchor: int,
    selected_endpoint: int,
    balance_measurement: dict[str, Any],
) -> dict[str, Any]:
    if not 0 <= window_start <= window_end or set(intervals) != set(CRITERIA):
        raise ValueError("ordered source window and six scoring intervals required")
    result = {}
    for reference in CRITERIA:
        interval = intervals[reference]
        start, end = int(interval["start"]), int(interval["end"])
        if reference == "body_rotation":
            start, end = initial_frame, balance_anchor
        elif reference == "follow_through":
            start, end = initial_frame, selected_endpoint
        if start > end:
            raise ValueError(f"unordered scoring interval: {reference}")
        requested = {start, end}
        if reference != "follow_through":
            requested.update((round((start + end) / 2), int(interval["anchor"])))
        if reference == "arm_balance" and balance_measurement.get("available"):
            event_start = int(balance_measurement["event_start"])
            event_end = int(balance_measurement["event_end"])
            if (
                not start <= event_start <= event_end <= end
                or event_end - event_start != 4
            ):
                raise ValueError("balance evidence must be the scored five-frame event")
            requested.update(range(event_start, event_end + 1))
        # Include actual visible interval boundaries if only part is rendered.
        visible_start, visible_end = max(start, window_start), min(end, window_end)
        if visible_start > visible_end:
            raise ValueError(f"no visible scored evidence for {reference}")
        requested.update((visible_start, visible_end))
        available = sorted(
            frame
            for frame in requested
            if window_start <= frame <= window_end and start <= frame <= end
        )
        result[reference] = {
            "source_interval": [start, end],
            "output_frame_indices": [frame - window_start for frame in available],
            "source_frame_indices": available,
            "source_window_start": window_start,
            "full_interval_visible": window_start <= start <= end <= window_end,
            "missing_requested_source_frames": sorted(requested - set(available)),
            "sampling": "scored boundaries, midpoint/anchor, and exact sustained balance event",
        }
        if reference == "arm_balance":
            result[reference]["measurement"] = {
                key: balance_measurement[key]
                for key in (
                    "available",
                    "reason",
                    "handedness",
                    "event_start",
                    "event_end",
                    "sustained_gap",
                    "tolerance",
                    "fraction",
                    "valid_fraction",
                )
                if key in balance_measurement
            }
    return result
