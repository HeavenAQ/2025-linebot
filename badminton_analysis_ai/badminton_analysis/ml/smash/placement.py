"""First-frame-only camera and standing placement for the accepted local overlay."""

from __future__ import annotations
from numpy.typing import NDArray
import numpy as np


def transport_corrected_by_student_displacement(
    corrected_pixels: NDArray[np.float32],
    detected_pixels: NDArray[np.float32],
    confidence: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Add only the student's global displacement to an anchored correction.

    ``corrected_pixels`` already contains the expert local motion, generated
    root trajectory, ankle--spine view projection, and the existing clip-level
    ankle/knee/hip placement.  Preserve that result exactly at frame zero and
    add one rigid translation equal to the student's smoothed pelvis movement
    from frame zero.  Torso centre and ankle midpoint are confidence fallbacks;
    per-frame foot motion is never the primary driver.
    """
    corrected = np.asarray(corrected_pixels, dtype=np.float32)
    detected = np.asarray(detected_pixels, dtype=np.float32)
    observed = np.asarray(confidence, dtype=np.float32)
    if (
        corrected.shape != detected.shape
        or corrected.ndim != 3
        or corrected.shape[1:] != (17, 2)
    ):
        raise ValueError("student displacement poses must share shape (T, 17, 2)")
    if observed.shape != corrected.shape[:2]:
        raise ValueError("student displacement confidence must have shape (T, 17)")

    pelvis = 0.5 * (detected[:, 11] + detected[:, 12])
    torso = 0.25 * (detected[:, 5] + detected[:, 6] + detected[:, 11] + detected[:, 12])
    ankles = 0.5 * (detected[:, 15] + detected[:, 16])
    pelvis_ok = np.minimum(observed[:, 11], observed[:, 12]) > 0.05
    torso_ok = np.minimum.reduce(observed[:, (5, 6, 11, 12)], axis=1) > 0.05
    ankles_ok = np.minimum(observed[:, 15], observed[:, 16]) > 0.05
    position = np.full((len(corrected), 2), np.nan, dtype=np.float64)
    position[pelvis_ok] = pelvis[pelvis_ok]
    fallback = ~pelvis_ok & torso_ok
    position[fallback] = torso[fallback]
    fallback = ~pelvis_ok & ~torso_ok & ankles_ok
    position[fallback] = ankles[fallback]
    valid = np.isfinite(position).all(axis=1)
    if not np.any(valid):
        return corrected.copy()
    timeline = np.arange(len(position))
    for axis in range(2):
        position[:, axis] = np.interp(timeline, timeline[valid], position[valid, axis])
    # Reject isolated detector jitter without suppressing real player travel.
    smoothed = position.copy()
    padded = np.pad(position, ((2, 2), (0, 0)), mode="edge")
    for frame in range(len(position)):
        smoothed[frame] = np.median(padded[frame : frame + 5], axis=0)
    displacement = smoothed - smoothed[0]
    displacement[0] = 0.0
    return np.asarray(corrected + displacement[:, None], dtype=np.float32)


def smooth_corrected_bbox_placement(
    corrected_pixels: NDArray[np.float32],
    *,
    alpha_current: float = 0.65,
) -> NDArray[np.float32]:
    """Stabilize correction placement with one rigid per-frame translation.

    Generated smash motions can contain a short, implausible excursion in the
    screen-space root even after their root-relative limb pose is repaired.
    Estimate placement from the body bounding-box centre (joints 5--16), apply
    a zero-phase EMA to that centre, and translate every joint by
    the same amount.  This deliberately leaves all local vectors, angles, and
    bone lengths unchanged.  Student displacement is added *after* this step,
    so the player's observed horizontal transport is never smoothed or lagged.

    The first and last frames remain exact to avoid changing the analysis
    window endpoints.
    """
    corrected = np.asarray(corrected_pixels, dtype=np.float32)
    if corrected.ndim != 3 or corrected.shape[1:] != (17, 2):
        raise ValueError("bbox placement smoothing requires shape (T, 17, 2)")
    if not 0.0 < alpha_current <= 1.0:
        raise ValueError("alpha_current must be in (0, 1]")
    if len(corrected) <= 2 or alpha_current >= 1.0:
        return corrected.copy()

    core = corrected[:, 5:17].astype(np.float64)
    anchor = 0.5 * (np.min(core, axis=1) + np.max(core, axis=1))
    forward = anchor.copy()
    for frame in range(1, len(anchor)):
        forward[frame] = (
            alpha_current * anchor[frame] + (1.0 - alpha_current) * forward[frame - 1]
        )
    backward = anchor.copy()
    for frame in range(len(anchor) - 2, -1, -1):
        backward[frame] = (
            alpha_current * anchor[frame] + (1.0 - alpha_current) * backward[frame + 1]
        )
    stable_anchor = 0.5 * (forward + backward)
    stable_anchor[0] = anchor[0]
    stable_anchor[-1] = anchor[-1]
    translation = stable_anchor - anchor
    return np.asarray(corrected + translation[:, None], dtype=np.float32)


MIN_CONFIDENCE = 0.05


def first_frame_ankle_spine_map(corrected_first, detected_first, confidence_first):
    """One proper similarity transform, fitted on the first frame only.

    Match the generated pelvis-to-shoulder spine to the observed spine, then
    ground the same anatomical support ankle. Pelvis defines the spine (and
    subsequently drives player transport); it is not a second competing
    translation target. Preserve stance width, joint angles and later lean.
    No reflection, per-frame fitting, or separate limb placement is allowed.
    """
    corrected = np.asarray(corrected_first, dtype=float)
    detected = np.asarray(detected_first, dtype=float)
    confidence = np.asarray(confidence_first, dtype=float)
    if (
        corrected.shape != (17, 2)
        or detected.shape != (17, 2)
        or confidence.shape != (17,)
    ):
        raise ValueError("Expected two COCO17 poses and 17 confidences")
    torso = [5, 6, 11, 12]
    if (
        not np.isfinite(corrected[torso]).all()
        or not np.isfinite(detected[torso]).all()
        or not np.isfinite(confidence[torso]).all()
        or np.any(confidence[torso] <= MIN_CONFIDENCE)
    ):
        raise ValueError(
            "Reliable first-frame torso required; do not silently refit later"
        )
    spine = lambda p: p[[5, 6]].mean(0) - p[[11, 12]].mean(0)
    source, target = spine(corrected), spine(detected)
    lengths = np.linalg.norm(source), np.linalg.norm(target)
    if min(lengths) <= 1e-6:
        raise ValueError("Nondegenerate first-frame spines required")
    a, b = source / lengths[0], target / lengths[1]
    cosine = np.dot(a, b)
    sine = a[0] * b[1] - a[1] * b[0]
    matrix = np.array([[cosine, sine], [-sine, cosine]]) * (lengths[1] / lengths[0])
    ankles = [
        j
        for j in (15, 16)
        if np.isfinite(corrected[j]).all()
        and np.isfinite(detected[j]).all()
        and np.isfinite(confidence[j])
        and confidence[j] > MIN_CONFIDENCE
    ]
    if not ankles:
        raise ValueError("Reliable first-frame ankle required")
    support = max(ankles, key=lambda j: detected[j, 1])
    translation = detected[support] - corrected[support] @ matrix
    return matrix, translation, support


def first_frame_standing_offsets(corrected_first, detected_first, confidence_first):
    """Retarget initial standing placement once, after camera projection.

    Feet and knees receive their own constant offsets; joints 0..12 receive
    one shared pelvis offset. This is NOT a rigid transform of the legs: it
    changes their initial geometry. Upper-body vectors and every joint's
    subsequent displacement are preserved. Never recalculate these offsets
    from subsequent observed poses or use them to fix actual torso lean.
    """
    q = np.asarray(corrected_first, float)
    p = np.asarray(detected_first, float)
    c = np.asarray(confidence_first, float)
    if q.shape != (17, 2) or p.shape != q.shape or c.shape != (17,):
        raise ValueError("Expected two COCO17 poses and 17 confidences")
    joints = [11, 12, 13, 14, 15, 16]
    if (
        not np.isfinite(q[joints]).all()
        or not np.isfinite(p[joints]).all()
        or not np.isfinite(c[joints]).all()
        or np.any(c[joints] <= MIN_CONFIDENCE)
    ):
        raise ValueError("Reliable first-frame pelvis, knees and ankles required")
    offsets = np.zeros((17, 2), dtype=float)
    offsets[:13] = p[[11, 12]].mean(0) - q[[11, 12]].mean(0)
    offsets[13:] = p[13:] - q[13:]
    return offsets
