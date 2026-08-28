"""Phase-constrained distances between a detected and corrected skeleton.

The functions in this module are deliberately label-free.  They compare a
learner trajectory with the generated expert correction in the learner's
canonical body/view coordinate system.  Human scores are evaluation data and
never enter these calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from numpy.typing import NDArray


TrajectoryDistance = Literal[
    "euclidean",
    "dtw",
    "derivative_dtw",
    "shape_dtw",
    "multi_feature_dtw",
]

_ANGLE_TRIPLETS = (
    (5, 7, 9),
    (6, 8, 10),
    (7, 5, 11),
    (8, 6, 12),
    (5, 11, 13),
    (6, 12, 14),
    (11, 13, 15),
    (12, 14, 16),
)
_EPS = 1e-8

_SERVE_MANIFOLD_TRIPLETS = (
    (5, 7, 9),
    (6, 8, 10),
    (11, 5, 7),
    (12, 6, 8),
    (5, 11, 13),
    (6, 12, 14),
    (11, 13, 15),
    (12, 14, 16),
    (5, 11, 12),
    (6, 12, 11),
)


@dataclass(frozen=True)
class ServeAngleManifold:
    standardized_experts: NDArray[np.float64]
    feature_median: NDArray[np.float64]
    feature_scale: NDArray[np.float64]
    expert_q80: float
    expert_scale: float


@dataclass(frozen=True)
class SmashMotionManifold:
    """Robust expert support for signed, phase-aligned overhead motion."""

    standardized_experts: NDArray[np.float64]
    feature_median: NDArray[np.float64]
    feature_scale: NDArray[np.float64]
    expert_q80: float
    expert_scale: float


@dataclass(frozen=True)
class SmashTrajectoryScorer:
    distance_method: TrajectoryDistance
    fusion: str
    criterion_ids: tuple[str, ...]
    criterion_maxima: NDArray[np.float64]
    start_fractions: NDArray[np.float64]
    end_fractions: NDArray[np.float64]
    joints: tuple[tuple[int, ...], ...]
    tolerance: NDArray[np.float64]
    scale: NDArray[np.float64]
    manifold: SmashMotionManifold


def _pelvis_relative(pose: NDArray[np.floating]) -> NDArray[np.float64]:
    values = np.asarray(pose, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (17, 2):
        raise ValueError("pose must have shape (T, 17, 2)")
    pelvis = 0.5 * (values[:, 11] + values[:, 12])
    return values - pelvis[:, None]


def _joint_angles(pose: NDArray[np.floating]) -> NDArray[np.float64]:
    values = np.asarray(pose, dtype=np.float64)
    output = []
    for first, centre, last in _ANGLE_TRIPLETS:
        incoming = values[:, first] - values[:, centre]
        outgoing = values[:, last] - values[:, centre]
        denominator = np.maximum(
            np.linalg.norm(incoming, axis=-1)
            * np.linalg.norm(outgoing, axis=-1),
            _EPS,
        )
        cosine = np.sum(incoming * outgoing, axis=-1) / denominator
        output.append(np.arccos(np.clip(cosine, -1.0, 1.0)) / np.pi)
    return np.stack(output, axis=-1)


def serve_angle_manifold_feature(
    pose: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Return a compact camera/scale-robust serve trajectory descriptor."""

    values = np.asarray(pose, dtype=np.float64)
    if values.shape != (64, 17, 2):
        raise ValueError("serve manifold pose must have shape (64, 17, 2)")
    trajectories = []
    for first, centre, last in _SERVE_MANIFOLD_TRIPLETS:
        incoming = values[:, first] - values[:, centre]
        outgoing = values[:, last] - values[:, centre]
        denominator = np.maximum(
            np.linalg.norm(incoming, axis=-1)
            * np.linalg.norm(outgoing, axis=-1),
            1e-6,
        )
        cosine = np.sum(incoming * outgoing, axis=-1) / denominator
        trajectories.append(
            np.arccos(np.clip(cosine, -1.0, 1.0)) / np.pi
        )
    sampled = np.stack(trajectories, axis=-1)[
        np.linspace(0, 63, 16).round().astype(np.int64)
    ]
    derivative = np.diff(sampled, axis=0, prepend=sampled[:1])
    return np.concatenate((sampled.ravel(), derivative.ravel()))


def fit_serve_angle_manifold(
    expert_pose: NDArray[np.floating],
    expert_subject_ids: Sequence[str],
) -> ServeAngleManifold:
    """Fit identity-held-out novelty bounds from expert motions only."""

    poses = np.asarray(expert_pose, dtype=np.float64)
    subjects = np.asarray(tuple(expert_subject_ids))
    if poses.ndim != 4 or poses.shape[1:] != (64, 17, 2):
        raise ValueError("expert serve poses must have shape (N, 64, 17, 2)")
    if len(subjects) != len(poses) or len(set(subjects.tolist())) < 2:
        raise ValueError("serve manifold requires at least two expert identities")
    features = np.stack([serve_angle_manifold_feature(pose) for pose in poses])
    median = np.median(features, axis=0)
    scale = np.maximum(
        1.4826 * np.median(np.abs(features - median), axis=0),
        0.03,
    )
    standardized = (features - median) / scale
    held_out_distances = []
    for index, subject in enumerate(subjects):
        other_identity = standardized[subjects != subject]
        distance = np.sqrt(
            np.mean(np.square(other_identity - standardized[index]), axis=1)
        )
        held_out_distances.append(float(np.min(distance)))
    held_out = np.asarray(held_out_distances, dtype=np.float64)
    q80 = float(np.quantile(held_out, 0.80))
    robust_scale = max(float(np.quantile(held_out, 0.95)) - q80, 0.5)
    return ServeAngleManifold(
        standardized_experts=standardized,
        feature_median=median,
        feature_scale=scale,
        expert_q80=q80,
        expert_scale=robust_scale,
    )


def serve_angle_manifold_distance(
    pose: NDArray[np.floating], manifold: ServeAngleManifold
) -> float:
    feature = (
        serve_angle_manifold_feature(pose) - manifold.feature_median
    ) / manifold.feature_scale
    distances = np.sqrt(
        np.mean(
            np.square(manifold.standardized_experts - feature),
            axis=1,
        )
    )
    return float(np.min(distances))


def smash_motion_manifold_feature(
    pose: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Describe smash shape, direction, and ordering without image translation.

    Joint angles alone cannot distinguish a forward overhead swing from its
    mirrored/backward counterpart.  Signed segment orientations (encoded as
    sine/cosine) and body-frame arm positions retain that information.
    """
    values = np.asarray(pose, dtype=np.float64)
    if values.shape != (64, 17, 2):
        raise ValueError("smash manifold pose must have shape (64, 17, 2)")
    pelvis = 0.5 * (values[:, 11] + values[:, 12])
    shoulder = 0.5 * (values[:, 5] + values[:, 6])
    torso = max(float(np.median(np.linalg.norm(shoulder - pelvis, axis=-1))), 1e-6)
    local = (values - pelvis[:, None]) / torso
    angles = _joint_angles(values)
    orientations = []
    for first, second in ((5, 6), (11, 12), (6, 8), (8, 10)):
        segment = local[:, second] - local[:, first]
        phase = np.arctan2(segment[:, 1], segment[:, 0])
        orientations.extend((np.sin(phase), np.cos(phase)))
    signed_orientations = np.stack(orientations, axis=-1)
    arm_positions = local[:, (7, 8, 9, 10)].reshape(len(local), -1)
    trajectory = np.concatenate(
        (angles, signed_orientations, arm_positions), axis=-1
    )
    sampled = trajectory[np.linspace(0, 63, 16).round().astype(np.int64)]
    derivative = np.diff(sampled, axis=0, prepend=sampled[:1])
    return np.concatenate((sampled.ravel(), derivative.ravel()))


def fit_smash_motion_manifold(
    expert_pose: NDArray[np.floating],
    expert_subject_ids: Sequence[str],
) -> SmashMotionManifold:
    poses = np.asarray(expert_pose, dtype=np.float64)
    subjects = np.asarray(tuple(expert_subject_ids))
    if poses.ndim != 4 or poses.shape[1:] != (64, 17, 2):
        raise ValueError("expert smash poses must have shape (N, 64, 17, 2)")
    if len(subjects) != len(poses) or len(set(subjects.tolist())) < 2:
        raise ValueError("smash manifold requires at least two expert identities")
    features = np.stack([smash_motion_manifold_feature(pose) for pose in poses])
    median = np.median(features, axis=0)
    scale = np.maximum(
        1.4826 * np.median(np.abs(features - median), axis=0),
        0.03,
    )
    standardized = (features - median) / scale
    held_out_distances = []
    for index, subject in enumerate(subjects):
        other_identity = standardized[subjects != subject]
        distances = np.sqrt(
            np.mean(np.square(other_identity - standardized[index]), axis=1)
        )
        held_out_distances.append(float(np.min(distances)))
    held_out = np.asarray(held_out_distances, dtype=np.float64)
    q80 = float(np.quantile(held_out, 0.80))
    robust_scale = max(float(np.quantile(held_out, 0.95)) - q80, 0.5)
    return SmashMotionManifold(
        standardized_experts=standardized,
        feature_median=median,
        feature_scale=scale,
        expert_q80=q80,
        expert_scale=robust_scale,
    )


def smash_motion_manifold_distance(
    pose: NDArray[np.floating], manifold: SmashMotionManifold
) -> float:
    feature = (
        smash_motion_manifold_feature(pose) - manifold.feature_median
    ) / manifold.feature_scale
    distances = np.sqrt(
        np.mean(np.square(manifold.standardized_experts - feature), axis=1)
    )
    return float(np.min(distances))


def load_smash_trajectory_scorer(path: str | Path) -> SmashTrajectoryScorer:
    with np.load(Path(path), allow_pickle=False) as archive:
        if str(archive["method"].item()) != "smash_expert_corrected_trajectory_v1":
            raise ValueError("not a smash corrected-trajectory scorer")
        if bool(archive["student_data_used_for_training_or_calibration"].item()):
            raise ValueError("smash trajectory scorer used forbidden student calibration")
        distance_method = str(archive["distance_method"].item())
        if distance_method not in {
            "euclidean", "dtw", "derivative_dtw", "shape_dtw",
            "multi_feature_dtw",
        }:
            raise ValueError(f"unsupported smash trajectory distance: {distance_method}")
        criterion_ids = tuple(str(value) for value in archive["criterion_ids"])
        joints = tuple(
            tuple(int(value) for value in str(serialized).split(","))
            for serialized in archive["criterion_joints"]
        )
        manifold = SmashMotionManifold(
            standardized_experts=np.asarray(
                archive["manifold_standardized_experts"], dtype=np.float64
            ),
            feature_median=np.asarray(
                archive["manifold_feature_median"], dtype=np.float64
            ),
            feature_scale=np.asarray(
                archive["manifold_feature_scale"], dtype=np.float64
            ),
            expert_q80=float(archive["manifold_expert_q80"].item()),
            expert_scale=float(archive["manifold_expert_scale"].item()),
        )
        scorer = SmashTrajectoryScorer(
            distance_method=distance_method,  # type: ignore[arg-type]
            fusion=str(archive["fusion"].item()),
            criterion_ids=criterion_ids,
            criterion_maxima=np.asarray(
                archive["criterion_maxima"], dtype=np.float64
            ),
            start_fractions=np.asarray(
                archive["criterion_start_fractions"], dtype=np.float64
            ),
            end_fractions=np.asarray(
                archive["criterion_end_fractions"], dtype=np.float64
            ),
            joints=joints,
            tolerance=np.asarray(archive["tolerance"], dtype=np.float64),
            scale=np.asarray(archive["scale"], dtype=np.float64),
            manifold=manifold,
        )
    count = len(scorer.criterion_ids)
    if count != 6 or any(
        len(values) != count
        for values in (
            scorer.criterion_maxima,
            scorer.start_fractions,
            scorer.end_fractions,
            scorer.joints,
            scorer.tolerance,
            scorer.scale,
        )
    ):
        raise ValueError("smash trajectory scorer must describe six checkpoints")
    if scorer.fusion not in {"manifold_gate", "extreme_manifold_gate"}:
        raise ValueError(f"unsupported smash trajectory fusion: {scorer.fusion}")
    return scorer


def apply_smash_trajectory_score(
    semantic_score: dict[str, Any],
    learner_pose: NDArray[np.floating],
    corrected_pose: NDArray[np.floating],
    scorer: SmashTrajectoryScorer,
) -> dict[str, Any]:
    """Apply an expert-only trajectory gate to semantic smash evidence."""
    semantic_by_id = {
        str(item["rule_reference"]): item for item in semantic_score["criteria"]
    }
    if tuple(semantic_by_id) != scorer.criterion_ids:
        raise ValueError("semantic and trajectory smash criteria do not match")
    manifold_distance = smash_motion_manifold_distance(
        learner_pose, scorer.manifold
    )
    manifold_ratio = expert_residual_ratio(
        manifold_distance,
        tolerance=scorer.manifold.expert_q80,
        scale=scorer.manifold.expert_scale,
    )
    boundary = float(
        np.exp(-1.0) if scorer.fusion == "manifold_gate" else np.exp(-2.0)
    )
    outside_support = manifold_ratio < boundary
    criteria = []
    for index, criterion_id in enumerate(scorer.criterion_ids):
        semantic = semantic_by_id[criterion_id]
        semantic_ratio = float(semantic["ratio"])
        cost = corrected_motion_distance(
            learner_pose,
            corrected_pose,
            joints=scorer.joints[index],
            start_fraction=float(scorer.start_fractions[index]),
            end_fraction=float(scorer.end_fractions[index]),
            method=scorer.distance_method,
        )
        residual_ratio = expert_residual_ratio(
            cost,
            tolerance=float(scorer.tolerance[index]),
            scale=float(scorer.scale[index]),
        )
        should_fuse = outside_support and residual_ratio < boundary
        ratio = (
            float(np.sqrt(max(semantic_ratio * residual_ratio, 0.0)))
            if should_fuse
            else semantic_ratio
        )
        criteria.append(
            {
                **semantic,
                "ratio": ratio,
                "score": float(100.0 / 6.0 * ratio),
                "semantic_ratio_before_trajectory": semantic_ratio,
                "trajectory_cost": cost,
                "trajectory_residual_ratio": residual_ratio,
                "trajectory_fused": should_fuse,
            }
        )
    ratios = np.asarray([item["ratio"] for item in criteria], dtype=np.float64)
    total = float(100.0 * np.exp(np.mean(np.log(np.maximum(ratios, 0.03)))))
    return {
        **semantic_score,
        "total_score": total,
        "criteria": criteria,
        "score_method": (
            "smash_serve_style_phase_aligned_euclidean_v1"
            if scorer.fusion == "manifold_gate"
            and scorer.distance_method == "euclidean"
            else "smash_semantic_multi_feature_dtw_extreme_manifold_v1"
        ),
        "trajectory_diagnostics": {
            "distance_method": scorer.distance_method,
            "fusion": scorer.fusion,
            "manifold_distance": manifold_distance,
            "manifold_ratio": manifold_ratio,
            "outside_expert_support": outside_support,
            "support_boundary_ratio": boundary,
            "fused_criterion_count": int(
                sum(bool(item["trajectory_fused"]) for item in criteria)
            ),
        },
    }


def _velocity(values: NDArray[np.floating]) -> NDArray[np.float64]:
    trajectory = np.asarray(values, dtype=np.float64)
    # Four canonical frames are approximately one sixteenth of a 64-frame
    # motion.  This factor keeps velocity and pose residuals on comparable
    # dimensionless scales without depending on the source video's FPS.
    return np.vstack(
        (np.zeros((1, trajectory.shape[1])), np.diff(trajectory, axis=0))
    ) * 4.0


def _shape_descriptor(values: NDArray[np.floating]) -> NDArray[np.float64]:
    trajectory = np.asarray(values, dtype=np.float64)
    padded = np.pad(trajectory, ((2, 2), (0, 0)), mode="edge")
    smooth = sum(padded[offset : offset + len(trajectory)] for offset in range(5))
    smooth /= 5.0
    return np.concatenate((trajectory, _velocity(smooth)), axis=-1)


def constrained_dtw_cost(
    source: NDArray[np.floating],
    target: NDArray[np.floating],
    *,
    radius: int,
    warp_penalty: float = 0.015,
) -> float:
    """Return path-length-normalized dependent multivariate DTW cost.

    A Sakoe--Chiba band and an off-diagonal penalty prevent a missing movement
    from being hidden by a pathological many-to-one alignment.
    """

    left = np.asarray(source, dtype=np.float64)
    right = np.asarray(target, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("DTW inputs must have shapes (T, D) and (U, D)")
    if not len(left) or not len(right):
        raise ValueError("DTW inputs cannot be empty")
    if radius < abs(len(left) - len(right)):
        raise ValueError("DTW radius cannot connect both sequence endpoints")
    if warp_penalty < 0.0:
        raise ValueError("DTW warp penalty cannot be negative")

    accumulated = np.full((len(left) + 1, len(right) + 1), np.inf)
    path_length = np.zeros((len(left) + 1, len(right) + 1), dtype=np.int64)
    accumulated[0, 0] = 0.0
    for source_index in range(1, len(left) + 1):
        start = max(1, source_index - radius)
        end = min(len(right), source_index + radius)
        for target_index in range(start, end + 1):
            local = float(
                np.sqrt(
                    np.mean(
                        np.square(
                            left[source_index - 1] - right[target_index - 1]
                        )
                    )
                )
            )
            predecessors = (
                (accumulated[source_index - 1, target_index - 1], 0.0),
                (accumulated[source_index - 1, target_index], warp_penalty),
                (accumulated[source_index, target_index - 1], warp_penalty),
            )
            choice = min(
                range(3),
                key=lambda index: predecessors[index][0]
                + predecessors[index][1],
            )
            if choice == 0:
                previous = (source_index - 1, target_index - 1)
            elif choice == 1:
                previous = (source_index - 1, target_index)
            else:
                previous = (source_index, target_index - 1)
            accumulated[source_index, target_index] = (
                local + predecessors[choice][0] + predecessors[choice][1]
            )
            path_length[source_index, target_index] = (
                path_length[previous] + 1
            )
    length = int(path_length[len(left), len(right)])
    if length == 0 or not np.isfinite(accumulated[len(left), len(right)]):
        raise ValueError("DTW constraints do not admit a valid path")
    return float(accumulated[len(left), len(right)] / length)


def corrected_motion_distance(
    learner_pose: NDArray[np.floating],
    corrected_pose: NDArray[np.floating],
    *,
    joints: Sequence[int],
    start_fraction: float,
    end_fraction: float,
    method: TrajectoryDistance,
) -> float:
    """Compare one qualitative checkpoint over motion-completeness bounds."""

    learner = _pelvis_relative(learner_pose)
    corrected = _pelvis_relative(corrected_pose)
    if learner.shape != corrected.shape:
        raise ValueError("learner and corrected poses must have the same shape")
    if not 0.0 <= start_fraction < end_fraction <= 1.0:
        raise ValueError("motion bounds must satisfy 0 <= start < end <= 1")
    selected = np.asarray(tuple(joints), dtype=np.int64)
    if selected.ndim != 1 or not len(selected):
        raise ValueError("at least one joint is required")

    start = min(int(np.floor(start_fraction * len(learner))), len(learner) - 1)
    end = min(
        len(learner),
        max(start + 1, int(np.ceil(end_fraction * len(learner)))),
    )
    learner_position = learner[:, selected].reshape(len(learner), -1)[start:end]
    corrected_position = corrected[:, selected].reshape(len(corrected), -1)[
        start:end
    ]
    if method == "euclidean":
        return float(
            np.sqrt(np.mean(np.square(learner_position - corrected_position)))
        )

    radius = max(2, int(round(0.10 * len(learner_position))))
    if method == "dtw":
        return constrained_dtw_cost(
            learner_position,
            corrected_position,
            radius=radius,
        )
    learner_velocity = _velocity(learner_position)
    corrected_velocity = _velocity(corrected_position)
    if method == "derivative_dtw":
        return constrained_dtw_cost(
            learner_velocity,
            corrected_velocity,
            radius=radius,
        )
    if method == "shape_dtw":
        return constrained_dtw_cost(
            _shape_descriptor(learner_position),
            _shape_descriptor(corrected_position),
            radius=radius,
        )
    if method == "multi_feature_dtw":
        learner_angles = _joint_angles(learner)[start:end]
        corrected_angles = _joint_angles(corrected)[start:end]
        learner_features = np.concatenate(
            (learner_position, 1.25 * learner_angles, 0.75 * learner_velocity),
            axis=-1,
        )
        corrected_features = np.concatenate(
            (
                corrected_position,
                1.25 * corrected_angles,
                0.75 * corrected_velocity,
            ),
            axis=-1,
        )
        return constrained_dtw_cost(
            learner_features,
            corrected_features,
            radius=radius,
        )
    raise ValueError(f"unsupported trajectory distance: {method}")


def expert_residual_ratio(
    cost: float,
    *,
    tolerance: float,
    scale: float,
) -> float:
    """Map a residual to [0, 1] using expert-only calibration statistics."""

    if scale <= 0.0:
        raise ValueError("expert residual scale must be positive")
    excess = max(0.0, float(cost) - float(tolerance))
    return float(np.exp(-excess / scale))
