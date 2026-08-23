from math import inf
from typing import Any, Literal, final

import numpy as np
from numpy.typing import NDArray

from badminton_analysis.core.logger import Logger
from badminton_analysis.models.joints import JOINTS
from badminton_analysis.models.types import (
    AngleDict,
    Coordinate,
    CoordinateDict,
    Skill,
    StepSequence,
)
from badminton_analysis.services.pose_detector import PoseDetector
from badminton_analysis.models.constants import (
    SMOOTHING_WINDOW_SIZE,
    IMPACT_FRAME_SEARCH_WINDOW_BEFORE,
    IMPACT_FRAME_SEARCH_WINDOW_AFTER,
    ANALYSIS_WINDOW_PADDING_BEFORE,
)


@final
class VideoAnalyzer:
    logger = Logger("VideoAnalyzer")

    @staticmethod
    def moving_average(
        positions: NDArray[np.floating[Any]] | list[Coordinate],
        window_size: int = 5,
        pad_mode: Literal["edge"] | Literal["reflect"] = "edge",
    ) -> NDArray[np.floating[Any]]:
        pos = np.asarray(positions, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[1] < 2:
            raise ValueError("positions must have shape (N, D), D >= 2")
        k = np.ones(window_size) / window_size
        pad = window_size // 2
        smoothed = []
        for dim in range(pos.shape[1]):
            padded = np.pad(pos[:, dim], (pad, pad), mode=pad_mode)
            smoothed.append(np.convolve(padded, k, mode="valid"))
        return np.column_stack(smoothed)

    @staticmethod
    def calc_velocity(
        positions: NDArray[np.floating[Any]] | list[Coordinate],
        dt: float,
        n: int = 1,
    ) -> NDArray[np.floating[Any]]:
        positions = np.asarray(positions, dtype=np.float64)
        pos_shift = positions[n:] - positions[:-n]
        velocity = 10.0 * (np.linalg.norm(pos_shift, axis=1) / (n * dt))
        return np.asarray(velocity, dtype=np.float64)

    @staticmethod
    def calc_acceleration(
        velocities: NDArray[np.floating[Any]],
        dt: float,
        n: int = 1,
    ) -> NDArray[np.floating[Any]]:
        return 10.0 * (np.diff(velocities) / (n * dt))

    @classmethod
    def find_acc_analysis_window(
        cls,
        hand_positions: list[Coordinate],
        anchor_positions: list[Coordinate] | None = None,
    ) -> tuple[int, int, int]:
        positions = cls.moving_average(
            hand_positions, window_size=SMOOTHING_WINDOW_SIZE
        )
        if anchor_positions is not None:
            anchors = cls.moving_average(
                anchor_positions, window_size=SMOOTHING_WINDOW_SIZE
            )
            if anchors.shape != positions.shape:
                raise ValueError("anchor positions must align with hand positions")
            positions = positions - anchors
        peak_frame = cls._directional_acceleration_peak(positions)
        start_frame = max(0, peak_frame - IMPACT_FRAME_SEARCH_WINDOW_BEFORE)
        end_frame = min(
            len(hand_positions) - 1, peak_frame + IMPACT_FRAME_SEARCH_WINDOW_AFTER
        )
        return start_frame, peak_frame, end_frame

    @staticmethod
    def _directional_acceleration_peak(
        positions: NDArray[np.floating[Any]],
        *,
        prefer_peak_velocity: bool = False,
    ) -> int:
        """Select the coherent forward swing rather than faster recovery."""
        trajectory = np.asarray(positions, dtype=np.float64)
        velocities = np.diff(trajectory, axis=0)
        if len(velocities) < 2:
            return 0
        speeds = np.linalg.norm(velocities, axis=1)
        if not np.any(speeds > 1e-8):
            return 0
        window = min(7, len(velocities))
        if window % 2 == 0:
            window -= 1
        window = max(window, 1)
        kernel = np.ones(window, dtype=np.float64) / window
        padding = window // 2
        coherent_velocity = np.column_stack(
            [
                np.convolve(
                    np.pad(
                        velocities[:, dimension],
                        (padding, padding),
                        mode="edge",
                    ),
                    kernel,
                    mode="valid",
                )
                for dimension in range(velocities.shape[1])
            ]
        )
        mean_speed = np.convolve(
            np.pad(speeds, (padding, padding), mode="edge"),
            kernel,
            mode="valid",
        )
        coherent_speed = np.linalg.norm(coherent_velocity, axis=1)
        consistency = coherent_speed / np.maximum(mean_speed, 1e-8)
        episode_score = (
            coherent_speed
            * np.square(consistency)
            * np.linspace(1.0, 0.65, len(coherent_speed))
        )
        if padding and len(episode_score) > 2 * padding:
            episode_score[:padding] = 0.0
            episode_score[-padding:] = 0.0
        episode_index = int(np.argmax(episode_score))
        axis = coherent_velocity[episode_index]
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-8:
            episode_index = int(np.argmax(speeds))
            axis = velocities[episode_index]
            axis_norm = float(np.linalg.norm(axis))
        forward_axis = axis / max(axis_norm, 1e-8)
        projected_velocity = velocities @ forward_axis
        directional_acceleration = np.diff(projected_velocity)
        direction_cosine = projected_velocity / np.maximum(speeds, 1e-8)
        search_start = max(0, episode_index - window)
        search_end = min(
            len(directional_acceleration), episode_index + padding + 1
        )
        candidate_velocity = projected_velocity[search_start + 1 : search_end + 1]
        candidate_alignment = direction_cosine[search_start + 1 : search_end + 1]
        if prefer_peak_velocity and candidate_velocity.size:
            return int(
                search_start
                + np.argmax(
                    candidate_velocity * np.clip(candidate_alignment, 0.0, 1.0)
                )
                + 2
            )
        score = (
            np.maximum(
                directional_acceleration[search_start:search_end], 0.0
            )
            * np.clip(candidate_alignment, 0.0, 1.0)
            * (candidate_velocity > 0.0)
        )
        return (
            int(search_start + np.argmax(score) + 2)
            if score.size and np.any(score > 0.0)
            else int(np.argmax(projected_velocity) + 1)
        )

    @staticmethod
    def dynamic_time_warping(
        from_signal: list[Coordinate],
        to_signal: list[Coordinate],
    ) -> tuple[NDArray[np.int64], float]:
        from_sig = np.asarray(from_signal)
        to_sig = np.asarray(to_signal)

        # init
        M, N = from_sig.shape[0] + 1, to_sig.shape[0] + 1
        dp = np.full((M, N), inf, dtype=np.float64)
        dp[0, 0] = 0.0

        # calculate minimum cost
        for i in range(1, M):
            for j in range(1, N):
                penalty = np.min(
                    [
                        dp[i - 1, j - 1],  # match
                        dp[i - 1, j],  # delete
                        dp[i, j - 1],  # insert
                    ]
                )
                diff = np.linalg.norm(from_sig[i - 1] - to_sig[j - 1])
                dp[i, j] = diff + penalty

        # the total cost mapping from_signal to to_signal
        cost = dp[M - 1][N - 1]

        # get the mapping
        path = []
        i, j = M - 1, N - 1
        while i > 0 and j > 0:
            path.append((i - 1, j - 1))
            choices = [
                dp[i - 1, j - 1],
                dp[i - 1, j],
                dp[i, j - 1],
            ]

            move = np.argmin(choices)
            if move == 0:
                i -= 1
                j -= 1
            elif move == 1:
                i -= 1
            else:
                j -= 1

        path.reverse()
        return np.asarray(path), cost

    @classmethod
    def __find_smash_analysis_window(
        cls,
        hand_positions: list[Coordinate],
        elbow_positions: list[Coordinate],
    ) -> tuple[int, int, int]:
        start_frame, _, acceleration_end_frame = cls.find_acc_analysis_window(
            hand_positions, elbow_positions
        )
        idx = np.argmin(
            np.asarray(hand_positions)[start_frame : acceleration_end_frame + 1, 1]
        )
        new_peak = int(idx + start_frame)
        new_start = max(0, new_peak - 2 * IMPACT_FRAME_SEARCH_WINDOW_BEFORE)
        # The acceleration window identifies impact, but a slower beginner can
        # finish the follow-through well after it. Search the complete
        # post-impact elbow path so the correction reaches its final pose when
        # the player's arm has actually come down.
        new_end = int(
            np.argmax(np.asarray(elbow_positions)[new_peak:, 1]) + new_peak
        )
        minimum_follow_through = max(4, IMPACT_FRAME_SEARCH_WINDOW_AFTER // 2)
        new_end = max(
            new_end,
            acceleration_end_frame,
            new_peak + minimum_follow_through,
        )
        new_end = min(len(hand_positions) - 1, new_end)
        return new_start, new_peak, new_end

    @classmethod
    def __find_serve_analysis_window(
        cls,
        hand_positions: list[Coordinate],
        elbow_positions: list[Coordinate],
    ) -> tuple[int, int, int]:
        start_frame, peak_frame, end_frame = cls.find_acc_analysis_window(
            hand_positions, elbow_positions
        )
        acceleration_end_frame = end_frame
        sub_range_positions = hand_positions[int(start_frame) : int(end_frame)]
        arr = np.asarray(sub_range_positions, dtype=np.float64)
        if arr.size > 0:
            y_values = arr[:, 1]
            lowest_hand_relative_index = int(np.argmax(y_values))
            peak_frame = start_frame + lowest_hand_relative_index
        subset_elbow_pos = elbow_positions[peak_frame:]
        arr_elbow = np.asarray(subset_elbow_pos, dtype=np.float64)
        composite_metric = (
            arr_elbow[:, 0] - arr_elbow[:, 1] if arr_elbow.size > 0 else np.array([])
        )
        relative_end_index = (
            int(np.argmax(composite_metric)) if composite_metric.size > 0 else 0
        )
        end_frame = int(peak_frame) + int(relative_end_index)
        end_frame = max(
            acceleration_end_frame,
            end_frame,
            peak_frame + IMPACT_FRAME_SEARCH_WINDOW_AFTER,
        )
        start_frame = max(0, peak_frame - ANALYSIS_WINDOW_PADDING_BEFORE)
        final_end_frame = min(len(hand_positions) - 1, end_frame)
        return int(start_frame), int(peak_frame), int(final_end_frame)

    @classmethod
    def __find_lift_analysis_window(
        cls,
        hand_positions: list[Coordinate],
        elbow_positions: list[Coordinate],
    ) -> tuple[int, int, int]:
        phases = cls.__find_lift_analysis_phases(hand_positions, elbow_positions)
        return phases[0], phases[2], phases[4]

    @classmethod
    def __find_lift_analysis_phases(
        cls,
        hand_positions: list[Coordinate],
        elbow_positions: list[Coordinate] | None = None,
    ) -> tuple[int, int, int, int, int]:
        positions = cls.moving_average(
            hand_positions, window_size=SMOOTHING_WINDOW_SIZE
        )
        last_frame = len(positions) - 1
        if last_frame < 8:
            raise ValueError("lift clip is too short for phase parsing")
        speeds = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        forward = min(last_frame - 1, int(np.argmax(speeds)) + 1)

        backswing_search_start = max(1, forward - 25)
        backswing_search_end = max(
            backswing_search_start + 1,
            forward - 5,
        )
        forward_velocity = positions[forward] - positions[forward - 1]
        forward_speed = float(np.linalg.norm(forward_velocity))
        if forward_speed > 1e-6:
            forward_axis = forward_velocity / forward_speed
            forward_projection = positions @ forward_axis
            backswing = backswing_search_start + int(
                np.argmin(
                    forward_projection[
                        backswing_search_start:backswing_search_end
                    ]
                )
            )
        else:
            backswing = backswing_search_start + int(
                np.argmin(speeds[backswing_search_start:backswing_search_end])
            )
        backswing = min(forward - 2, backswing)
        start = max(0, backswing - 15)

        peak_speed = float(speeds[max(0, forward - 1)])
        settle_threshold = 0.18 * peak_speed
        settle_frames = 5
        completion = last_frame
        settle_search_start = min(last_frame, forward + 7)
        for candidate in range(
            settle_search_start,
            max(settle_search_start, last_frame - settle_frames + 1),
        ):
            if np.all(
                speeds[candidate : candidate + settle_frames]
                <= settle_threshold
            ):
                completion = min(last_frame, candidate + settle_frames)
                break

        forward = max(backswing + 1, forward)
        completion = max(forward + 1, completion)
        completion = min(last_frame, completion)
        if completion <= forward:
            completion = last_frame
        transition = (start + backswing) // 2
        if transition <= start:
            transition = start + 1
        if transition >= backswing:
            transition = backswing - 1
        if not start < transition < backswing < forward < completion:
            raise ValueError("lift clip does not contain five ordered motion phases")
        return start, transition, backswing, forward, completion

    @classmethod
    def find_analysis_phases(
        cls,
        *,
        skill: Skill,
        hand_positions: list[Coordinate] | None,
        elbow_positions: list[Coordinate] | None,
    ) -> tuple[int, int, int, int, int]:
        if skill == Skill.LIFT and hand_positions:
            return cls.__find_lift_analysis_phases(hand_positions, elbow_positions)
        start, peak, end = cls.find_analysis_window(
            skill=skill,
            hand_positions=hand_positions,
            elbow_positions=elbow_positions,
        )
        return start, (start + peak) // 2, peak, (peak + end) // 2, end

    @classmethod
    def find_footwork_patterns(
        cls,
        right_foot_positions: list[Coordinate],
        left_foot_positions: list[Coordinate],
    ) -> StepSequence:
        right = cls.moving_average(right_foot_positions)
        left = cls.moving_average(left_foot_positions)

        right_vel = cls.calc_velocity(right, 1, 1)
        left_vel = cls.calc_velocity(left, 1, 1)

        sequence: StepSequence = []
        right_vel_thresh = right_vel.mean() - right_vel.std()
        left_vel_thresh = left_vel.mean() - left_vel.std()
        is_right_moving = False
        is_left_moving = False

        for i in range(len(right_vel)):
            # Append if to the foot is moving
            if is_right_moving and right_vel[i] > right_vel_thresh:
                sequence.append("R")

            if is_left_moving and left_vel[i] > left_vel_thresh:
                sequence.append("L")

            is_right_moving = right_vel[i] > right_vel_thresh
            is_left_moving = left_vel[i] > left_vel_thresh

        return sequence

    @classmethod
    def find_analysis_window(
        cls,
        *,
        skill: Skill,
        hand_positions: list[Coordinate] | None,
        elbow_positions: list[Coordinate] | None,
    ) -> tuple[int, int, int]:
        if not any([hand_positions, elbow_positions]):
            cls.logger.error("At least one coordinate list should be provided")
            return -1, -1, -1

        match skill:
            case Skill.SERVE:
                if hand_positions and elbow_positions:
                    return cls.__find_serve_analysis_window(
                        hand_positions, elbow_positions
                    )
                cls.logger.error("At least one coordinate list should be provided")
                return -1, -1, -1

            case Skill.LIFT:
                if hand_positions and elbow_positions:
                    return cls.__find_lift_analysis_window(
                        hand_positions, elbow_positions
                    )
                cls.logger.error("At least one coordinate list should be provided")
                return -1, -1, -1

            case Skill.CLEAR | Skill.SMASH:
                if hand_positions and elbow_positions:
                    return cls.__find_smash_analysis_window(
                        hand_positions, elbow_positions
                    )
                cls.logger.error("At least one coordinate list should be provided")
                return -1, -1, -1

            case Skill.FOOTWORK:
                return -1, -1, -1

        return -1, -1, -1

    @staticmethod
    def mirror_angles(angles: AngleDict) -> AngleDict:
        """Swap Left↔Right labels to normalize a left-handed player's data to right-handed frame."""
        mapping = {
            "Left Elbow Angle": "Right Elbow Angle",
            "Right Elbow Angle": "Left Elbow Angle",
            "Left Knee Angle": "Right Knee Angle",
            "Right Knee Angle": "Left Knee Angle",
            "Left Shoulder Angle": "Right Shoulder Angle",
            "Right Shoulder Angle": "Left Shoulder Angle",
            "Left Crotch Angle": "Right Crotch Angle",
            "Right Crotch Angle": "Left Crotch Angle",
            "Nose Right Shoulder Elbow Angle": "Nose Left Shoulder Elbow Angle",
            "Nose Left Shoulder Elbow Angle": "Nose Right Shoulder Elbow Angle",
        }
        return {mapping.get(k, k): v for k, v in angles.items()}

    @staticmethod
    def compute_angles(
        landmark: CoordinateDict,
    ) -> AngleDict:
        angles: dict[str, float] = {key: 0.0 for key in JOINTS.keys()}
        for joint_name, (point_a_id, point_b_id, point_c_id) in JOINTS.items():
            if all(kp in landmark for kp in (point_a_id, point_b_id, point_c_id)):
                point_a = landmark[point_a_id]
                point_b = landmark[point_b_id]
                point_c = landmark[point_c_id]
                angle = PoseDetector.compute_angle(point_a, point_b, point_c)
                if angle is not None and isinstance(angle, float):
                    angles[joint_name] = angle
        return angles
