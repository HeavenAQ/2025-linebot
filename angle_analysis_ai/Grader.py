from abc import ABC, abstractmethod
from math import atan2, degrees, exp, sqrt

from typing import override
import pandas as pd
from pathlib import Path
from Types import (
    Body2DCoordinates,
    COCOKeypoints,
    GradingDetail,
    GradingOutcome,
    Handedness,
    Skill,
)

# Types
AngleDict = dict[str, float]
AngleDicts = list[AngleDict]


class Grader(ABC):
    """
    Base class for all graders. Each grader should implement the `grade` method.
    """

    @abstractmethod
    def grade(
        self, angles: AngleDicts, landmark_list: list[Body2DCoordinates]
    ) -> GradingOutcome:
        """
        Abstract method to grade the performance based on angles.

        Args:
            angles (list[dict[str, float]]): list of angles for the frames to be graded.

        Returns:
            float: Grading score.
        """
        pass


class GraderRegistry:
    _registry = {}

    @classmethod
    def register(cls, skill: Skill, handedness: Handedness, grader_class: type):
        """
        Register a grader class for a specific skill and handedness.

        Args:
            skill (str): Badminton skill (e.g., 'serve', 'clear', 'smash').
            handedness (str): Handedness (e.g., 'left', 'right').
            grader_class (type): The grader class to register.
        """
        cls._registry[(skill, handedness)] = grader_class

    @classmethod
    def get(cls, skill: Skill, handedness: Handedness) -> Grader:
        """
        Retrieve the grader class for the given skill and handedness.

        Args:
            skill (str): Badminton skill.
            handedness (str): Handedness.

        Returns:
            Grader: An instance of the appropriate grader.
        """
        grader_class = cls._registry.get((skill, handedness))
        if not grader_class:
            raise ValueError(
                f"No grader registered for skill={skill}, handedness={handedness}"
            )
        # Prefer constructors that accept handedness; fallback to no-arg
        try:
            return grader_class(handedness)
        except TypeError:
            return grader_class()


class ServeGrader(Grader):
    _DATA_DIR = Path(__file__).resolve().parent / "stats" / "serve"
    serve_mean = pd.read_csv(_DATA_DIR / "mean1.csv").set_index("feature")
    serve_std = pd.read_csv(_DATA_DIR / "std1.csv").set_index("feature")
    serve_mean.columns = [0, 1, 2, 3, 4]
    serve_std.columns = [0, 1, 2, 3, 4]

    def __init__(self, handedness: Handedness) -> None:
        self.handedness = handedness

    # --- Handedness-aware key helpers ---
    @property
    def _dom_side(self) -> str:
        return "Right" if self.handedness == Handedness.RIGHT else "Left"

    @property
    def _non_dom_side(self) -> str:
        return "Left" if self.handedness == Handedness.RIGHT else "Right"

    @property
    def dominant_shoulder_key(self) -> str:
        return f"{self._dom_side} Shoulder Angle"

    @property
    def non_dominant_shoulder_key(self) -> str:
        return f"{self._non_dom_side} Shoulder Angle"

    @property
    def dominant_crotch_key(self) -> str:
        return f"{self._dom_side} Crotch Angle"

    @property
    def non_dominant_crotch_key(self) -> str:
        return f"{self._non_dom_side} Crotch Angle"

    @property
    def dominant_elbow_key(self) -> str:
        return f"{self._dom_side} Elbow Angle"

    @property
    def dominant_shoulder_elbow_key(self) -> str:
        return f"Nose {self._dom_side} Shoulder Elbow Angle"

    @classmethod
    def z_score(cls, value: float, mean: float, std: float):
        if std < 1e-6:
            return 0.0
        return (value - mean) / std

    @classmethod
    def angle_grader(
        cls,
        max_grade: float,
        joint_name: str,
        frame_idx: int,
        angles: AngleDicts,
    ) -> float:
        # expert stats
        idx = joint_name, frame_idx
        mean = cls.serve_mean.loc[idx]
        std = cls.serve_std.loc[idx]

        # Calculate the min and max angle based on the mean and std
        min_angle = mean - std
        max_angle = mean + std

        # get current angle
        current_angle = angles[frame_idx][joint_name]

        if min_angle <= current_angle <= max_angle:
            return max_grade
        else:
            if min_angle > current_angle:
                return max_grade * (current_angle / min_angle)
            else:
                return max_grade * (max_angle / current_angle)

    @classmethod
    def disp_grader(
        cls,
        max_grade: int,
        learner_disp: float,
        start_index: tuple[str, int],
        end_index: tuple[str, int],
    ):
        expert_mean_disp = (
            cls.serve_mean.loc[end_index] - cls.serve_mean.loc[start_index]
        )
        expert_std_disp = sqrt(
            cls.serve_std.loc[start_index] ** 2 + cls.serve_std.loc[end_index] ** 2
        )

        z = cls.z_score(
            learner_disp,
            expert_mean_disp,
            expert_std_disp,
        )

        if z >= 0:
            return max_grade
        return max_grade * exp(-0.5 * (z / 0.8) ** 2)

    def grade_checkpoint_1_arms(self, angles: AngleDicts, frame_idx: int) -> float:
        """
        The preparation phase of the serve. Full score for this checkpoint: 10
        """
        if not angles:
            return 0
        grade = 0.0
        if angles[frame_idx][self.dominant_shoulder_key] >= 25:
            grade += self.angle_grader(5, self.dominant_shoulder_key, frame_idx, angles)
        if angles[frame_idx][self.non_dominant_shoulder_key] >= 25:
            grade += self.angle_grader(
                5, self.non_dominant_shoulder_key, frame_idx, angles
            )
        return float(grade)

    def grade_checkpoint_1_legs(self, angles: AngleDicts, frame_idx: int) -> float:
        """
        The preparation phase of the serve. Full score for this checkpoint: 10
        """
        if not angles:
            return 0
        if (
            angles[frame_idx][self.dominant_crotch_key]
            <= angles[frame_idx][self.non_dominant_crotch_key]
        ):
            return 10
        return 0

    def grade_checkpoint_2_lower_body(
        self,
        angles: AngleDicts,
        start_frame: int,
        end_frame: int,
    ) -> float:
        """
        Lower Body weight transfer. Full score for this checkpoint: 20
        """
        if not angles:
            return 0

        dom_crotch_diff = (
            angles[end_frame][self.dominant_crotch_key]
            - angles[start_frame][self.dominant_crotch_key]
        )
        non_dom_crotch_diff = (
            angles[end_frame][self.non_dominant_crotch_key]
            - angles[start_frame][self.non_dominant_crotch_key]
        )

        grade = 0.0
        grade += self.disp_grader(
            10,
            dom_crotch_diff,
            (self.dominant_crotch_key, start_frame),
            (self.dominant_crotch_key, end_frame),
        )
        grade += self.disp_grader(
            10,
            non_dom_crotch_diff,
            (self.non_dominant_crotch_key, start_frame),
            (self.non_dominant_crotch_key, end_frame),
        )

        return grade

    def grade_checkpoint_2_upper_body(
        self, landmark_list: list[Body2DCoordinates], start_frame: int, end_frame: int
    ) -> float:
        """
        Upper Body weight transfer. Full score for this checkpoint: 20
        """
        # extract the coordinates needed for analysis
        part = (
            COCOKeypoints.RIGHT_EYE
            if self.handedness == Handedness.RIGHT
            else COCOKeypoints.LEFT_EYE
        )
        try:
            start_eye = landmark_list[start_frame][part][0]
            end_eye = landmark_list[end_frame][part][0]
        except KeyError:
            try:
                part = (
                    COCOKeypoints.RIGHT_EAR
                    if self.handedness == Handedness.RIGHT
                    else COCOKeypoints.LEFT_EAR
                )
                start_eye = landmark_list[start_frame][part][0]
                end_eye = landmark_list[end_frame][part][0]
            except KeyError:
                part = (
                    COCOKeypoints.LEFT_EYE
                    if self.handedness == Handedness.RIGHT
                    else COCOKeypoints.RIGHT_EYE
                )
                start_eye = landmark_list[start_frame][part][0]
                end_eye = landmark_list[end_frame][part][0]

        # calculate the displacement between coordinates
        learner_disp = end_eye - start_eye
        return self.disp_grader(
            20,
            learner_disp,
            (f"{part}_x", start_frame),
            (f"{part}_x", end_frame),
        )

    def _hip_angle(self, frame: Body2DCoordinates) -> float:
        lp = COCOKeypoints.LEFT_HIP
        rp = COCOKeypoints.RIGHT_HIP

        lx, ly = frame[lp]
        rx, ry = frame[rp]

        return degrees(atan2(ry - ly, rx - lx))

    def grade_checkpoint_3(
        self,
        landmark_list: list[Body2DCoordinates],
        start_frame: int,
        end_frame: int,
    ) -> float:
        """
        Bottom rotation using hip-axis angle: Full 20
        """

        # Learner rotation
        start_angle = self._hip_angle(landmark_list[start_frame])
        end_angle = self._hip_angle(landmark_list[end_frame])

        learner_rot = end_angle - start_angle
        learner_rot = (learner_rot + 180) % 360 - 180  # normalize to [-180, 180]

        # Expert mean rotation
        exp_start_lp = self.serve_mean.loc[f"{COCOKeypoints.LEFT_HIP}_x", start_frame]
        exp_start_lp_y = self.serve_mean.loc[f"{COCOKeypoints.LEFT_HIP}_y", start_frame]
        exp_start_rp = self.serve_mean.loc[f"{COCOKeypoints.RIGHT_HIP}_x", start_frame]
        exp_start_rp_y = self.serve_mean.loc[
            f"{COCOKeypoints.RIGHT_HIP}_y", start_frame
        ]

        exp_end_lp = self.serve_mean.loc[f"{COCOKeypoints.LEFT_HIP}_x", end_frame]
        exp_end_lp_y = self.serve_mean.loc[f"{COCOKeypoints.LEFT_HIP}_y", end_frame]
        exp_end_rp = self.serve_mean.loc[f"{COCOKeypoints.RIGHT_HIP}_x", end_frame]
        exp_end_rp_y = self.serve_mean.loc[f"{COCOKeypoints.RIGHT_HIP}_y", end_frame]

        exp_start_angle = degrees(
            atan2(exp_start_rp_y - exp_start_lp_y, exp_start_rp - exp_start_lp)
        )
        exp_end_angle = degrees(
            atan2(exp_end_rp_y - exp_end_lp_y, exp_end_rp - exp_end_lp)
        )

        expert_mean_rot = exp_end_angle - exp_start_angle
        expert_mean_rot = (expert_mean_rot + 180) % 360 - 180

        # Expert std (angle domain)
        stds = [
            self.serve_std.loc[f"{COCOKeypoints.LEFT_HIP}_x", start_frame],
            self.serve_std.loc[f"{COCOKeypoints.LEFT_HIP}_y", start_frame],
            self.serve_std.loc[f"{COCOKeypoints.RIGHT_HIP}_x", start_frame],
            self.serve_std.loc[f"{COCOKeypoints.RIGHT_HIP}_y", start_frame],
            self.serve_std.loc[f"{COCOKeypoints.LEFT_HIP}_x", end_frame],
            self.serve_std.loc[f"{COCOKeypoints.LEFT_HIP}_y", end_frame],
            self.serve_std.loc[f"{COCOKeypoints.RIGHT_HIP}_x", end_frame],
            self.serve_std.loc[f"{COCOKeypoints.RIGHT_HIP}_y", end_frame],
        ]

        # Soft pooled std (much smaller than full propagation)
        expert_std_rot = sqrt(sum(s**2 for s in stds) / len(stds))

        if expert_std_rot < 1e-6:
            return 0.0

        z = (learner_rot - expert_mean_rot) / expert_std_rot

        max_grade = 20
        if z >= -0.1:
            return max_grade

        return 0

    def grade_checkpoint_4(self, angles: AngleDicts, frame_idx: int) -> float:
        """
        Wrist flick. Full score for this checkpoint: 20
        """
        return self.angle_grader(20, self.dominant_elbow_key, frame_idx, angles)

    def grade_checkpoint_5(self, angles: AngleDicts, frame_idx: int) -> float:
        """
        Shoulder rotation. Full score for this checkpoint: 20
        """
        grade = 0.0
        grade += self.angle_grader(10, self.dominant_shoulder_key, frame_idx, angles)
        grade += self.angle_grader(
            10, self.dominant_shoulder_elbow_key, frame_idx, angles
        )
        return float(grade)

    @override
    def grade(
        self, angles: AngleDicts, landmark_list: list[Body2DCoordinates]
    ) -> GradingOutcome:
        # full score for this: 100
        check1_arms = self.grade_checkpoint_1_arms(angles, 0)
        check1_legs = self.grade_checkpoint_1_legs(angles, 0)

        check2_lower = self.grade_checkpoint_2_lower_body(
            angles,
            0,
            1,
        )
        check2_upper = self.grade_checkpoint_2_upper_body(
            landmark_list,
            1,
            3,
        )
        # if one fails, body weight transfer fails
        check2 = min(check2_lower, check2_upper)

        check3 = self.grade_checkpoint_3(
            landmark_list,
            0,
            4,
        )
        check4 = self.grade_checkpoint_4(angles, 3)
        check5 = self.grade_checkpoint_5(angles, 4)
        total = check1_arms + check1_legs + check2 + check3 + check4 + check5
        print(f"Total grade: {total}")
        grading_details: list[GradingDetail] = [
            GradingDetail(description="雙手平舉", grade=check1_arms),
            GradingDetail(description="將重心放至持拍腳", grade=check1_legs),
            GradingDetail(description="重心轉移至非持拍腳", grade=check2),
            GradingDetail(description="髖關節前旋", grade=check3),
            GradingDetail(description="持拍手手腕發力", grade=check4),
            GradingDetail(description="肩膀旋轉朝前", grade=check5),
        ]

        return GradingOutcome(
            grading_details=grading_details,
            total_grade=total,
        )


# Register both handedness for serve using a single handedness-aware grader
GraderRegistry.register(Skill.SERVE, Handedness.RIGHT, ServeGrader)
GraderRegistry.register(Skill.SERVE, Handedness.LEFT, ServeGrader)
