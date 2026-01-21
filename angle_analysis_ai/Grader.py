from abc import ABC, abstractmethod
from tracemalloc import start
from typing import override
import pandas as pd
from Types import (
    Body2DCoordinates,
    COCOKeypoints,
    GradingDetail,
    GradingOutcome,
    Handedness,
    Skill,
)

# Types
AngleDict = dict[str, float] | None
AngleDicts = list[AngleDict]

# Expert data
serve_mean = pd.read_excel(
    "./stats/serve/expert angle stats.xlsx", sheet_name="mean"
).set_index("Unnamed: 0")
serve_std = pd.read_excel(
    "./stats/serve/expert angle stats.xlsx", sheet_name="std"
).set_index("Unnamed: 0")


def serve_angle_grader(
    angle_max_grade: float,
    joint_name: str,
    frame_idx: str,
    angle_dict: dict[str, float],
) -> float:
    # Use joint name and frame index to get the mean and std from the expert data
    idx = joint_name, frame_idx
    mean = serve_mean.loc[idx]
    std = serve_std.loc[idx]

    # Calculate the min and max angle based on the mean and std
    min_angle = mean - std
    max_angle = mean + std

    # get current angle
    current_angle = angle_dict[joint_name]

    if min_angle <= current_angle <= max_angle:
        return angle_max_grade
    else:
        if min_angle > current_angle:
            return angle_max_grade * (current_angle / min_angle)
        else:
            return angle_max_grade * (max_angle / current_angle)


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
        return grader_class()


class ServeRightHandedGrader(Grader):
    def grade_checkpoint_1_arms(self, angle_dict: AngleDict) -> float:
        """
        The preparation phase of the serve. Full score for this checkpoint: 10
        """
        if not angle_dict:
            return 0

        grade = 0
        if angle_dict["Right Shoulder"] >= 25:
            grade += serve_angle_grader(5, "Right Shoulder", "check1", angle_dict)

        if angle_dict["Left Shoulder"] >= 25:
            grade += serve_angle_grader(5, "Left Shoulder", "check1", angle_dict)
        return grade

    def grade_checkpoint_1_legs(self, angle_dict: AngleDict) -> float:
        """
        The preparation phase of the serve. Full score for this checkpoint: 10
        """
        if not angle_dict:
            return 0
        if angle_dict["Right Crotch"] <= angle_dict["Left Crotch"]:
            return 10
        return 0

    def grade_checkpoint_2_lower_body(
        self, angles_start: AngleDict, angles_end: AngleDict
    ) -> float:
        """
        Lower Body weight transfer. Full score for this checkpoint: 10
        """
        if not angles_start or not angles_end:
            return 0

        grade = 0
        if (angles_start["Right Crotch"] - angles_end["Right Crotch"]) <= -3:
            grade += 5
        if (angles_start["Left Crotch"] - angles_end["Left Crotch"]) >= 3:
            grade += 5
        return grade

    def grade_checkpoint_2_upper_body(
        self, start_frame: Body2DCoordinates, end_frame: Body2DCoordinates
    ) -> float:
        """
        Upper Body weight transfer. Full score for this checkpoint: 10
        """
        # extract the coordinates needed for analysis
        start_left_shoulder = start_frame[COCOKeypoints.LEFT_SHOULDER][0]
        start_right_shoulder = start_frame[COCOKeypoints.RIGHT_SHOULDER][0]
        end_left_shoulder = end_frame[COCOKeypoints.LEFT_SHOULDER][0]
        end_right_shoulder = end_frame[COCOKeypoints.RIGHT_SHOULDER][0]

        # calculate the displacement between coordinates
        left_shoulder_disp = end_left_shoulder - start_left_shoulder
        right_shoulder_disp = end_right_shoulder - start_right_shoulder
        if left_shoulder_disp > 5 and right_shoulder_disp > 5:
            return 10
        else:
            return 0

    def grade_checkpoint_3(self, angle_dict: AngleDict) -> float:
        """
        Bottom rotation. Full score for this checkpoint: 20
        """
        grade = 0
        if not angle_dict:
            return grade
        if angle_dict["Right Crotch"] > angle_dict["Left Crotch"]:
            grade += 20
        return grade

    def grade_checkpoint_4(self, angle_dict: AngleDict) -> float:
        """
        Wrist flick. Full score for this checkpoint: 20
        """
        grade = 0
        if not angle_dict:
            return grade
        grade += serve_angle_grader(20, "Right Elbow", "check4", angle_dict)
        return grade

    def grade_checkpoint_5(self, angle: AngleDict) -> float:
        """
        Shoulder rotation. Full score for this checkpoint: 20
        """
        grade = 0
        if not angle:
            return grade
        grade += serve_angle_grader(10, "Right Shoulder", "check5", angle)
        grade += serve_angle_grader(10, "Nose Right Shoulder Elbow", "check5", angle)
        return grade

        # full score for this frame: 20

    @override
    def grade(
        self, angles: AngleDicts, landmark_list: list[Body2DCoordinates]
    ) -> GradingOutcome:
        # full score for this: 100
        check1_arms = self.grade_checkpoint_1_arms(angles[0])
        check1_legs = self.grade_checkpoint_1_legs(angles[0])
        check2_lower = self.grade_checkpoint_2_lower_body(angles[0], angles[1])
        check2_upper = self.grade_checkpoint_2_upper_body(
            landmark_list[1],
            landmark_list[3],
        )
        check3 = self.grade_checkpoint_3(angles[2])
        check4 = self.grade_checkpoint_4(angles[3])
        check5 = self.grade_checkpoint_5(angles[4])
        total = (
            check1_arms
            + check1_legs
            + check2_lower
            + check2_upper
            + check3
            + check4
            + check5
        )
        print(f"Total grade: {total}")
        grading_details: list[GradingDetail] = [
            GradingDetail(description="雙手平舉", grade=check1_arms),
            GradingDetail(description="將重心放至持拍腳", grade=check1_legs),
            GradingDetail(description="下盤重心轉移至非持拍腳", grade=check2_lower),
            GradingDetail(description="上肢重心轉移至非持拍腳", grade=check2_upper),
            GradingDetail(description="髖關節前旋", grade=check3),
            GradingDetail(description="持拍手手腕發力", grade=check4),
            GradingDetail(description="肩膀旋轉朝前", grade=check5),
        ]

        return GradingOutcome(
            grading_details=grading_details,
            total_grade=total,
        )


class ServeLeftHandedGrader(Grader):
    @override
    def grade(
        self, angles: AngleDicts, landmark_list: list[Body2DCoordinates]
    ) -> GradingOutcome:
        print(angles)
        return GradingOutcome(grading_details=[], total_grade=0)
        # Example grading logic for right-handed serve
        # score = 100 - abs(angles[1]["Left Shoulder"] - 90)
        # return max(0, score)


GraderRegistry.register(Skill.SERVE, Handedness.LEFT, ServeLeftHandedGrader)
GraderRegistry.register(Skill.SERVE, Handedness.RIGHT, ServeRightHandedGrader)
