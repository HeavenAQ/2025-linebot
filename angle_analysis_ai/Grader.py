from abc import ABC, abstractmethod
import pandas as pd
from Types import GradingDetail, GradingOutcome, Handedness, Skill

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
    def grade(self, angles: AngleDicts) -> GradingOutcome:
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
        The preparation phase of the serve. Full score for this checkpoint: 20
        """
        if not angle_dict:
            return 0
        grade = 0
        grade += serve_angle_grader(5, "Right Shoulder", "check1", angle_dict)
        grade += serve_angle_grader(5, "Left Shoulder", "check1", angle_dict)
        return grade

    def grade_checkpoint_1_legs(self, angle_dict: AngleDict) -> float:
        """
        The preparation phase of the serve. Full score for this checkpoint: 20
        """
        if not angle_dict:
            return 0
        if angle_dict["Right Crotch"] <= angle_dict["Left Crotch"]:
            return 10
        return 0

    def grade_checkpoint_2(self, angle_dict1: AngleDict, angle_dict2) -> float:
        """
        Body weight transfer. Full score for this checkpoint: 20
        """
        if not angle_dict1 or not angle_dict2:
            return 0
        grade = 0
        if angle_dict1["Right Crotch"] < angle_dict2["Right Crotch"]:
            grade += 10
        if angle_dict1["Left Crotch"] > angle_dict2["Left Crotch"]:
            grade += 10
        return grade

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

    def grade(self, angles: AngleDicts) -> GradingOutcome:
        # full score for this: 100
        check1_arms = self.grade_checkpoint_1_arms(angles[0])
        check1_legs = self.grade_checkpoint_1_legs(angles[0])
        check2 = self.grade_checkpoint_2(angles[0], angles[1])
        check3 = self.grade_checkpoint_3(angles[2])
        check4 = self.grade_checkpoint_4(angles[3])
        check5 = self.grade_checkpoint_5(angles[4])
        total = check1_arms + check1_legs + check2 + check3 + check4 + check5
        grading_details: list[GradingDetail] = [
            GradingDetail(description="雙手平舉", grade=check1_arms),
            GradingDetail(description="將重心放至持拍腳", grade=check1_legs),
            GradingDetail(description="身體重心轉移至非持拍腳", grade=check2),
            GradingDetail(description="髖關節前旋", grade=check4),
            GradingDetail(description="持拍手手腕發力", grade=check4),
            GradingDetail(description="肩膀旋轉朝前", grade=check5),
        ]

        return GradingOutcome(
            grading_details=grading_details,
            total_grade=total,
        )


class ServeLeftHandedGrader(Grader):
    def grade(self, angles: AngleDicts) -> GradingOutcome:
        print(angles)
        return GradingOutcome(grading_details=[], total_grade=0)
        # Example grading logic for right-handed serve
        # score = 100 - abs(angles[1]["Left Shoulder"] - 90)
        # return max(0, score)


GraderRegistry.register(Skill.SERVE, Handedness.LEFT, ServeLeftHandedGrader)
GraderRegistry.register(Skill.SERVE, Handedness.RIGHT, ServeRightHandedGrader)
