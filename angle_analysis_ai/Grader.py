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
    angle_max_grade: float, joint_name: str, frame_idx: str, cur_angle: float
) -> float:
    idx = joint_name, frame_idx
    mean = serve_mean.loc[idx]
    std = serve_std.loc[idx]

    min_angle = mean - std
    max_angle = mean + std

    if min_angle <= cur_angle <= max_angle:
        return angle_max_grade
    else:
        if min_angle > cur_angle:
            return angle_max_grade * (cur_angle / min_angle)
        else:
            return angle_max_grade * (max_angle / cur_angle)


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
    def grade_checkpoint_1(self, angle_set: AngleDict) -> float:
        """
        The preparation phase of the serve. Full score for this checkpoint: 25
        """
        if not angle_set:
            return 0
        grade = 0

        # crotch angle
        if angle_set["Left Crotch"] >= angle_set["Right Crotch"]:
            grade += 10

        # shoulder angles
        grade += serve_angle_grader(
            5.5, "Left Shoulder", "start", angle_set["Left Shoulder"]
        )
        grade += serve_angle_grader(
            5.5, "Right Shoulder", "start", angle_set["Right Shoulder"]
        )

        # right elbow angle
        grade += serve_angle_grader(4, "Right Elbow", "start", angle_set["Right Elbow"])
        return grade

    def grade_checkpoint_2(self, angle_set: AngleDict) -> float:
        """
        Body weight transfer. Full score for this checkpoint: 25
        """
        if not angle_set:
            return 0
        grade = 0
        if angle_set["Left Crotch"] < angle_set["Right Crotch"]:
            grade += 10

        grade += serve_angle_grader(7.5, "Left Crotch", "mid", angle_set["Left Crotch"])

        grade += serve_angle_grader(
            7.5, "Right Crotch", "mid", angle_set["Right Crotch"]
        )

        return grade

    def grade_checkpoint_3(self, angle_set: AngleDict) -> float:
        """
        Wrist snap. Full score for this checkpoint: 25
        """
        if not angle_set:
            return 0
        return serve_angle_grader(25, "Right Elbow", "mid", angle_set["Right Elbow"])

    def grade_checkpoint_4(self, angle: AngleDict) -> float:
        """
        The ending pose of the serve. Full score for this checkpoint: 25
        """
        if not angle:
            return 0
        grade = 0
        if angle["Left Crotch"] > angle["Right Crotch"]:
            grade += 5
        grade += serve_angle_grader(2.5, "Left Crotch", "end", angle["Left Crotch"])
        grade += serve_angle_grader(2.5, "Right Crotch", "end", angle["Right Crotch"])
        grade += serve_angle_grader(5, "Right Shoulder", "end", angle["Right Shoulder"])
        grade += serve_angle_grader(5, "Right Elbow", "end", angle["Right Elbow"])
        return grade

        # full score for this frame: 20

    def grade(self, angles: AngleDicts) -> GradingOutcome:
        # full score for this: 100
        check1 = self.grade_checkpoint_1(angles[0])
        check2 = self.grade_checkpoint_2(angles[1])
        check3 = self.grade_checkpoint_3(angles[1])
        check4 = self.grade_checkpoint_4(angles[2])
        total = check1 + check2 + check3 + check4
        grading_details: list[GradingDetail] = [
            {"description": "Preparation", "grade": check1},
            {"description": "Body Weight Transfer", "grade": check2},
            {"description": "Wrist Snap", "grade": check3},
            {"description": "Ending Pose", "grade": check4},
        ]

        return {
            "grading_details": grading_details,
            "total_grade": total,
        }


class ServeLeftHandedGrader(Grader):
    def grade(self, angles: AngleDicts) -> GradingOutcome:
        print(angles)
        return {"grading_details": [], "total_grade": 0}
        # Example grading logic for right-handed serve
        # score = 100 - abs(angles[1]["Left Shoulder"] - 90)
        # return max(0, score)


GraderRegistry.register(Skill.SERVE, Handedness.LEFT, ServeLeftHandedGrader)
GraderRegistry.register(Skill.SERVE, Handedness.RIGHT, ServeRightHandedGrader)
