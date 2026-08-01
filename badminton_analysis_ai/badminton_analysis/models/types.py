from enum import IntEnum, auto
from typing import Literal, NotRequired, TypedDict, TypeAlias

from typing_extensions import override

import numpy as np
from numpy.typing import NDArray


class COCOKeypoints(IntEnum):
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


class Skill(IntEnum):
    SERVE = auto()
    CLEAR = auto()
    SMASH = auto()
    LIFT = auto()
    BACKHAND_DRIVE = auto()
    FOREHAND_DRIVE = auto()
    BACKHAND_NETKILL = auto()
    FOREHAND_NETKILL = auto()
    FOOTWORK = auto()

    @classmethod
    def convert_to_enum(cls, skill: str) -> "Skill":
        return Skill[skill.upper()]

    @override
    def __str__(self) -> str:
        return self.name.lower()


class Handedness(IntEnum):
    RIGHT = 0
    LEFT = 1

    @classmethod
    def convert_to_enum(cls, handedness: str) -> "Handedness":
        return Handedness[handedness.upper()]

    @override
    def __str__(self) -> str:
        return self.name.lower()


Coordinate2D: TypeAlias = NDArray[np.float64]  # shape (2,)
Coordinate3D: TypeAlias = NDArray[np.float64]  # shape (3,)
Coordinate: TypeAlias = NDArray[np.float64]  # shape (D,)
Coordinates: TypeAlias = NDArray[np.float64]  # shape (N, D)
CoordinateDict: TypeAlias = dict[COCOKeypoints, Coordinate3D]
Coordinate2DDict: TypeAlias = dict[COCOKeypoints, Coordinate2D]
WholeBodyCoordinateDict: TypeAlias = dict[int, Coordinate2D]
AngleDict: TypeAlias = dict[str, float]


class GradingDetail(TypedDict):
    description: str
    grade: float


class GradingOutcome(TypedDict):
    total_grade: float
    grading_details: list[GradingDetail]


class TrackingData(TypedDict):
    frames: list[NDArray[np.uint8]]
    original_landmarks: list[CoordinateDict]
    body_landmarks_2d: NotRequired[list[Coordinate2DDict]]
    hand_positions: list[Coordinate2D]
    elbow_positions: list[Coordinate2D]
    time_intervals: list[float]
    source_frame_indices: NotRequired[list[int]]
    wholebody_landmarks: NotRequired[list[WholeBodyCoordinateDict]]


StepSequence: TypeAlias = list[Literal["L", "R"]]
