from enum import IntEnum
from typing import Dict, Tuple

from pydantic import BaseModel


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
    SERVE = 0
    CLEAR = 1

    @classmethod
    def convert_to_enum(cls, skill: str):
        return Skill[skill.upper()]

    def __str__(self):
        return self.name.lower()


class Handedness(IntEnum):
    RIGHT = 0
    LEFT = 1

    @classmethod
    def convert_to_enum(cls, handedness: str):
        return Handedness[handedness.upper()]

    def __str__(self):
        return self.name.lower()


Body2DCoordinates = Dict[COCOKeypoints, Tuple[float, float]]


# Response related models (Pydantic)


class GradingDetail(BaseModel):
    description: str
    grade: float


class GradingOutcome(BaseModel):
    total_grade: float
    grading_details: list[GradingDetail]


class VideoAnalysisResponse(BaseModel):
    grade: GradingOutcome
    used_angles_data: list[dict[str, float] | None]
    processed_video: str
