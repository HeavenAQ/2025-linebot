from Types import COCOKeypoints

JOINTS = {
    "Left Elbow": (
        COCOKeypoints.LEFT_SHOULDER,
        COCOKeypoints.LEFT_ELBOW,
        COCOKeypoints.LEFT_WRIST,
    ),
    "Right Elbow": (
        COCOKeypoints.RIGHT_SHOULDER,
        COCOKeypoints.RIGHT_ELBOW,
        COCOKeypoints.RIGHT_WRIST,
    ),
    "Left Knee": (
        COCOKeypoints.LEFT_HIP,
        COCOKeypoints.LEFT_KNEE,
        COCOKeypoints.LEFT_ANKLE,
    ),
    "Right Knee": (
        COCOKeypoints.RIGHT_HIP,
        COCOKeypoints.RIGHT_KNEE,
        COCOKeypoints.RIGHT_ANKLE,
    ),
    "Left Shoulder": (
        COCOKeypoints.LEFT_HIP,
        COCOKeypoints.LEFT_SHOULDER,
        COCOKeypoints.LEFT_ELBOW,
    ),
    "Right Shoulder": (
        COCOKeypoints.RIGHT_HIP,
        COCOKeypoints.RIGHT_SHOULDER,
        COCOKeypoints.RIGHT_ELBOW,
    ),
    "Left Crotch": (
        COCOKeypoints.LEFT_KNEE,
        COCOKeypoints.LEFT_HIP,
        COCOKeypoints.RIGHT_HIP,
    ),
    "Right Crotch": (
        COCOKeypoints.RIGHT_KNEE,
        COCOKeypoints.RIGHT_HIP,
        COCOKeypoints.LEFT_HIP,
    ),
    "Nose Right Shoulder Elbow": (
        COCOKeypoints.NOSE,
        COCOKeypoints.RIGHT_SHOULDER,
        COCOKeypoints.RIGHT_ELBOW,
    ),
    "Nose Left Shoulder Elbow": (
        COCOKeypoints.NOSE,
        COCOKeypoints.LEFT_SHOULDER,
        COCOKeypoints.LEFT_ELBOW,
    ),
}

SKELETON_CONNECTIONS = [
    (COCOKeypoints.LEFT_SHOULDER, COCOKeypoints.RIGHT_SHOULDER),
    (COCOKeypoints.LEFT_SHOULDER, COCOKeypoints.LEFT_ELBOW),
    (COCOKeypoints.LEFT_ELBOW, COCOKeypoints.LEFT_WRIST),
    (COCOKeypoints.RIGHT_SHOULDER, COCOKeypoints.RIGHT_ELBOW),
    (COCOKeypoints.RIGHT_ELBOW, COCOKeypoints.RIGHT_WRIST),
    (COCOKeypoints.LEFT_HIP, COCOKeypoints.RIGHT_HIP),
    (COCOKeypoints.LEFT_HIP, COCOKeypoints.LEFT_KNEE),
    (COCOKeypoints.LEFT_KNEE, COCOKeypoints.LEFT_ANKLE),
    (COCOKeypoints.RIGHT_HIP, COCOKeypoints.RIGHT_KNEE),
    (COCOKeypoints.RIGHT_KNEE, COCOKeypoints.RIGHT_ANKLE),
    (COCOKeypoints.LEFT_SHOULDER, COCOKeypoints.LEFT_HIP),
    (COCOKeypoints.RIGHT_SHOULDER, COCOKeypoints.RIGHT_HIP),
]
