import numpy as np

from badminton_analysis.ml.expert_motion_preprocessing import _serve_swing_positions
from badminton_analysis.models.types import Handedness, Skill
from badminton_analysis.services.video_analyzer import VideoAnalyzer


def _serve_then_let_down(frames: int = 140) -> tuple[list, list]:
    """A fast swing through the bottom of its arc at frame 70, then the racket
    let down slowly and further, reaching its lowest point near frame 120."""
    shoulder = np.tile([300.0, 300.0], (frames, 1))
    hand = np.tile([360.0, 380.0], (frames, 1))
    swing = np.arange(60, 81)
    hand[swing, 0] = 360.0 + 18.0 * (swing - 60)
    hand[swing, 1] = 380.0 + 60.0 * np.sin(np.pi * (swing - 60) / 20)
    hand[81:, 0] = hand[80, 0]
    lowering = np.arange(100, 125)
    hand[lowering, 1] = 380.0 + 5.0 * (lowering - 100)
    hand[125:, 1] = hand[124, 1]
    return [tuple(p) for p in hand], [tuple(p) for p in shoulder]


def test_serve_contact_is_the_swing_not_the_racket_let_down_after_it() -> None:
    hand, shoulder = _serve_then_let_down()
    start, peak, end = VideoAnalyzer.find_analysis_window(
        skill=Skill.SERVE,
        hand_positions=hand,
        elbow_positions=shoulder,
        shoulder_positions=shoulder,
    )
    assert 60 <= peak <= 80
    assert start < peak < end


def test_a_poorly_seen_wrist_jump_is_bridged_before_the_swing_is_searched() -> None:
    hand, _ = _serve_then_let_down()
    skeleton = np.zeros((len(hand), 17, 2))
    skeleton[:, 6] = (300.0, 300.0)
    confidence = np.ones((len(hand), 17))
    glitched = list(hand)
    glitched[10] = (900.0, 900.0)
    confidence[10, 10] = 0.5
    bridged, shoulder = _serve_swing_positions(glitched, skeleton, confidence, Handedness.RIGHT)
    assert bridged[10] == hand[10]
    _, peak, _ = VideoAnalyzer.find_analysis_window(
        skill=Skill.SERVE,
        hand_positions=bridged,
        elbow_positions=shoulder,
        shoulder_positions=shoulder,
    )
    assert 60 <= peak <= 80
