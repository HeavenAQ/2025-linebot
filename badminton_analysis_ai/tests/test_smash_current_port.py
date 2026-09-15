from pathlib import Path
import json

import numpy as np
import pytest

from badminton_analysis.ml.smash_current_scoring import compose_points, MAXIMA
from badminton_analysis.ml.smash_current_runtime import canonical_source, sha256
from badminton_analysis.ml.smash_current_placement import (
    smooth_corrected_bbox_placement,
    transport_corrected_by_student_displacement,
)
from badminton_analysis.ml.skill_specs import get_skill_spec
from service.pipeline import _source_qualitative_phase_results
from badminton_analysis.models.types import Skill


def test_composition_has_no_original_score_floor_after_span_penalty():
    original = np.array([10, 10, 20, 20, 20, 20.0])
    result = compose_points(original, rotation=0, endpoint=20, span_gate=0)
    np.testing.assert_array_equal(result, [5, 0, 5, 20, 30, 0])
    np.testing.assert_array_equal(compose_points(original), MAXIMA)


@pytest.mark.parametrize("prior", [[100] * 6, [np.nan] * 6, [-1] * 6, [1] * 5])
def test_invalid_prior_rejected(prior):
    with pytest.raises(ValueError):
        compose_points(prior)


def test_left_canonicalization_swaps_labels_without_reflection():
    p = np.arange(4 * 17 * 2).reshape(4, 17, 2)
    c = np.ones((4, 17))
    left, _ = canonical_source(p, c, "left")
    np.testing.assert_array_equal(left[:, 10], p[:, 9])
    roundtrip, _ = canonical_source(left, c, "left")
    np.testing.assert_array_equal(roundtrip, p)


def test_bbox_ema_leaves_joint_vectors_intact_and_transport_happens_after():
    rng = np.random.default_rng(14)
    q = rng.normal(size=(30, 17, 2)).astype(np.float32)
    smoothed = smooth_corrected_bbox_placement(q)
    np.testing.assert_allclose(smoothed - smoothed[:, [11]], q - q[:, [11]], atol=5e-7)
    np.testing.assert_array_equal(smoothed[[0, -1]], q[[0, -1]])
    p = np.zeros_like(q)
    p[:, :, 0] = np.arange(30)[:, None] * 10
    moved = transport_corrected_by_student_displacement(smoothed, p, np.ones((30, 17)))
    np.testing.assert_allclose(
        moved[:, :, 0] - smoothed[:, :, 0],
        np.broadcast_to(np.arange(30)[:, None] * 10, (30, 17)),
        atol=2e-5,
    )


def test_frontend_uses_scored_source_checkpoints_not_full_clip_resampling():
    frames = dict(
        zip(
            (
                "preparation",
                "body_rotation",
                "arm_balance",
                "elbow_forward",
                "wrist_flick",
                "follow_through",
            ),
            [12, 30, 34, 51, 52, 110],
        )
    )
    phases = _source_qualitative_phase_results(
        get_skill_spec(Skill.SMASH),
        phase_indices=(0, 25, 50, 60, 63),
        source_phase_frames=[10, 35, 52, 75, 90],
        normalized_sequence_length=64,
        source_sequence_length=130,
        analysis_window_start_frame=0,
        analysis_window_end_frame=129,
        fps=30,
        checkpoint_source_frames=frames,
    )
    for phase in phases:
        assert phase.timestamp_seconds == pytest.approx(frames[phase.id] / 30)
        assert phase.normalized_position == pytest.approx(frames[phase.id] / 129)


def test_committed_current_artifact_hashes():
    root = (
        Path(__file__).resolve().parents[1]
        / "models/error_isolated_motion/smash/checkpoint_scorer_v1"
    )
    manifest = json.loads((root / "manifest.json").read_text())
    assert len(manifest) == 4
    for filename, expected in manifest.items():
        assert sha256(root / filename) == expected
    contract = json.loads((root / "calibration.json").read_text())
    assert contract["maxima"] == [5, 20, 5, 20, 30, 20]
    assert contract["generation"]["candidates"] == 8
    assert contract["generation"]["seed"] == 19


@pytest.mark.parametrize("selected_end", [5, 8])
def test_renderer_does_not_transform_scored_pixels(monkeypatch, tmp_path, selected_end):
    import service.renderer as renderer
    from badminton_analysis.models.types import Handedness

    written, drawn = [], []

    class Writer:
        def __init__(self, *args):
            pass

        @staticmethod
        def fourcc(*args):
            return 0

        def isOpened(self):
            return True

        def write(self, frame):
            written.append(frame)

        def release(self):
            pass

    def forbidden(*args, **kwargs):
        raise AssertionError("Renderer attempted a second fit, EMA, or timeline warp")

    monkeypatch.setattr(renderer.cv2, "VideoWriter", Writer)
    for name in (
        "_fit_affine",
        "_align_smash_contact_timeline",
        "_smooth_corrected_bbox_placement",
    ):
        monkeypatch.setattr(renderer, name, forbidden)
    monkeypatch.setattr(
        renderer, "_draw_skeleton", lambda frame, pose, *args: drawn.append(pose.copy())
    )
    monkeypatch.setattr(renderer, "_draw_header", lambda *args: None)
    monkeypatch.setattr(
        renderer, "_transcode_preserving_frame_rate", lambda *args: None
    )
    p = np.ones((9, 17, 2), np.float32)
    q = p + np.arange(9)[:, None, None]
    renderer.render_correction_video(
        tracking=dict(
            frames=[np.zeros((32, 32, 3), np.uint8) for _ in p],
            body_keypoints_2d=p,
            body_confidence_2d=np.ones((9, 17)),
            original_landmarks=[{}] * 9,
        ),
        original=p[:5],
        corrected=p[:5],
        confidence=np.ones((5, 17)),
        projected_corrected_pixels=q[: selected_end + 1],
        window=(0, 4, selected_end),
        handedness=Handedness.RIGHT,
        skill=Skill.SMASH,
        filename="test.mp4",
        score=50,
        output_path=tmp_path / "video.mp4",
        fps=30,
        fixed_hierarchical_placement=True,
    )
    assert len(written) == selected_end + 1
    np.testing.assert_array_equal(np.array(drawn[1::2]), q[: selected_end + 1])


def test_checkpoint_ranges_remain_on_cropped_unpaused_source_clock():
    spec = get_skill_spec(Skill.SMASH)
    frames = dict(zip((r.id for r in spec.rules), [12, 30, 34, 51, 52, 110]))
    evidence = {
        r.id: {"source_interval": [max(0, frames[r.id] - 5), frames[r.id]]}
        for r in spec.rules
    }
    evidence["follow_through"]["source_interval"] = [10, 110]
    phases = _source_qualitative_phase_results(
        spec,
        phase_indices=(0, 25, 50, 60, 63),
        source_phase_frames=[10, 35, 52, 75, 90],
        normalized_sequence_length=64,
        source_sequence_length=130,
        analysis_window_start_frame=0,
        analysis_window_end_frame=110,
        fps=30,
        checkpoint_source_frames=frames,
        checkpoint_evidence=evidence,
    )
    for marker in phases:
        expected = evidence[marker.id]["source_interval"]
        assert marker.start_seconds == pytest.approx(expected[0] / 30)
        assert marker.end_seconds == pytest.approx(expected[1] / 30)
    assert phases[-1].normalized_position == 1
    assert phases[-1].timestamp_seconds == pytest.approx(110 / 30)
