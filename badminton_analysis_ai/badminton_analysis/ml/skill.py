"""What every skill shares, and the hooks a skill overrides.

A skill is one stroke the system can grade. The pipeline is the same for all
of them: find the stroke in the clip, normalize it onto the generator's clock,
let the skill's EIMD prior generate the expert version, score the learner
against it, then render and coach. `SkillDefinition` holds that shared path;
a skill subclasses it and answers only what is particular to it -- where its
stroke starts and ends, how it is scored, and what the pipeline and the coach
need to say about it.

Adding a skill means: a `Skill` enum value and its rubric in `skill_specs.py`,
its EIMD model under `models/error_isolated_motion/<skill>/`, a
`ml/<skill>/skill.py` with a `SkillDefinition` subclass, and one line in
`_DEFINITIONS` below.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from badminton_analysis.ml.eimd.inference import (
    ErrorIsolatedMotionBundle,
    correct_student_motion_error_isolated,
)
from badminton_analysis.ml.motion.samples import (
    ExpertCorrection,
    ExpertPhaseModel,
    MotionSample,
)
from badminton_analysis.ml.motion.view import (
    align_expert_correction_to_ankle_spine_view,
)
from badminton_analysis.ml.skeleton_normalization import (
    estimate_foot_contacts,
    interpolate_pose_sequence,
    normalize_skeleton_motion,
    resample_detected_phase_indices,
    resample_sequence,
    tracking_body_arrays,
)
from badminton_analysis.ml.skeleton_scoring import (
    TORSO_WIDTH_BONES,
    project_stable_bone_lengths,
)
from badminton_analysis.ml.skill_specs import SkillCorrectionSpec, get_skill_spec
from badminton_analysis.models.types import Handedness, Skill, TrackingData
from badminton_analysis.services.video_analyzer import VideoAnalyzer

PhaseContract = Literal["current", "eimd_v3"]
PreparedSample = tuple[MotionSample, tuple[int, int, int], NDArray[np.int64]]

# Grading and display use the "current" window; generation priors and the
# requested-skill guard were trained or calibrated on the "eimd_v3" one.
PHASE_CONTRACTS: tuple[PhaseContract, ...] = ("current", "eimd_v3")


@dataclass(frozen=True)
class ScoringContext:
    """Everything a scorer may read about one upload."""

    tracking: TrackingData
    handedness: Handedness
    filename: str
    sample: MotionSample
    window: tuple[int, int, int]
    source_indices: NDArray[np.int64]
    correction: ExpertCorrection
    fps: float


@dataclass(frozen=True)
class ScoredMotion:
    """A scorer's result, in the shape the backend turns into a grade."""

    score: dict[str, Any]
    window: tuple[int, int, int]
    # The sample the grade was measured on, when it is not the generation one.
    scoring_sample: MotionSample
    # The correction to render: in the learner's view when the scorer put it there.
    correction: ExpertCorrection
    corrected_pixels: NDArray[np.float32] | None = None
    view_rotation: NDArray[np.float64] | None = None
    checkpoint_scorer_active: bool = False


class SkillScorer(ABC):
    """Grades a learner against the expert motion the skill's prior generated."""

    def __init__(
        self,
        definition: SkillDefinition,
        *,
        root: Path,
        model_path: Path,
        bundle: ErrorIsolatedMotionBundle,
        score_model: ExpertPhaseModel,
        candidates: int,
        seed: int,
        align_ankle_spine_view: bool,
        target_frames: int,
    ) -> None:
        self.definition = definition
        self.spec = definition.spec
        self.root = root
        self.model_path = model_path
        self.bundle = bundle
        self.score_model = score_model
        self.candidates = candidates
        self.seed = seed
        self.align_ankle_spine_view = align_ankle_spine_view
        self.target_frames = target_frames

    # The window the generation sample is cut with.
    generation_contract: PhaseContract = "eimd_v3"

    @abstractmethod
    def score(self, context: ScoringContext) -> ScoredMotion:
        """Grade one upload."""

    def generate(self, sample: MotionSample) -> ExpertCorrection:
        return correct_student_motion_error_isolated(
            self.bundle,
            sample,
            candidates=self.candidates,
            seed=self.seed,
            limit_wrist_velocity=self.definition.limit_generated_wrist_velocity,
        )

    def preparation_bounds(self, length: int) -> tuple[int, int]:
        preparation = next(
            window for window in self.spec.phase_windows if window.name == "preparation"
        )
        return preparation.bounds(length)

    def view_aligned(
        self, correction: ExpertCorrection
    ) -> tuple[ExpertCorrection, NDArray[np.float64] | None]:
        """The correction in the learner's ankle--spine view, when that is on."""
        if not self.align_ankle_spine_view:
            return correction, None
        start, end = self.preparation_bounds(len(correction.aligned_student_pose))
        return align_expert_correction_to_ankle_spine_view(
            correction, start=start, end=end
        )


class SkillDefinition:
    """One stroke: the shared pipeline, answered by the skill's own modules.

    Each skill package provides `phases`, `scorer` and `coaching` modules with
    the same interface; a subclass names its skill and those three modules.
    """

    skill: Skill
    phases: ModuleType
    scorer: ModuleType
    coaching: ModuleType

    @property
    def spec(self) -> SkillCorrectionSpec:
        return get_skill_spec(self.skill)

    # -- Finding the stroke (phases) ----------------------------------------

    @property
    def required_source_fps(self) -> float | None:
        return self.phases.REQUIRED_SOURCE_FPS

    def swing_positions(self, hand_positions, motion_skeleton, confidence, handedness):
        return self.phases.swing_positions(
            hand_positions, motion_skeleton, confidence, handedness
        )

    def phase_window(self, contract: PhaseContract, phases, **context):
        return getattr(self.phases, f"{contract}_phases")(phases, **context)

    def refine_phase_indices(self, pose, phase_indices, source):
        return self.phases.refine_phase_indices(pose, phase_indices, source)

    def alignment_contract(self, contract: PhaseContract) -> str:
        return self.phases.alignment_contract(contract)

    def prepare_sample(
        self,
        tracking: TrackingData,
        handedness: Handedness,
        filename: str,
        *,
        target_frames: int = 64,
        phase_contract: PhaseContract = "current",
    ) -> PreparedSample:
        """Apply the same 2D extraction contract used by the frozen generator."""
        body_2d = tracking.get("body_landmarks_2d")
        if not body_2d or len(body_2d) < 5:
            raise ValueError("at least five aligned 2D poses are required")
        full_skeleton, full_confidence = tracking_body_arrays(tracking)
        motion_skeleton, _ = interpolate_pose_sequence(full_skeleton, full_confidence)
        hand_positions = tracking.get("hand_positions")
        hand_positions, shoulder_positions = self.swing_positions(
            hand_positions, motion_skeleton, full_confidence, handedness
        )
        phases = VideoAnalyzer.find_analysis_phases(
            skill=self.skill,
            hand_positions=hand_positions,
            elbow_positions=tracking.get("elbow_positions"),
            shoulder_positions=shoulder_positions,
        )
        if phase_contract not in PHASE_CONTRACTS:
            raise ValueError(f"unsupported phase contract: {phase_contract}")
        phases, phase_source = self.phase_window(
            phase_contract,
            phases,
            tracking=tracking,
            full_skeleton=full_skeleton,
            full_confidence=full_confidence,
            motion_skeleton=motion_skeleton,
            handedness=handedness,
        )
        if any(second <= first for first, second in zip(phases, phases[1:])):
            raise ValueError("analysis phases must be strictly increasing")
        start, peak, end = int(phases[0]), int(phases[2]), int(phases[-1])
        if start < 0 or end >= len(full_skeleton) or end - start < 4:
            raise ValueError(f"invalid analysis window: {(start, peak, end)}")

        normalized = normalize_skeleton_motion(
            full_skeleton[start : end + 1],
            full_confidence[start : end + 1],
            handedness,
        )
        pose = resample_sequence(normalized.skeleton, target_frames)
        confidence = np.clip(
            resample_sequence(normalized.confidence, target_frames), 0.0, 1.0
        )
        pose = project_stable_bone_lengths(
            pose,
            pose,
            confidence,
            expert_length_bones=TORSO_WIDTH_BONES,
        )
        root = resample_sequence(normalized.root_trajectory, target_frames)
        contacts = estimate_foot_contacts(pose, root, confidence)
        phase_indices = resample_detected_phase_indices(phases, target_frames)
        phase_indices, phase_source = self.refine_phase_indices(
            pose, phase_indices, phase_source
        )

        source_indices = np.rint(np.linspace(start, end, target_frames)).astype(np.int64)
        sample = MotionSample(
            path=Path(filename),
            pose=pose.astype(np.float32),
            confidence=confidence.astype(np.float32),
            root=root.astype(np.float32),
            foot_contacts=contacts.astype(np.float32),
            phase_indices=phase_indices,
            handedness=str(handedness),
            skill=str(self.skill),
            video_name=filename,
            subject_id="inference",
            phase_source=phase_source,
            alignment_contract=self.alignment_contract(phase_contract),
            identity_level="inference_only",
        )
        return sample, (start, peak, end), source_indices

    # -- Scoring (scorer) ----------------------------------------------------

    @property
    def limit_generated_wrist_velocity(self) -> bool:
        return self.scorer.LIMIT_GENERATED_WRIST_VELOCITY

    def create_scorer(self, **config: Any) -> SkillScorer:
        return self.scorer.create_scorer(self, **config)

    # -- What the coach reads (coaching) --------------------------------------

    @property
    def owns_checkpoint_evidence(self) -> bool:
        return self.coaching.OWNS_CHECKPOINT_EVIDENCE

    @property
    def coaching_instructions(self) -> str:
        return self.coaching.coaching_instructions()

    @property
    def score_warning_zh_tw(self) -> str:
        return self.coaching.score_warning()

    def describe_scoring(self, diagnostics, status, method):
        return self.coaching.describe_scoring(diagnostics, status, method)

    def attach_grade_context(self, correction_grade, score) -> None:
        self.coaching.attach_grade_context(correction_grade, score)

    def feedback_phase(self, frame_index: int, anchor_1: int, anchor_2: int) -> str:
        return self.coaching.feedback_phase(frame_index, anchor_1, anchor_2)

    def comparison_frames(self, rule, samples, anchors) -> list[int]:
        return self.coaching.comparison_frames(rule, samples, anchors)


@cache
def _registry() -> dict[Skill, SkillDefinition]:
    from badminton_analysis.ml.serve.skill import ServeSkill
    from badminton_analysis.ml.smash.skill import SmashSkill

    return {definition.skill: definition for definition in (ServeSkill(), SmashSkill())}


def supported_skills() -> tuple[Skill, ...]:
    return tuple(_registry())


def supported_skills_text() -> str:
    return " and ".join(str(skill) for skill in supported_skills())


def skill_definition(skill: Skill) -> SkillDefinition:
    try:
        return _registry()[skill]
    except KeyError:
        raise ValueError(
            f"expert-motion generation currently supports {supported_skills_text()}"
        ) from None


def prepare_expert_motion_sample(
    tracking: TrackingData,
    handedness: Handedness,
    skill: Skill,
    filename: str,
    *,
    target_frames: int = 64,
    phase_contract: PhaseContract = "current",
) -> PreparedSample:
    """Apply the same 2D extraction contract used by the frozen generator."""
    return skill_definition(skill).prepare_sample(
        tracking,
        handedness,
        filename,
        target_frames=target_frames,
        phase_contract=phase_contract,
    )
