from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from badminton_analysis.models.types import Skill


SUPPORTED_CORRECTION_SKILLS = (
    Skill.SERVE,
    Skill.LIFT,
    Skill.CLEAR,
    Skill.SMASH,
)

CANONICAL_JOINTS = {
    0: "head",
    5: "non_dominant_shoulder",
    6: "dominant_shoulder",
    7: "non_dominant_elbow",
    8: "dominant_elbow",
    9: "non_dominant_wrist",
    10: "dominant_wrist",
    11: "non_dominant_hip",
    12: "dominant_hip",
    13: "non_dominant_knee",
    14: "dominant_knee",
    15: "non_dominant_ankle",
    16: "dominant_ankle",
}

CANONICAL_JOINTS_ZH_TW = {
    0: "頭部",
    5: "非慣用側肩膀",
    6: "慣用側肩膀",
    7: "非慣用側手肘",
    8: "慣用側手肘",
    9: "非慣用側手腕",
    10: "慣用側手腕",
    11: "非慣用側髖部",
    12: "慣用側髖部",
    13: "非慣用側膝蓋",
    14: "慣用側膝蓋",
    15: "非慣用側腳踝",
    16: "慣用側腳踝",
}


@dataclass(frozen=True)
class CorrectionDetailSpec:
    description: str
    name_zh_tw: str
    maximum: float
    start: int
    end: int
    joints: tuple[int, ...] | None = None


@dataclass(frozen=True)
class PhaseWindowSpec:
    name: str
    start: int
    end: int


@dataclass(frozen=True)
class FeedbackRuleSpec:
    id: str
    name_zh_tw: str
    phase: str
    maximum: float
    calculation_zh_tw: str
    measured_joints: tuple[int, ...]
    coaching_joints: tuple[int, ...]
    allowed_anchor_indices: tuple[int, ...]

    def as_prompt_dict(self) -> dict[str, str | float | list[int]]:
        return {
            "id": self.id,
            "name_zh_tw": self.name_zh_tw,
            "phase": self.phase,
            "maximum": self.maximum,
            "calculation_zh_tw": self.calculation_zh_tw,
            "measured_joint_ids": list(self.measured_joints),
            "coaching_joint_ids": list(self.coaching_joints),
        }


@dataclass(frozen=True)
class SkillCorrectionSpec:
    skill: Skill
    name_zh_tw: str
    description_zh_tw: str
    checkpoint_roles_zh_tw: tuple[str, str, str, str, str]
    joint_weights: tuple[float, ...]
    details: tuple[CorrectionDetailSpec, ...]
    phase_windows: tuple[PhaseWindowSpec, ...]
    rules: tuple[FeedbackRuleSpec, ...]
    model_version: str = "v1"

    def __post_init__(self) -> None:
        if len(self.joint_weights) != 17:
            raise ValueError(f"{self.skill} must define 17 joint weights")
        if len(self.checkpoint_roles_zh_tw) != 5:
            raise ValueError(f"{self.skill} must define five checkpoint roles")
        if abs(sum(rule.maximum for rule in self.rules) - 100.0) > 1e-8:
            raise ValueError(f"{self.skill} rule maxima must total 100")
        if abs(sum(detail.maximum for detail in self.details) - 100.0) > 1e-8:
            raise ValueError(f"{self.skill} detail maxima must total 100")
        if tuple(rule.name_zh_tw for rule in self.rules) != tuple(
            detail.name_zh_tw for detail in self.details
        ):
            raise ValueError(f"{self.skill} rules and correction details are misaligned")

    @property
    def slug(self) -> str:
        return str(self.skill)

    @property
    def model_stem(self) -> str:
        return f"{self.slug}_expert_guided_{self.model_version}"

    @property
    def dataset_root(self) -> Path:
        return Path("datasets/skeleton_sequences") / self.slug

    @property
    def model_path(self) -> Path:
        return Path("models/skeleton_correction") / f"{self.model_stem}.pt"

    @property
    def training_metrics_dir(self) -> Path:
        return Path("stats/skeleton_correction") / f"{self.model_stem}_training"

    @property
    def grading_output_dir(self) -> Path:
        return Path("stats/skeleton_correction") / f"{self.model_stem}_grades"

    @property
    def joint_weights_array(self) -> NDArray[np.float64]:
        return np.asarray(self.joint_weights, dtype=np.float64)

    def rule(self, rule_id: str) -> FeedbackRuleSpec:
        for rule in self.rules:
            if rule.id == rule_id:
                return rule
        raise KeyError(f"unknown {self.slug} feedback rule: {rule_id}")


_CLEAR_RULES = (
    FeedbackRuleSpec(
        "preparation",
        "球拍舉至腰部預備",
        "preparation",
        10,
        "第0關鍵幀：比較雙側肩膀、手肘與手腕相對專家動作所需的修正幅度。",
        (5, 6, 7, 8, 9, 10),
        (6, 8, 10),
        (0,),
    ),
    FeedbackRuleSpec(
        "body_rotation",
        "轉身",
        "rotation",
        10,
        "第0至第1關鍵幀：比較雙肩、雙髖與下肢旋轉相對專家動作所需的修正幅度。",
        (5, 6, 11, 12, 13, 14, 15, 16),
        (11, 12),
        (1,),
    ),
    FeedbackRuleSpec(
        "arm_balance",
        "雙手手肘平衡",
        "rotation",
        20,
        "第1關鍵幀：比較雙側肩膀、手肘與手腕相對專家平衡姿勢所需的修正幅度。",
        (5, 6, 7, 8, 9, 10),
        (7, 8),
        (1,),
    ),
    FeedbackRuleSpec(
        "elbow_forward",
        "手肘往前轉至前方",
        "contact",
        20,
        "第2關鍵幀附近：比較慣用側肩膀、手肘與手腕相對專家擊球姿勢所需的修正幅度。",
        (0, 6, 8, 10),
        (6, 8),
        (2,),
    ),
    FeedbackRuleSpec(
        "wrist_flick",
        "手腕發力",
        "contact",
        20,
        "第2關鍵幀附近：比較慣用側手肘、手腕的位置、角度與速度修正幅度。",
        (6, 8, 10),
        (8, 10),
        (2,),
    ),
    FeedbackRuleSpec(
        "follow_through",
        "慣用手肩膀往前轉",
        "follow_through",
        20,
        "第3至第4關鍵幀：比較慣用側肩膀與軀幹隨揮相對專家動作所需的修正幅度。",
        (5, 6, 8, 10, 11, 12),
        (6,),
        (3, 4),
    ),
)


_SMASH_RULES = (
    FeedbackRuleSpec(
        "preparation",
        "球拍舉至腰部預備",
        "preparation",
        10,
        "第0關鍵幀：確認持拍手完成引拍準備，並比較雙臂相對專家姿勢的修正幅度。",
        (5, 6, 7, 8, 9, 10),
        (6, 8, 10),
        (0,),
    ),
    FeedbackRuleSpec(
        "body_rotation",
        "轉身",
        "rotation",
        10,
        "第0至第1關鍵幀：比較雙肩、雙髖與下肢蓄力旋轉相對專家殺球動作的修正幅度。",
        (5, 6, 11, 12, 13, 14, 15, 16),
        (11, 12),
        (1,),
    ),
    FeedbackRuleSpec(
        "arm_balance",
        "雙手手肘平衡",
        "rotation",
        20,
        "第1關鍵幀：比較非慣用手平衡與慣用手蓄力姿勢相對專家動作的修正幅度。",
        (5, 6, 7, 8, 9, 10),
        (7, 8),
        (1,),
    ),
    FeedbackRuleSpec(
        "elbow_forward",
        "手肘往前轉至前方",
        "contact",
        20,
        "第2關鍵幀附近：比較慣用側肩膀、手肘與手腕向前加速相對專家動作的修正幅度。",
        (0, 6, 8, 10),
        (6, 8),
        (2,),
    ),
    FeedbackRuleSpec(
        "wrist_flick",
        "手腕發力",
        "contact",
        20,
        "第2關鍵幀附近：比較慣用側前臂伸展與手腕加速相對專家殺球動作的修正幅度。",
        (6, 8, 10),
        (8, 10),
        (2,),
    ),
    FeedbackRuleSpec(
        "follow_through",
        "慣用手肩膀往前轉",
        "follow_through",
        20,
        "第3至第4關鍵幀：比較慣用側肩膀、軀幹前傾與隨揮相對專家動作的修正幅度。",
        (5, 6, 8, 10, 11, 12),
        (6,),
        (3, 4),
    ),
)


_SERVE_RULES = (
    FeedbackRuleSpec(
        "arms_raised",
        "雙手平舉",
        "preparation",
        10,
        "第1關鍵幀：比較雙側肩膀、手肘與手腕的準備高度相對專家發球姿勢所需的修正幅度。",
        (5, 6, 7, 8, 9, 10),
        (7, 8),
        (1,),
    ),
    FeedbackRuleSpec(
        "racket_foot_weight",
        "將重心放至持拍腳",
        "preparation",
        10,
        "第1關鍵幀：比較雙髖、雙膝與雙腳踝相對專家持拍腳承重姿勢所需的修正幅度。",
        (11, 12, 13, 14, 15, 16),
        (12, 14, 16),
        (1,),
    ),
    FeedbackRuleSpec(
        "weight_transfer",
        "重心轉移至非持拍腳",
        "weight_transfer",
        20,
        "第1至第3關鍵幀：比較骨盆、膝蓋與腳踝的重心轉移相對專家發球動作所需的修正幅度。",
        (5, 6, 11, 12, 13, 14, 15, 16),
        (11, 12, 15, 16),
        (2,),
    ),
    FeedbackRuleSpec(
        "hip_rotation",
        "髖關節前旋",
        "follow_through",
        20,
        "第1至第4關鍵幀：比較雙髖連線旋轉與軀幹帶動相對專家發球動作所需的修正幅度。",
        (5, 6, 11, 12),
        (11, 12),
        (4,),
    ),
    FeedbackRuleSpec(
        "wrist_flick",
        "持拍手手腕發力",
        "contact",
        20,
        "第3關鍵幀：比較慣用側手肘與手腕的位置、角度和速度相對專家擊球動作的修正幅度。",
        (6, 8, 10),
        (8, 10),
        (3,),
    ),
    FeedbackRuleSpec(
        "shoulder_rotation",
        "肩膀旋轉朝前",
        "follow_through",
        20,
        "第4關鍵幀：比較雙肩連線、慣用側肩膀與手肘隨揮相對專家動作所需的修正幅度。",
        (5, 6, 8, 10, 11, 12),
        (6,),
        (4,),
    ),
)


_LIFT_RULES = (
    FeedbackRuleSpec(
        "preparation",
        "手腕放置腰部放鬆預備",
        "preparation",
        10,
        "第0關鍵幀：比較雙側肩膀、手肘與手腕的放鬆準備姿勢相對專家動作所需的修正幅度。",
        (5, 6, 7, 8, 9, 10),
        (6, 8, 10),
        (0,),
    ),
    FeedbackRuleSpec(
        "backswing",
        "手腕往後引拍",
        "backswing",
        25,
        "第2關鍵幀：比較慣用側肩膀、手肘與手腕向後引拍相對專家挑球動作所需的修正幅度。",
        (6, 8, 10),
        (8, 10),
        (2,),
    ),
    FeedbackRuleSpec(
        "forward_extension",
        "手腕往前壓",
        "contact",
        35,
        "第3關鍵幀：比較慣用側肩膀、手肘與手腕向前加速相對專家挑球動作所需的修正幅度。",
        (6, 8, 10),
        (8, 10),
        (3,),
    ),
    FeedbackRuleSpec(
        "completion",
        "手腕放鬆回到預備姿勢",
        "follow_through",
        30,
        "第4關鍵幀：比較慣用側肩膀、手肘與手腕放鬆並回到預備姿勢相對專家動作所需的修正幅度。",
        (6, 8, 10),
        (6, 8, 10),
        (4,),
    ),
)


def _details(rules: tuple[FeedbackRuleSpec, ...], windows: tuple[tuple[int, int, tuple[int, ...] | None], ...]) -> tuple[CorrectionDetailSpec, ...]:
    return tuple(
        CorrectionDetailSpec(
            description=f"{rule.id.replace('_', ' ').title()} correction",
            name_zh_tw=rule.name_zh_tw,
            maximum=rule.maximum,
            start=start,
            end=end,
            joints=joints,
        )
        for rule, (start, end, joints) in zip(rules, windows, strict=True)
    )


_UPPER_BODY_WEIGHTS = (
    0.5, 0.25, 0.25, 0.25, 0.25,
    1.5, 2.0, 1.25, 3.0, 1.5, 4.0,
    1.5, 1.5, 1.25, 1.25, 1.25, 1.25,
)


SKILL_SPECS: dict[Skill, SkillCorrectionSpec] = {
    Skill.CLEAR: SkillCorrectionSpec(
        skill=Skill.CLEAR,
        name_zh_tw="高遠球",
        description_zh_tw="高遠球動作",
        checkpoint_roles_zh_tw=(
            "第0關鍵幀：準備動作與轉身起點",
            "第1關鍵幀：轉身終點與雙手平衡",
            "第2關鍵幀：手肘前轉與手腕發力",
            "第3關鍵幀：隨揮候選畫面",
            "第4關鍵幀：隨揮候選畫面與動作終點",
        ),
        joint_weights=_UPPER_BODY_WEIGHTS,
        details=_details(
            _CLEAR_RULES,
            (
                (0, 16, None),
                (8, 32, (5, 6, 11, 12, 13, 14, 15, 16)),
                (16, 40, (5, 7, 9, 6, 8, 10)),
                (27, 38, (6, 8, 10)),
                (24, 48, (6, 8, 10)),
                (40, 64, None),
            ),
        ),
        phase_windows=(
            PhaseWindowSpec("preparation", 0, 16),
            PhaseWindowSpec("rotation", 8, 32),
            PhaseWindowSpec("contact", 27, 38),
            PhaseWindowSpec("follow_through", 38, 64),
        ),
        rules=_CLEAR_RULES,
        model_version="v3",
    ),
    Skill.SMASH: SkillCorrectionSpec(
        skill=Skill.SMASH,
        name_zh_tw="殺球",
        description_zh_tw="殺球動作",
        checkpoint_roles_zh_tw=(
            "第0關鍵幀：引拍準備與轉身起點",
            "第1關鍵幀：蓄力轉身與雙手平衡",
            "第2關鍵幀：手肘前轉與前臂加速",
            "第3關鍵幀：軀幹前傾與隨揮",
            "第4關鍵幀：殺球動作終點",
        ),
        joint_weights=(
            0.5, 0.25, 0.25, 0.25, 0.25,
            1.5, 2.5, 1.25, 3.5, 1.25, 4.5,
            2.0, 2.0, 1.5, 1.5, 1.25, 1.25,
        ),
        details=_details(
            _SMASH_RULES,
            (
                (0, 16, None),
                (8, 32, (5, 6, 11, 12, 13, 14, 15, 16)),
                (16, 40, (5, 7, 9, 6, 8, 10)),
                (27, 38, (6, 8, 10)),
                (24, 48, (6, 8, 10)),
                (40, 64, (5, 6, 8, 10, 11, 12)),
            ),
        ),
        phase_windows=(
            PhaseWindowSpec("preparation", 0, 16),
            PhaseWindowSpec("rotation", 8, 32),
            PhaseWindowSpec("contact", 27, 38),
            PhaseWindowSpec("follow_through", 38, 64),
        ),
        rules=_SMASH_RULES,
    ),
    Skill.SERVE: SkillCorrectionSpec(
        skill=Skill.SERVE,
        name_zh_tw="發球",
        description_zh_tw="發球動作",
        checkpoint_roles_zh_tw=(
            "第0關鍵幀：發球準備起點",
            "第1關鍵幀：雙手平舉與持拍腳承重",
            "第2關鍵幀：重心轉移過渡",
            "第3關鍵幀：擊球與手腕發力",
            "第4關鍵幀：髖部及肩膀完成前旋",
        ),
        joint_weights=(
            0.5, 0.25, 0.25, 0.25, 0.25,
            1.75, 2.0, 1.5, 2.5, 1.5, 3.0,
            2.5, 2.5, 2.0, 2.0, 2.0, 2.0,
        ),
        details=_details(
            _SERVE_RULES,
            (
                (8, 32, (5, 6, 7, 8, 9, 10)),
                (8, 32, (11, 12, 13, 14, 15, 16)),
                (12, 52, (5, 6, 11, 12, 13, 14, 15, 16)),
                (16, 64, (5, 6, 11, 12)),
                (36, 56, (6, 8, 10)),
                (48, 64, (5, 6, 8, 10, 11, 12)),
            ),
        ),
        phase_windows=(
            PhaseWindowSpec("preparation", 0, 24),
            PhaseWindowSpec("weight_transfer", 16, 48),
            PhaseWindowSpec("contact", 36, 56),
            PhaseWindowSpec("follow_through", 48, 64),
        ),
        rules=_SERVE_RULES,
    ),
    Skill.LIFT: SkillCorrectionSpec(
        skill=Skill.LIFT,
        name_zh_tw="挑球",
        description_zh_tw="挑球動作",
        checkpoint_roles_zh_tw=(
            "第0關鍵幀：腰部放鬆準備",
            "第1關鍵幀：向後引拍過渡",
            "第2關鍵幀：向後引拍最低點",
            "第3關鍵幀：手腕往前壓",
            "第4關鍵幀：手腕放鬆回到預備姿勢",
        ),
        joint_weights=(
            0.5, 0.25, 0.25, 0.25, 0.25,
            1.25, 2.5, 1.0, 3.5, 1.0, 4.5,
            1.25, 1.5, 1.0, 1.25, 1.0, 1.25,
        ),
        details=_details(
            _LIFT_RULES,
            (
                (0, 20, (5, 6, 7, 8, 9, 10)),
                (16, 40, (6, 8, 10)),
                (32, 56, (6, 8, 10)),
                (44, 64, (5, 6, 7, 8, 9, 10)),
            ),
        ),
        phase_windows=(
            PhaseWindowSpec("preparation", 0, 20),
            PhaseWindowSpec("backswing", 12, 40),
            PhaseWindowSpec("contact", 36, 56),
            PhaseWindowSpec("follow_through", 48, 64),
        ),
        rules=_LIFT_RULES,
    ),
}


def get_skill_spec(skill: Skill | str) -> SkillCorrectionSpec:
    try:
        resolved = Skill.convert_to_enum(skill) if isinstance(skill, str) else skill
        return SKILL_SPECS[resolved]
    except KeyError as exc:
        supported = ", ".join(str(value) for value in SUPPORTED_CORRECTION_SKILLS)
        raise ValueError(
            f"skeleton correction does not support {skill}; choose one of: {supported}"
        ) from exc


def supported_skill_choices() -> tuple[str, ...]:
    return tuple(str(skill) for skill in SUPPORTED_CORRECTION_SKILLS)


def validate_checkpoint_spec(
    checkpoint: Mapping[str, Any], spec: SkillCorrectionSpec
) -> None:
    checkpoint_skill = str(checkpoint.get("skill", "clear"))
    if checkpoint_skill != spec.slug:
        raise ValueError(
            f"checkpoint skill is {checkpoint_skill}, but requested skill is "
            f"{spec.slug}"
        )
    if "joint_weights" not in checkpoint:
        if spec.skill == Skill.CLEAR:
            return
        raise ValueError(f"{spec.slug} checkpoint does not contain joint weights")
    checkpoint_weights = np.asarray(checkpoint["joint_weights"], dtype=np.float64)
    if checkpoint_weights.shape != (17,) or not np.allclose(
        checkpoint_weights, spec.joint_weights_array
    ):
        raise ValueError(
            f"{spec.slug} checkpoint joint weights do not match the current skill contract"
        )
