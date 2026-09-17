from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from badminton_analysis.models.types import Skill

SUPPORTED_CORRECTION_SKILLS = (
    Skill.SERVE,
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
    start_fraction: float
    end_fraction: float
    joints: tuple[int, ...] | None = None
    metric: str = "window_distance"

    def bounds(self, frame_count: int) -> tuple[int, int]:
        return motion_completion_bounds(
            frame_count, self.start_fraction, self.end_fraction
        )


@dataclass(frozen=True)
class PhaseWindowSpec:
    name: str
    start_fraction: float
    end_fraction: float

    def bounds(self, frame_count: int) -> tuple[int, int]:
        return motion_completion_bounds(
            frame_count, self.start_fraction, self.end_fraction
        )


def motion_completion_bounds(
    frame_count: int, start_fraction: float, end_fraction: float
) -> tuple[int, int]:
    """Resolve a fractional motion interval to an end-exclusive frame range."""
    if frame_count < 1:
        raise ValueError("motion must contain at least one frame")
    if not 0.0 <= start_fraction < end_fraction <= 1.0:
        raise ValueError("motion completion bounds must satisfy 0 <= start < end <= 1")
    start = int(np.floor(start_fraction * frame_count))
    end = int(np.ceil(end_fraction * frame_count))
    start = min(start, frame_count - 1)
    end = min(frame_count, max(start + 1, end))
    return start, end


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
    # Most criteria display inside their semantic phase. A transition may
    # instead culminate on a shared event boundary, such as serve weight
    # transfer at the contact anchor.
    display_phase: str | None = None

    def as_prompt_dict(self) -> dict[str, str | float | list[int]]:
        payload: dict[str, str | float | list[int]] = {
            "id": self.id,
            "name_zh_tw": self.name_zh_tw,
            "phase": self.phase,
            "maximum": self.maximum,
            "calculation_zh_tw": self.calculation_zh_tw,
            "measured_joint_ids": list(self.measured_joints),
            "coaching_joint_ids": list(self.coaching_joints),
        }
        if self.display_phase is not None:
            payload["display_phase"] = self.display_phase
        return payload


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
            raise ValueError(
                f"{self.skill} rules and correction details are misaligned"
            )

    @property
    def slug(self) -> str:
        return str(self.skill)

    @property
    def joint_weights_array(self) -> NDArray[np.float64]:
        return np.asarray(self.joint_weights, dtype=np.float64)

    def rule(self, rule_id: str) -> FeedbackRuleSpec:
        for rule in self.rules:
            if rule.id == rule_id:
                return rule
        raise KeyError(f"unknown {self.slug} feedback rule: {rule_id}")


_SMASH_RULES = (
    FeedbackRuleSpec(
        "preparation",
        "球拍舉至腰部預備",
        "preparation",
        5,
        "球拍舉至腰部預備：在預備評分區間將持拍手維持腰部附近。評分以手腕沿髖部至肩部軸的相對高度作為代理；手腕升到肩部附近會扣分，不可將高舉持拍手視為滿分預備。這不是直接偵測拍頭高度。",
        (5, 6, 7, 8, 9, 10),
        (6, 8, 10),
        (0,),
    ),
    FeedbackRuleSpec(
        "body_rotation",
        "轉身",
        "rotation",
        20,
        "比較起始到雙手平衡的轉身過程：慣用側髖－慣用側踝－非慣用側踝角、肩軸與肩相對髖軸的變化均需達到評分標準。不能只看最後姿勢相似；需檢查持拍側腿帶動與肩髖的實際轉動，並依提供的量測證據說明。",
        (5, 6, 11, 12, 13, 14, 15, 16),
        (11, 12),
        (1,),
    ),
    FeedbackRuleSpec(
        "arm_balance",
        "雙手手肘平衡",
        "rotation",
        5,
        "雙手手肘保持平衡：檢查整段指定雙手平衡區間，而非只看較晚的抬手姿勢。非慣用手已抬至肩附近時，若慣用手腕持續低於自身肩部超過專家容許程度，會受到扣分；稍後抬起不能抹除前面的不足。兩手可有合理高低差，不要求等高；左右必須依持拍手判斷。",
        (5, 6, 7, 8, 9, 10),
        (7, 8),
        (1,),
    ),
    FeedbackRuleSpec(
        "elbow_forward",
        "手肘往前轉至前方",
        "contact",
        20,
        "擊球前讓慣用手手肘往前轉到身體前方，再帶動前臂加速。",
        (0, 6, 8, 10),
        (6, 8),
        (2,),
    ),
    FeedbackRuleSpec(
        "wrist_flick",
        "手腕發力",
        "contact",
        30,
        "在評分指定的擊球加速關鍵幀及其鄰近區間，檢查慣用側肩、肘與腕的協調及揮拍通過擊球點的動態。單一腕關節點不能直接證明手腕屈曲、握拍力量或球拍速度，不可僅憑結尾姿勢判斷手腕發力。",
        (6, 8, 10),
        (8, 10),
        (2,),
    ),
    FeedbackRuleSpec(
        "follow_through",
        "慣用手肩膀往前轉",
        "follow_through",
        20,
        "使用評分器選定的最佳合格隨揮終點，與起始姿勢比較慣用側肩膀往前轉及肩寬縮短。肩寬幾乎不變時，即使原始隨揮分數很高仍可降至零。此項保留最佳終點，不採幀平均，也不因最佳終點之後肩部回退而額外扣分。",
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
        5,
        "發球準備時雙手平舉，雙臂自然抬起並保持穩定；持拍手與非持拍手可因功能不同呈現合理高低差，不要求雙手或雙肘等高。",
        (5, 6, 7, 8, 9, 10),
        (7, 8),
        (0,),
    ),
    FeedbackRuleSpec(
        "racket_foot_weight",
        "將重心放至持拍腳",
        "preparation",
        5,
        "動作開始時先將重心放在持拍腳，準備向前轉移。",
        (11, 12, 13, 14, 15, 16),
        (12, 14, 16),
        (1,),
    ),
    FeedbackRuleSpec(
        "weight_transfer",
        "重心轉移至非持拍腳",
        "weight_transfer",
        30,
        "揮拍過程將重心由持拍腳轉到非持拍腳，同時讓上半身順勢向前。",
        (5, 6, 11, 12, 13, 14, 15, 16),
        (5, 6, 11, 12, 15, 16),
        (2,),
        display_phase="contact",
    ),
    FeedbackRuleSpec(
        "hip_rotation",
        "髖關節前旋",
        "follow_through",
        10,
        "重心轉移時讓髖關節向前旋轉，帶動整個揮拍動作。",
        (5, 6, 11, 12),
        (11, 12),
        (4,),
    ),
    FeedbackRuleSpec(
        "wrist_flick",
        "持拍手手腕發力",
        "contact",
        30,
        "擊球瞬間用持拍手手腕發力，讓球拍快速向前。",
        (6, 8, 10),
        (8, 10),
        (2,),
    ),
    FeedbackRuleSpec(
        "shoulder_rotation",
        "肩膀旋轉朝前",
        "follow_through",
        20,
        "隨揮時讓慣用側肩膀旋轉朝前，持拍前臂順勢收向對側頸部附近，完成發球動作。",
        (5, 6, 8, 10, 11, 12),
        (6, 8, 10),
        (4,),
    ),
)


def _details(
    rules: tuple[FeedbackRuleSpec, ...],
    windows: tuple[tuple[float, float, tuple[int, ...] | None], ...],
) -> tuple[CorrectionDetailSpec, ...]:
    return tuple(
        CorrectionDetailSpec(
            description=f"{rule.id.replace('_', ' ').title()} correction",
            name_zh_tw=rule.name_zh_tw,
            maximum=rule.maximum,
            start_fraction=start,
            end_fraction=end,
            joints=joints,
            metric=(
                "serve_follow_through_cross_body"
                if rule.id == "shoulder_rotation"
                else "window_distance"
            ),
        )
        for rule, (start, end, joints) in zip(rules, windows, strict=True)
    )


SKILL_SPECS: dict[Skill, SkillCorrectionSpec] = {
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
            0.5,
            0.25,
            0.25,
            0.25,
            0.25,
            1.5,
            2.5,
            1.25,
            3.5,
            1.25,
            4.5,
            2.0,
            2.0,
            1.5,
            1.5,
            1.25,
            1.25,
        ),
        details=_details(
            _SMASH_RULES,
            (
                (0.0, 0.25, None),
                (0.125, 0.5, (5, 6, 11, 12, 13, 14, 15, 16)),
                (0.25, 0.625, (5, 7, 9, 6, 8, 10)),
                (0.421875, 0.59375, (6, 8, 10)),
                (0.375, 0.75, (6, 8, 10)),
                (0.625, 1.0, (5, 6, 8, 10, 11, 12)),
            ),
        ),
        phase_windows=(
            PhaseWindowSpec("preparation", 0.0, 0.25),
            PhaseWindowSpec("rotation", 0.125, 0.5),
            PhaseWindowSpec("contact", 0.421875, 0.59375),
            PhaseWindowSpec("follow_through", 0.59375, 1.0),
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
            "第2關鍵幀：重心轉移與最大手腕加速度（擊球事件）",
            "第3關鍵幀：擊球後隨揮",
            "第4關鍵幀：髖部及肩膀完成前旋",
        ),
        joint_weights=(
            0.5,
            0.25,
            0.25,
            0.25,
            0.25,
            1.75,
            2.0,
            1.5,
            2.5,
            1.5,
            3.0,
            2.5,
            2.5,
            2.0,
            2.0,
            2.0,
            2.0,
        ),
        details=_details(
            _SERVE_RULES,
            (
                (0.125, 0.5, (5, 6, 7, 8, 9, 10)),
                (0.125, 0.5, (11, 12, 13, 14, 15, 16)),
                (0.0, 1.0, (11, 12, 13, 14, 15, 16)),
                (0.25, 1.0, (5, 6, 11, 12)),
                (0.5625, 0.875, (6, 8, 10)),
                (0.75, 1.0, (5, 6, 8, 10, 11, 12)),
            ),
        ),
        phase_windows=(
            PhaseWindowSpec("preparation", 0.0, 0.375),
            PhaseWindowSpec("weight_transfer", 0.25, 0.75),
            PhaseWindowSpec("contact", 0.5625, 0.875),
            PhaseWindowSpec("follow_through", 0.75, 1.0),
        ),
        rules=_SERVE_RULES,
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
