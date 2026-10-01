"""The smash: its phases, scorer and coaching modules."""

from badminton_analysis.ml.smash import coaching, phases, scorer
from badminton_analysis.ml.skill import SkillDefinition
from badminton_analysis.models.types import Skill


class SmashSkill(SkillDefinition):
    skill = Skill.SMASH
    phases = phases
    scorer = scorer
    coaching = coaching
