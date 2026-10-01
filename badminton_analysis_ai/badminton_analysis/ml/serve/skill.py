"""The serve: its phases, scorer and coaching modules."""

from badminton_analysis.ml.serve import coaching, phases, scorer
from badminton_analysis.ml.skill import SkillDefinition
from badminton_analysis.models.types import Skill


class ServeSkill(SkillDefinition):
    skill = Skill.SERVE
    phases = phases
    scorer = scorer
    coaching = coaching
