export const SkillNameMap = {
  serve: "發球",
  smash: "殺球",
  clear: "高遠球",
} as const

export type Skill = keyof typeof SkillNameMap

