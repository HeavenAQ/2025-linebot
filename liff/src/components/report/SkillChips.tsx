'use client'

import * as React from 'react'

import { cn } from '@/lib/utils'
import { Skill, SkillNameMap } from '@/lib/types'

interface SkillChipsProps {
  skills: readonly Skill[]
  value: Skill
  onChange: (_skill: Skill) => void
}

/**
 * Skill picker, pinned directly under the top bar on every page that has one.
 * Chips rather than a dropdown: four short labels fit, and one tap beats two.
 */
export default function SkillChips({ skills, value, onChange }: SkillChipsProps) {
  return (
    <div className="border-b border-border bg-background">
      <div
        role="group"
        aria-label="選擇技能"
        className="mx-auto flex w-full max-w-content gap-2 overflow-x-auto px-4 py-3"
      >
        {skills.map(skill => (
          <button
            key={skill}
            type="button"
            aria-pressed={skill === value}
            onClick={() => onChange(skill)}
            className={cn(
              'shrink-0 rounded-lg border px-3 py-1.5 text-sm font-semibold transition-colors duration-150',
              skill === value
                ? 'border-primary bg-primary text-primary-foreground'
                : 'border-border bg-card text-muted-foreground hover:text-foreground'
            )}
          >
            {SkillNameMap[skill]}
          </button>
        ))}
      </div>
    </div>
  )
}
