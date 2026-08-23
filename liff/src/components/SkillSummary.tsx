'use client'

import { Sparkles } from 'lucide-react'

import { SkillNameMap, type Skill } from '@/lib/types'

interface SkillSummaryProps {
  skill: Skill
  summary: string
  loading: boolean
  error?: string
}

const SkeletonLine = ({ width }: { width: string }) => (
  <span className="block h-3.5 animate-pulse rounded bg-muted" style={{ width }} />
)

/**
 * The AI's read on how this skill is going — the first thing a student sees.
 *
 * It leads the page rather than sitting in a panel among panels: the tinted
 * surface and the rule under the heading separate it from the numbers below,
 * and the body runs at reading size because on a phone this is a paragraph
 * someone actually reads, not a label they scan.
 */
export default function SkillSummary({ skill, summary, loading, error }: SkillSummaryProps) {
  return (
    <section
      aria-label="AI 學習總結"
      className="rounded-xl border border-primary/20 bg-primary/[0.04] p-4"
    >
      <div className="flex items-center gap-2 border-b border-primary/15 pb-2.5">
        <Sparkles size={15} className="shrink-0 text-primary" aria-hidden />
        <h2 className="text-[13px] font-semibold tracking-tight text-primary">AI 學習總結</h2>
        <span className="ml-auto rounded-full bg-primary/10 px-2 py-0.5 text-[11px] font-medium text-primary">
          {SkillNameMap[skill]}
        </span>
      </div>

      <div className="pt-3">
        {loading ? (
          <div className="space-y-2" aria-live="polite" aria-busy="true">
            <SkeletonLine width="100%" />
            <SkeletonLine width="92%" />
            <SkeletonLine width="64%" />
          </div>
        ) : error ? (
          <p className="text-[13px] leading-6 text-muted-foreground">{error}</p>
        ) : summary ? (
          <p className="whitespace-pre-line break-words text-[15px] leading-7">{summary}</p>
        ) : (
          <p className="text-[13px] leading-6 text-muted-foreground">
            上傳練習影片或與教練機器人對話後，這裡會整理你的學習重點。
          </p>
        )}
      </div>
    </section>
  )
}
