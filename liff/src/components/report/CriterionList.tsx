import * as React from 'react'

import { cn } from '@/lib/utils'
import type { GradingDetail } from '@/types'

interface CriterionListProps {
  details: readonly GradingDetail[]
  /** Points available per criterion. */
  max?: number
}

/**
 * Criteria as rows rather than a chart: the labels are Chinese phrases that
 * never fit inside a bar, and a phone reads a list far better than a rotated
 * axis. Order is the coaching order and is left untouched.
 */
export default function CriterionList({ details, max = 20 }: CriterionListProps) {
  if (details.length === 0) {
    return <p className="py-6 text-sm text-muted-foreground">這次分析沒有細項評分。</p>
  }

  const weakest = details.reduce((low, d) => (d.grade < low.grade ? d : low), details[0])

  return (
    <ul className="divide-y divide-border">
      {details.map((detail, i) => {
        const ratio = Math.max(0, Math.min(1, detail.grade / max))
        const needsWork = ratio < 0.6
        const isWeakest = detail === weakest && details.length > 1

        return (
          <li key={`${detail.description}-${i}`} className="py-3">
            <div className="flex items-baseline justify-between gap-3">
              <span className="min-w-0 text-sm font-medium leading-snug">
                {detail.description}
                {isWeakest && (
                  <span className="ml-2 whitespace-nowrap rounded-sm bg-destructive/10 px-1.5 py-0.5 align-middle text-[10px] font-semibold text-destructive">
                    最需改進
                  </span>
                )}
              </span>
              <span className="num shrink-0 font-data text-metric tabular-nums">
                {detail.grade.toFixed(1)}
                <span className="ml-0.5 text-xs font-medium text-muted-foreground">/{max}</span>
              </span>
            </div>
            <div className="mt-2 h-1.5 w-full overflow-hidden rounded-sm bg-muted">
              <div
                className={cn('h-full rounded-sm', needsWork ? 'bg-destructive' : 'bg-primary')}
                style={{ width: `${ratio * 100}%` }}
              />
            </div>
          </li>
        )
      })}
    </ul>
  )
}
