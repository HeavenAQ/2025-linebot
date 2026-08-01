'use client'

import * as React from 'react'
import { ArrowDownRight, ArrowUpRight, Minus } from 'lucide-react'

import { cn } from '@/lib/utils'

export interface Session {
  date: string
  grade: number
}

interface ScoreBandProps {
  /** Oldest to newest. */
  sessions: readonly Session[]
  selectedDate: string
  onSelectDate: (_date: string) => void
  skillName: string
}

const shortDate = (date: string) => date.slice(5, 10).replace('-', '/')
const timeOfDay = (date: string) => {
  const parts = date.split('-')
  return parts.length >= 5 ? `${parts[3]}:${parts[4]}` : ''
}

/**
 * The session tape: one tick per recorded session, height proportional to the
 * score, newest on the right. It doubles as the date picker — tapping a tick
 * loads that session — so it earns its space instead of decorating the header.
 */
export default function ScoreBand({
  sessions,
  selectedDate,
  onSelectDate,
  skillName
}: ScoreBandProps) {
  if (sessions.length === 0) return null

  const index = sessions.findIndex(s => s.date === selectedDate)
  const current = index >= 0 ? sessions[index] : sessions[sessions.length - 1]
  const previous = index > 0 ? sessions[index - 1] : undefined
  const delta = previous ? current.grade - previous.grade : undefined

  const Trend = delta === undefined || Math.abs(delta) < 0.05 ? Minus : delta > 0 ? ArrowUpRight : ArrowDownRight

  return (
    <section className="bg-primary text-primary-foreground">
      <div className="mx-auto w-full max-w-content px-4 py-5">
        <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-primary-foreground/60">
          {skillName}
          {timeOfDay(current.date) ? ` · ${shortDate(current.date)} ${timeOfDay(current.date)}` : ''}
        </p>

        <div className="mt-2 flex items-end gap-4">
          <p className="num font-data text-score">{current.grade.toFixed(1)}</p>
          <div className="mb-1.5 min-w-0">
            <p className="text-xs font-semibold text-primary-foreground/70">總分 / 100</p>
            {delta !== undefined && (
              <p className="num mt-1 flex items-center gap-1 text-sm font-semibold">
                <Trend aria-hidden size={15} />
                {delta > 0 ? '+' : ''}
                {delta.toFixed(1)}
                <span className="font-sans text-xs font-medium text-primary-foreground/70">
                  較上次
                </span>
              </p>
            )}
          </div>
        </div>

        {sessions.length > 1 && (
          <div className="mt-5">
            <div
              role="group"
              aria-label="歷次成績，選擇以載入該次分析"
              className="flex h-12 items-end gap-1"
            >
              {sessions.map(session => {
                const active = session.date === selectedDate
                return (
                  <button
                    key={session.date}
                    type="button"
                    onClick={() => onSelectDate(session.date)}
                    aria-current={active}
                    aria-label={`${shortDate(session.date)}，${session.grade.toFixed(1)} 分`}
                    title={`${shortDate(session.date)} · ${session.grade.toFixed(1)}`}
                    className="group relative flex h-full flex-1 items-end justify-center"
                  >
                    <span
                      className={cn(
                        'w-full transition-colors duration-150',
                        active
                          ? 'bg-primary-foreground'
                          : 'bg-primary-foreground/25 group-hover:bg-primary-foreground/50'
                      )}
                      style={{ height: `${Math.max(6, Math.min(100, session.grade))}%` }}
                    />
                  </button>
                )
              })}
            </div>
            <div className="mt-1.5 flex justify-between text-[10px] font-medium text-primary-foreground/50">
              <span className="num">{shortDate(sessions[0].date)}</span>
              <span>{sessions.length} 次分析</span>
              <span className="num">{shortDate(sessions[sessions.length - 1].date)}</span>
            </div>
          </div>
        )}
      </div>
    </section>
  )
}
