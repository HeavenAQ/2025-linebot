'use client'

import * as React from 'react'

import { cn } from '@/lib/utils'

export interface SegmentedOption<T extends string> {
  value: T
  label: string
  icon?: React.ComponentType<{ size?: number | string; 'aria-hidden'?: boolean }>
}

interface SegmentedProps<T extends string> {
  options: readonly SegmentedOption<T>[]
  value: T
  onChange: (_value: T) => void
  /** `tablist` drives tab panels, `group` is a plain toggle set. */
  role?: 'tablist' | 'group'
  label: string
  size?: 'sm' | 'md'
  className?: string
}

/**
 * One segmented control for every page-level view switch — the personal page
 * tabs and the video view-mode toggle previously each rolled their own.
 */
export function Segmented<T extends string>({
  options,
  value,
  onChange,
  role = 'group',
  label,
  size = 'md',
  className
}: SegmentedProps<T>) {
  const isTabs = role === 'tablist'

  return (
    <div
      role={role}
      aria-label={label}
      className={cn(
        'grid gap-1 glass-inset rounded-xl border border-border/70 p-1',
        size === 'sm' ? 'h-9' : 'h-11',
        className
      )}
      style={{ gridTemplateColumns: `repeat(${options.length}, minmax(0, 1fr))` }}
    >
      {options.map(({ value: optionValue, label: optionLabel, icon: Icon }) => {
        const selected = optionValue === value
        return (
          <button
            key={optionValue}
            type="button"
            role={isTabs ? 'tab' : undefined}
            aria-selected={isTabs ? selected : undefined}
            aria-pressed={isTabs ? undefined : selected}
            onClick={() => onChange(optionValue)}
            className={cn(
              'flex min-w-0 items-center justify-center gap-1.5 rounded-lg px-2 font-medium transition-colors duration-200',
              size === 'sm' ? 'text-xs' : 'text-sm',
              selected
                ? 'bg-primary text-primary-foreground'
                : 'text-muted-foreground hover:bg-accent/60 hover:text-accent-foreground'
            )}
          >
            {Icon ? <Icon aria-hidden size={size === 'sm' ? 14 : 16} /> : null}
            <span className="truncate">{optionLabel}</span>
          </button>
        )
      })}
    </div>
  )
}
