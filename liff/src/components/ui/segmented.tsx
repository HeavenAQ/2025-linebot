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
  role?: 'tablist' | 'group'
  label: string
  /** `underline` for page-level views, `solid` for a compact in-panel toggle. */
  variant?: 'underline' | 'solid'
  className?: string
}

export function Segmented<T extends string>({
  options,
  value,
  onChange,
  role = 'group',
  label,
  variant = 'underline',
  className
}: SegmentedProps<T>) {
  const isTabs = role === 'tablist'

  if (variant === 'solid') {
    return (
      <div
        role={role}
        aria-label={label}
        className={cn('inline-grid h-8 gap-px rounded-lg bg-border p-px', className)}
        style={{ gridTemplateColumns: `repeat(${options.length}, minmax(0, 1fr))` }}
      >
        {options.map(({ value: v, label: l }) => {
          const selected = v === value
          return (
            <button
              key={v}
              type="button"
              aria-pressed={selected}
              onClick={() => onChange(v)}
              className={cn(
                'rounded-md px-3 text-xs font-semibold transition-colors duration-150',
                selected
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-card text-muted-foreground hover:text-foreground'
              )}
            >
              {l}
            </button>
          )
        })}
      </div>
    )
  }

  return (
    <div role={role} aria-label={label} className={cn('flex gap-6 border-b border-border', className)}>
      {options.map(({ value: v, label: l, icon: Icon }) => {
        const selected = v === value
        return (
          <button
            key={v}
            type="button"
            role={isTabs ? 'tab' : undefined}
            aria-selected={isTabs ? selected : undefined}
            onClick={() => onChange(v)}
            className={cn(
              // -1px pulls the active rule onto the container's border
              'relative -mb-px flex items-center gap-2 border-b-2 pb-2.5 pt-1 text-sm font-semibold transition-colors duration-150',
              selected
                ? 'border-primary text-foreground'
                : 'border-transparent text-muted-foreground hover:text-foreground'
            )}
          >
            {Icon ? <Icon aria-hidden size={15} /> : null}
            {l}
          </button>
        )
      })}
    </div>
  )
}
