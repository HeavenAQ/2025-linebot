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
 * Tabs marked by a vermilion rule rather than a filled pill — the selected
 * state is stated once, quietly, and the row stays flat.
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
      className={cn('flex items-stretch gap-7 border-b border-border', className)}
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
              // -1px lifts the marker onto the container rule
              'relative -mb-px flex min-w-0 items-center gap-2 border-b pb-3 pt-1 tracking-[0.06em] transition-colors duration-300',
              size === 'sm' ? 'text-xs' : 'text-[13px]',
              selected
                ? 'border-highlight text-foreground'
                : 'border-transparent text-muted-foreground hover:text-foreground'
            )}
          >
            {Icon ? <Icon aria-hidden size={size === 'sm' ? 13 : 15} /> : null}
            <span className="truncate">{optionLabel}</span>
          </button>
        )
      })}
    </div>
  )
}
