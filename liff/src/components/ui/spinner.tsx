import React from 'react'

import { cn } from '@/lib/utils'

interface SpinnerProps {
  size?: 'sm' | 'md' | 'lg'
  /** Centre in the viewport. Off by default so it can sit inline inside a card. */
  fullscreen?: boolean
  label?: string
  className?: string
}

const sizes = {
  sm: 'h-5 w-5 border-2',
  md: 'h-8 w-8 border-2',
  lg: 'h-12 w-12 border-[3px]'
} as const

export default function Spinner({
  size = 'md',
  fullscreen = false,
  label = '載入中',
  className
}: SpinnerProps) {
  return (
    <div
      role="status"
      aria-label={label}
      className={cn(
        'flex items-center justify-center',
        fullscreen ? 'min-h-[60vh]' : 'py-10',
        className
      )}
    >
      <div
        className={cn(
          'animate-spin rounded-full border-border border-t-primary',
          sizes[size]
        )}
      />
      <span className="sr-only">{label}</span>
    </div>
  )
}
