import React from 'react'

import { cn } from '@/lib/utils'

interface SpinnerProps {
  size?: 'sm' | 'md' | 'lg'
  fullscreen?: boolean
  label?: string
  className?: string
}

const sizes = {
  sm: 'h-4 w-4',
  md: 'h-6 w-6',
  lg: 'h-9 w-9'
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
        fullscreen ? 'min-h-[50vh]' : 'py-14',
        className
      )}
    >
      {/* A single hairline arc, turning slowly. */}
      <div
        className={cn(
          'animate-spin rounded-full border border-border border-t-foreground [animation-duration:1.1s]',
          sizes[size]
        )}
      />
      <span className="sr-only">{label}</span>
    </div>
  )
}
