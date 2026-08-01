import React from 'react'

import { cn } from '@/lib/utils'

interface SpinnerProps {
  size?: 'sm' | 'md'
  fullscreen?: boolean
  label?: string
  className?: string
}

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
        fullscreen ? 'min-h-[50vh]' : 'py-10',
        className
      )}
    >
      <div
        className={cn(
          'animate-spin rounded-full border-2 border-border border-t-primary',
          size === 'sm' ? 'h-4 w-4' : 'h-7 w-7'
        )}
      />
      <span className="sr-only">{label}</span>
    </div>
  )
}
