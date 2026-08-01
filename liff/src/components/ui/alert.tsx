import * as React from 'react'

import { cn } from '@/lib/utils'

type Tone = 'info' | 'fault'

export interface AlertProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: Tone
  title?: string
}

/**
 * Empty and failed states. Left rule instead of a full tinted box — it states
 * the problem without shouting, and says what to do next.
 */
export function Alert({ className, variant = 'info', title, children, ...props }: AlertProps) {
  return (
    <div
      role="status"
      className={cn(
        'border-l-2 py-1 pl-4',
        variant === 'fault' ? 'border-destructive' : 'border-border',
        className
      )}
      {...props}
    >
      {title ? <p className="text-sm font-semibold">{title}</p> : null}
      {children ? (
        <div className={cn('text-sm leading-6 text-muted-foreground', title && 'mt-1')}>
          {children}
        </div>
      ) : null}
    </div>
  )
}
