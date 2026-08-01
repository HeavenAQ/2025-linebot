import * as React from 'react'

import { cn } from '@/lib/utils'

export interface AlertProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'info' | 'warning' | 'error'
  title?: string
}

/**
 * Empty and failed states. A single rule and plain sentences — say what
 * happened and what to do, then stop.
 */
export function Alert({ className, variant = 'info', title, children, ...props }: AlertProps) {
  return (
    <div
      role="status"
      className={cn(
        'border-l py-1 pl-5',
        variant === 'info' ? 'border-border' : 'border-highlight',
        className
      )}
      {...props}
    >
      {title ? <p className="mincho text-[15px]">{title}</p> : null}
      {children ? (
        <div className={cn('text-[13px] leading-7 text-muted-foreground', title && 'mt-1.5')}>
          {children}
        </div>
      ) : null}
    </div>
  )
}
