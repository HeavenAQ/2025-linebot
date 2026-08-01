import * as React from 'react'
import { cva, type VariantProps } from 'class-variance-authority'
import { AlertTriangle, Info, XCircle } from 'lucide-react'

import { cn } from '@/lib/utils'

const alertVariants = cva(
  'flex items-start gap-3 rounded-xl border bg-card p-4 text-sm leading-6 shadow-card',
  {
    variants: {
      variant: {
        info: 'border-border text-card-foreground',
        warning: 'border-highlight/40 text-card-foreground',
        error: 'border-destructive/40 text-card-foreground'
      }
    },
    defaultVariants: { variant: 'info' }
  }
)

const icons = {
  info: Info,
  warning: AlertTriangle,
  error: XCircle
} as const

const iconTone = {
  info: 'text-muted-foreground',
  warning: 'text-highlight',
  error: 'text-destructive'
} as const

export interface AlertProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof alertVariants> {
  title?: string
}

/** Single treatment for the "no data" / "load failed" messages scattered across pages. */
export function Alert({ className, variant = 'info', title, children, ...props }: AlertProps) {
  const tone = variant ?? 'info'
  const Icon = icons[tone]

  return (
    <div role="status" className={cn(alertVariants({ variant }), className)} {...props}>
      <Icon aria-hidden size={18} className={cn('mt-0.5 shrink-0', iconTone[tone])} />
      <div className="min-w-0">
        {title ? <p className="font-semibold">{title}</p> : null}
        {children ? <div className={cn(title && 'mt-1 text-muted-foreground')}>{children}</div> : null}
      </div>
    </div>
  )
}
