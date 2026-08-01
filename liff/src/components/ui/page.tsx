import * as React from 'react'

import { cn } from '@/lib/utils'

/** The single content width shared by the navbar, hero and every page. */
export function PageContainer({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn('mx-auto w-full max-w-content px-4', className)} {...props} />
}

export function PageSection({ className, ...props }: React.HTMLAttributes<HTMLElement>) {
  return <section className={cn('space-y-4', className)} {...props} />
}

interface SectionHeadingProps extends React.HTMLAttributes<HTMLHeadingElement> {
  description?: React.ReactNode
}

export function SectionHeading({ className, children, description, ...props }: SectionHeadingProps) {
  return (
    <div className="space-y-1">
      <h2 className={cn('text-lg font-semibold tracking-tight', className)} {...props}>
        {children}
      </h2>
      {description ? <p className="text-sm text-muted-foreground">{description}</p> : null}
    </div>
  )
}
