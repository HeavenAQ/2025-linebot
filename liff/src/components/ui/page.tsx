import * as React from 'react'

import { cn } from '@/lib/utils'

/** One content measure, shared by the navbar and every page. */
export function PageContainer({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn('mx-auto w-full max-w-content px-5', className)} {...props} />
}

export function PageSection({ className, ...props }: React.HTMLAttributes<HTMLElement>) {
  return <section className={cn('space-y-4', className)} {...props} />
}

interface SectionHeadingProps extends React.HTMLAttributes<HTMLHeadingElement> {
  description?: React.ReactNode
  /** Small Latin caption set above the title, signage-style. */
  caption?: string
}

export function SectionHeading({
  className,
  children,
  description,
  caption,
  ...props
}: SectionHeadingProps) {
  return (
    <div className="space-y-2">
      {caption ? <p className="caption-latin">{caption}</p> : null}
      <h2 className={cn('section-label mincho text-[15px]', className)} {...props}>
        {children}
      </h2>
      {description ? (
        <p className="text-[13px] leading-7 text-muted-foreground">{description}</p>
      ) : null}
    </div>
  )
}
