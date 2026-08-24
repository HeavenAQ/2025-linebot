import * as React from 'react'

import { cn } from '@/lib/utils'

/** The single content width shared by the navbar, hero and every page. */
export function PageContainer({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn('mx-auto w-full max-w-content px-4', className)} {...props} />
}
