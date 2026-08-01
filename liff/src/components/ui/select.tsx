import * as React from 'react'
import { ChevronDown } from 'lucide-react'

import { cn } from '@/lib/utils'

/**
 * Skinned native <select> — keeps the OS picker on mobile, which is the whole
 * audience here, while matching the report's hairline chrome.
 */
const Select = React.forwardRef<HTMLSelectElement, React.SelectHTMLAttributes<HTMLSelectElement>>(
  ({ className, children, ...props }, ref) => (
    <div className="relative w-full">
      <select
        ref={ref}
        className={cn(
          'select-reset h-10 w-full rounded-lg border border-border bg-card pl-3 pr-9 text-sm font-semibold text-foreground transition-colors duration-150 hover:border-primary/40 disabled:cursor-not-allowed disabled:opacity-40',
          className
        )}
        {...props}
      >
        {children}
      </select>
      <ChevronDown
        aria-hidden="true"
        size={15}
        className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground"
      />
    </div>
  )
)
Select.displayName = 'Select'

interface SelectFieldProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  label: string
  className?: string
}

const SelectField = React.forwardRef<HTMLSelectElement, SelectFieldProps>(
  ({ label, className, ...props }, ref) => (
    <label className={cn('block min-w-0 space-y-1.5', className)}>
      <span className="eyebrow">{label}</span>
      <Select ref={ref} {...props} />
    </label>
  )
)
SelectField.displayName = 'SelectField'

export { Select, SelectField }
