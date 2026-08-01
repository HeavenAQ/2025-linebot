import * as React from 'react'
import { ChevronDown } from 'lucide-react'

import { cn } from '@/lib/utils'

/**
 * Skinned native <select>. Native keeps the OS picker on mobile (the LIFF target)
 * while the chrome matches the rest of the control set.
 */
const Select = React.forwardRef<HTMLSelectElement, React.SelectHTMLAttributes<HTMLSelectElement>>(
  ({ className, children, ...props }, ref) => (
    <div className="relative w-full">
      <select
        ref={ref}
        className={cn(
          'select-reset h-10 w-full rounded-lg glass-inset border border-border/70 pl-3 pr-9 text-sm font-medium text-card-foreground transition-colors duration-200 hover:bg-accent/60 disabled:cursor-not-allowed disabled:opacity-50',
          className
        )}
        {...props}
      >
        {children}
      </select>
      <ChevronDown
        aria-hidden="true"
        size={16}
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

/** Label + select pair, so every filter control on every page lines up identically. */
const SelectField = React.forwardRef<HTMLSelectElement, SelectFieldProps>(
  ({ label, className, ...props }, ref) => (
    <label className={cn('block min-w-0 space-y-1.5', className)}>
      <span className="text-[13px] font-medium text-muted-foreground">
        {label}
      </span>
      <Select ref={ref} {...props} />
    </label>
  )
)
SelectField.displayName = 'SelectField'

export { Select, SelectField }
