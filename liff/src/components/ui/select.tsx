import * as React from 'react'

import { cn } from '@/lib/utils'

/**
 * Native <select> on a single underline. Boxing every control makes a page
 * look busy; a rule states "this is editable" with one line.
 */
const Select = React.forwardRef<HTMLSelectElement, React.SelectHTMLAttributes<HTMLSelectElement>>(
  ({ className, children, ...props }, ref) => (
    <div className="relative w-full">
      <select
        ref={ref}
        className={cn(
          'select-reset w-full border-b border-border bg-transparent py-2 pl-0 pr-6 text-sm text-foreground transition-colors duration-300 hover:border-foreground focus:border-foreground disabled:cursor-not-allowed disabled:opacity-40',
          className
        )}
        {...props}
      >
        {children}
      </select>
      <span
        aria-hidden="true"
        className="pointer-events-none absolute right-1 top-1/2 h-1.5 w-1.5 -translate-y-2/3 rotate-45 border-b border-r border-muted-foreground"
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
    <label className={cn('block min-w-0', className)}>
      <span className="text-[11px] tracking-[0.12em] text-muted-foreground">{label}</span>
      <Select ref={ref} {...props} />
    </label>
  )
)
SelectField.displayName = 'SelectField'

export { Select, SelectField }
