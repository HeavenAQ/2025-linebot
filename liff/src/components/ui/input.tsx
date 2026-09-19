import * as React from 'react'

import { cn } from '@/lib/utils'

const Input = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
  ({ className, ...props }, ref) => (
    <input
      ref={ref}
      className={cn(
        'h-10 w-full rounded-lg glass-inset border border-border/70 px-3 text-sm font-medium text-card-foreground transition-colors duration-200 placeholder:font-normal placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/40 disabled:cursor-not-allowed disabled:opacity-50',
        className
      )}
      {...props}
    />
  )
)
Input.displayName = 'Input'

interface InputFieldProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label: string
  /** Shown under the field; red when it is an error. */
  hint?: string
  error?: string
  className?: string
}

/** Label + input + hint, matching SelectField so forms line up with filters. */
const InputField = React.forwardRef<HTMLInputElement, InputFieldProps>(
  ({ label, hint, error, className, id, ...props }, ref) => {
    const generated = React.useId()
    const inputId = id ?? generated
    const describedBy = error || hint ? `${inputId}-hint` : undefined
    return (
      <div className={cn('block min-w-0 space-y-1.5', className)}>
        <label htmlFor={inputId} className="text-[13px] font-medium text-muted-foreground">
          {label}
        </label>
        <Input
          ref={ref}
          id={inputId}
          aria-invalid={error ? true : undefined}
          aria-describedby={describedBy}
          {...props}
        />
        {(error || hint) && (
          <p
            id={describedBy}
            role={error ? 'alert' : undefined}
            className={cn('text-xs', error ? 'text-destructive' : 'text-muted-foreground')}
          >
            {error || hint}
          </p>
        )}
      </div>
    )
  }
)
InputField.displayName = 'InputField'

export { Input, InputField }
