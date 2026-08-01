import * as React from 'react'
import { cva, type VariantProps } from 'class-variance-authority'

import { cn } from '@/lib/utils'

const buttonVariants = cva(
  'inline-flex shrink-0 items-center justify-center gap-2 rounded-lg text-[13px] font-medium tracking-[0.06em] transition-colors duration-300 disabled:pointer-events-none disabled:opacity-40 [&_svg]:shrink-0',
  {
    variants: {
      variant: {
        primary: 'bg-primary text-primary-foreground hover:bg-primary/85',
        secondary: 'bg-secondary text-secondary-foreground hover:bg-accent',
        outline: 'border border-border bg-transparent text-foreground hover:bg-accent',
        ghost: 'text-muted-foreground hover:text-foreground',
        destructive: 'bg-destructive text-destructive-foreground hover:bg-destructive/90'
      },
      size: {
        sm: 'h-8 px-3 text-xs',
        md: 'h-10 px-5',
        lg: 'h-12 px-7',
        icon: 'h-10 w-10'
      }
    },
    defaultVariants: {
      variant: 'primary',
      size: 'md'
    }
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, type = 'button', ...props }, ref) => (
    <button
      ref={ref}
      type={type}
      className={cn(buttonVariants({ variant, size }), className)}
      {...props}
    />
  )
)
Button.displayName = 'Button'

export { Button, buttonVariants }
