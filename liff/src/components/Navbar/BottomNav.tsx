'use client'

import React from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { BarChart3, MessageSquareText, Users } from 'lucide-react'

import { cn } from '@/lib/utils'

const destinations = [
  { label: '個人成績', href: '/personal', icon: BarChart3 },
  { label: '班級對照', href: '/class', icon: Users },
  { label: '教練建議', href: '/gpt-chat', icon: MessageSquareText }
]

export default function BottomNav() {
  const pathname = usePathname()

  return (
    <nav
      aria-label="主要導覽"
      className="fixed inset-x-0 bottom-0 z-30 border-t border-border bg-background/95 pb-[env(safe-area-inset-bottom)] backdrop-blur-md"
    >
      <div className="mx-auto flex w-full max-w-content">
        {destinations.map(({ label, href, icon: Icon }) => {
          const active = pathname === href
          return (
            <Link
              key={href}
              href={href}
              aria-current={active ? 'page' : undefined}
              className={cn(
                'flex flex-1 flex-col items-center gap-1 pb-2 pt-2.5 text-[11px] font-semibold transition-colors duration-150',
                active ? 'text-primary' : 'text-muted-foreground hover:text-foreground'
              )}
            >
              <Icon aria-hidden size={19} strokeWidth={active ? 2.4 : 1.8} />
              {label}
            </Link>
          )
        })}
      </div>
    </nav>
  )
}
