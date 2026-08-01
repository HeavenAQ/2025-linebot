'use client'
import React, { Fragment } from 'react'
import { Menu, Transition } from '@headlessui/react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'

import { cn } from '@/lib/utils'

const items = [
  { displayName: '個人成績', latin: 'Personal', href: '/personal' },
  { displayName: '班級排名', latin: 'Class', href: '/class' },
  { displayName: '教練建議', latin: 'Coaching', href: '/gpt-chat' }
]

export default function DropDownIcon() {
  const pathname = usePathname()

  return (
    <Menu as="div" className="relative inline-block text-left">
      <Menu.Button
        aria-label="開啟選單"
        className="flex h-10 w-10 flex-col items-center justify-center gap-[5px] text-foreground"
      >
        <span aria-hidden className="block h-px w-4 bg-current" />
        <span aria-hidden className="block h-px w-4 bg-current" />
      </Menu.Button>
      <Transition
        as={Fragment}
        enter="transition ease-out duration-300"
        enterFrom="opacity-0 translate-y-1"
        enterTo="opacity-100 translate-y-0"
        leave="transition ease-in duration-200"
        leaveFrom="opacity-100"
        leaveTo="opacity-0"
      >
        <Menu.Items className="absolute right-0 top-full z-40 mt-3 w-52 origin-top-right border border-border bg-popover text-popover-foreground shadow-elevated focus:outline-none">
          {items.map(item => {
            const isActive = pathname === item.href
            return (
              <Menu.Item key={item.href}>
                {({ active }) => (
                  <Link
                    href={item.href}
                    className={cn(
                      'flex items-baseline justify-between gap-3 border-b border-border px-5 py-4 transition-colors duration-200 last:border-b-0',
                      active && 'bg-accent'
                    )}
                  >
                    <span className="mincho flex items-center gap-2.5 text-sm">
                      {isActive && <span aria-hidden className="h-[3px] w-[3px] bg-highlight" />}
                      <span className={cn(!isActive && 'pl-[13px]')}>{item.displayName}</span>
                    </span>
                    <span className="caption-latin">{item.latin}</span>
                  </Link>
                )}
              </Menu.Item>
            )
          })}
        </Menu.Items>
      </Transition>
    </Menu>
  )
}
