'use client'
import React, { Fragment } from 'react'
import { Menu, Transition } from '@headlessui/react'
import { BarChart3, Menu as MenuIcon, MessageSquareText, Users } from 'lucide-react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'

import { buttonVariants } from '@/components/ui/button'
import { cn } from '@/lib/utils'

const items = [
  { displayName: '個人成績', href: '/personal', icon: BarChart3 },
  { displayName: '班級排名', href: '/class', icon: Users },
  { displayName: 'GPT評估建議', href: '/gpt-chat', icon: MessageSquareText }
]

export default function DropDownIcon() {
  const pathname = usePathname()

  return (
    <Menu as="div" className="relative inline-block text-left">
      <Menu.Button
        aria-label="開啟選單"
        className={cn(buttonVariants({ variant: 'outline', size: 'icon' }))}
      >
        <MenuIcon size={17} />
      </Menu.Button>
      <Transition
        as={Fragment}
        enter="transition ease-out duration-150"
        enterFrom="transform opacity-0 scale-95"
        enterTo="transform opacity-100 scale-100"
        leave="transition ease-in duration-100"
        leaveFrom="transform opacity-100 scale-100"
        leaveTo="transform opacity-0 scale-95"
      >
        <Menu.Items className="absolute right-0 top-full z-40 mt-2 w-56 origin-top-right glass rounded-xl p-1.5 text-popover-foreground shadow-elevated focus:outline-none">
          {items.map(item => {
            const isActive = pathname === item.href
            return (
              <Menu.Item key={item.href}>
                {({ active }) => (
                  <Link
                    href={item.href}
                    className={cn(
                      'flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-sm font-medium transition-colors duration-150',
                      isActive
                        ? 'bg-primary text-primary-foreground'
                        : active
                          ? 'bg-accent text-accent-foreground'
                          : 'text-popover-foreground'
                    )}
                  >
                    <item.icon aria-hidden size={16} />
                    {item.displayName}
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
