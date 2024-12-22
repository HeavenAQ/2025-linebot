'use client'
import React from 'react'
import { Fragment, useState } from 'react'
import { Menu, Transition } from '@headlessui/react'
import { FiMenu, FiX } from 'react-icons/fi'
import Link from 'next/link'

const items = [
  {
    displayName: '個人成績',
    href: '/personal'
  },
  {
    displayName: '班級排名',
    href: '/class'
  },
  {
    displayName: 'GPT評估建議',
    href: '/gpt-chat'
  }
]
export default function DropDownIcon() {
  const [isOpen, setIsOpen] = useState(false)
  return (
    <Menu as="div" className="relative inline-block rounded-lg border border-zinc-500 text-left">
      <Menu.Button
        className="relative flex cursor-pointer items-center justify-center rounded-lg p-[0.5rem] duration-200"
        onClick={() => {
          setIsOpen(!isOpen)
        }}
      >
        {isOpen ? <FiX /> : <FiMenu />}
      </Menu.Button>
      <Transition
        as={Fragment}
        enter="transition ease-out duration-100"
        enterFrom="transform opacity-0 scale-95"
        enterTo="transform opacity-100 scale-100"
        leave="transition ease-in duration-75"
        leaveFrom="transform opacity-100 scale-100"
        leaveTo="transform opacity-0 scale-95"
      >
        <Menu.Items className="absolute -bottom-36 right-0 flex w-56 flex-col rounded-md bg-orange-50 p-1 text-black dark:divide-zinc-100 dark:bg-gray-700 dark:text-white">
          <Menu.Item>
            <div className="w-full rounded-md p-1 pl-2 font-bold">Content</div>
          </Menu.Item>
          {items.map((item, i) => (
            <Menu.Item key={i}>
              {({ active }) => (
                <Link
                  href={`${item.href}`}
                  className={`${
                    active && 'bg-zinc-600 text-white dark:bg-orange-50 dark:text-black'
                  } w-full rounded-md p-1 px-3`}
                >
                  {item.displayName}
                </Link>
              )}
            </Menu.Item>
          ))}
        </Menu.Items>
      </Transition>
    </Menu>
  )
}
