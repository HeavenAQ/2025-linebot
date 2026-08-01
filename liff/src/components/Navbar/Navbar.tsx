'use client'

import React from 'react'
import Image from 'next/image'

import Logo from './Logo'
import ThemeToggleIcon from './ThemeToggleIcon'
import { useLiff } from '@/app/LiffProvider'

/**
 * Identity and settings only. Destinations live in the bottom bar, where a
 * thumb can reach them.
 */
export default function Navbar() {
  const { profile } = useLiff()

  return (
    <header className="fixed inset-x-0 top-0 z-30 border-b border-border bg-background/90 backdrop-blur-md">
      <div className="mx-auto flex h-14 w-full max-w-content items-center gap-3 px-4">
        <Logo />
        <div className="ml-auto flex items-center gap-3">
          {profile?.displayName && (
            <div className="flex min-w-0 items-center gap-2">
              {profile.pictureUrl && (
                <Image
                  src={profile.pictureUrl}
                  alt=""
                  width={56}
                  height={56}
                  className="h-7 w-7 shrink-0 rounded-full object-cover"
                />
              )}
              <span className="max-w-[8rem] truncate text-sm font-medium">
                {profile.displayName}
              </span>
            </div>
          )}
          <ThemeToggleIcon />
        </div>
      </div>
    </header>
  )
}
