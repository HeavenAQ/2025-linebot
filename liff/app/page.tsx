'use client'
import React, { useEffect } from 'react'
import Image from 'next/image'

import { useLiff } from './LiffProvider'

export default function Home() {
  const { liff, profile } = useLiff()

  useEffect(() => {
    if (!liff?.isLoggedIn) {
      liff?.login()
    }
  }, [liff])

  return (
    <div className="grid min-h-screen grid-rows-[20px_1fr_20px] items-center justify-items-center gap-16 p-8 pb-20 font-[family-name:var(--font-geist-sans)] sm:p-20">
      <main className="row-start-2 flex flex-col items-center gap-8 sm:items-start">
        <Image
          src={profile?.pictureUrl ?? '/next.svg'}
          alt="Next.js logo"
          width={180}
          height={38}
          priority
          className="mx-auto w-9/12 rounded-full border-2 p-1 sm:w-full dark:border-white"
        />
        <div className="w-full rounded-lg p-3 text-center dark:bg-zinc-800">
          <p className="text-center text-sm">
            歡迎回來 <strong>{profile?.displayName}</strong> !
          </p>
        </div>
      </main>
    </div>
  )
}
