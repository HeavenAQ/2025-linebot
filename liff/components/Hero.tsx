'use client'
import React, { useEffect, useState } from 'react'
import Image from 'next/image'

import { useLiff } from '../app/LiffProvider'

export default function Hero() {
  const { liff, profile } = useLiff()
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    if (liff) {
      if (!liff.isLoggedIn) {
        liff.login()
      }
      setIsLoading(false)
    }
  }, [liff])

  if (isLoading) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="h-16 w-16 animate-spin rounded-full border-b-2 border-t-2 border-zinc-700"></div>
      </div>
    )
  }

  return (
    <div className="mx-auto w-11/12 max-w-[800px] animate-fade-down pt-20">
      <header className="row-start-2 flex flex-col items-center gap-8 sm:items-start">
        <Image
          src={profile?.pictureUrl ?? ''}
          alt="Next.js logo"
          width={180}
          height={38}
          priority
          className="mx-auto w-36 rounded-full border border-white p-[2px]"
        />
        <div className="mx-auto w-11/12 rounded-lg bg-[#f4f0e9] p-3 text-center dark:bg-zinc-800">
          <p className="text-center text-sm text-black dark:text-white">
            歡迎回來 <strong>{profile?.displayName}</strong> !
          </p>
        </div>
      </header>
    </div>
  )
}
