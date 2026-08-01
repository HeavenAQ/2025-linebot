'use client'
import React, { useEffect, useState } from 'react'
import Image from 'next/image'

import { useLiff } from '../app/LiffProvider'
import Spinner from './ui/spinner'
import { PageContainer } from './ui/page'

export default function Hero() {
  const { liff, profile } = useLiff()
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    if (liff) {
      if (!liff.isLoggedIn()) liff.login()
      setIsLoading(false)
    }
  }, [liff])

  if (isLoading) {
    return <Spinner fullscreen />
  }

  return (
    <PageContainer className="enter pt-ma-sm">
      <header className="flex items-center gap-4 border-b border-border pb-7">
        {profile?.pictureUrl ? (
          <Image
            src={profile.pictureUrl}
            alt=""
            width={96}
            height={96}
            priority
            className="h-12 w-12 shrink-0 rounded-full object-cover grayscale"
          />
        ) : (
          <div className="h-12 w-12 shrink-0 rounded-full bg-muted" aria-hidden />
        )}
        <div className="min-w-0">
          <p className="caption-latin">Welcome back</p>
          <p className="mincho mt-1.5 truncate text-lg leading-none">{profile?.displayName}</p>
        </div>
      </header>
    </PageContainer>
  )
}
