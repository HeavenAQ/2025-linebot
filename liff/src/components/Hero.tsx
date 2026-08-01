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
    <PageContainer className="pt-6">
      <header className="flex animate-fade-down items-center gap-4 rounded-2xl border border-border bg-card p-4 shadow-card">
        {profile?.pictureUrl ? (
          <Image
            src={profile.pictureUrl}
            alt={profile.displayName ?? '使用者頭像'}
            width={112}
            height={112}
            priority
            className="h-14 w-14 shrink-0 rounded-full border border-border object-cover"
          />
        ) : (
          <div className="h-14 w-14 shrink-0 rounded-full bg-muted" aria-hidden />
        )}
        <div className="min-w-0">
          <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
            歡迎回來
          </p>
          <p className="truncate text-lg font-semibold tracking-tight">{profile?.displayName}</p>
        </div>
      </header>
    </PageContainer>
  )
}
