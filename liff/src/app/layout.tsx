import type { Metadata } from 'next'
import './globals.css'
import React from 'react'
import { LiffProvider } from './LiffProvider'
import Navbar from '@/components/Navbar/Navbar'
import BottomNav from '@/components/Navbar/BottomNav'

export const metadata: Metadata = {
  title: '羽球動作分析',
  description: '個人動作評分、班級對照與教練建議'
}

// Applies the stored theme before first paint so the page never flashes.
const themeScript = `(function(){try{var t=localStorage.getItem('theme');if(!t){t=window.matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light'}if(t==='dark'){document.documentElement.classList.add('dark')}}catch(e){}})()`

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="zh-Hant" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body suppressHydrationWarning>
        <LiffProvider liffId={process.env.NEXT_PUBLIC_LIFF_ID || ''}>
          <Navbar />
          <main className="pb-24 pt-14">{children}</main>
          <BottomNav />
        </LiffProvider>
      </body>
    </html>
  )
}
