import type { Metadata } from 'next'
import './globals.css'
import React from 'react'
import { LiffProvider } from './LiffProvider'
import Navbar from '@/components/Navbar/Navbar'
import Hero from '@/components/Hero'

export const metadata: Metadata = {
  title: '動作分析 Badminton',
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
        <Navbar />
        <LiffProvider liffId={process.env.NEXT_PUBLIC_LIFF_ID || ''}>
          <div className="pb-ma pt-16">
            <Hero />
            {children}
          </div>
        </LiffProvider>
      </body>
    </html>
  )
}
