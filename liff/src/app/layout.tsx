import type { Metadata } from 'next'
import './globals.css'
import React from 'react'
import { LiffProvider } from './LiffProvider'
import Navbar from '@/components/Navbar/Navbar'
import { mPlusRounded1c } from '@/components/Fonts/M_PLUS_Rounded_1c'
import Hero from '@/components/Hero'

export const metadata: Metadata = {
  title: 'NSTC LINE BOT PROJECT (115)',
  description: 'Student learning dashboard'
}

// Applies the stored theme before first paint so the page never flashes light-on-dark.
const themeScript = `(function(){try{var t=localStorage.getItem('theme');if(!t){t=window.matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light'}if(t==='dark'){document.documentElement.classList.add('dark')}}catch(e){}})()`

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="zh-Hant" className={mPlusRounded1c.className} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body suppressHydrationWarning>
        <Navbar />
        <LiffProvider liffId={process.env.NEXT_PUBLIC_LIFF_ID || ''}>
          <div className="pb-16 pt-14">
            <Hero />
            {children}
          </div>
        </LiffProvider>
      </body>
    </html>
  )
}
