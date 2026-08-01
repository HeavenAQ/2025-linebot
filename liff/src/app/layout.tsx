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


export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="zh-Hant" className={mPlusRounded1c.className} suppressHydrationWarning>
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
