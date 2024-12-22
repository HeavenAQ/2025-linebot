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
    <html lang="en" className={`${mPlusRounded1c.className} duration-200`}>
      <body className="bg-[#eee7d7] text-black dark:bg-zinc-900 dark:text-white">
        <Navbar />
        <LiffProvider liffId={process.env.NEXT_PUBLIC_LIFF_ID || ''}>
          <Hero />
          {children}
        </LiffProvider>
      </body>
    </html>
  )
}
