import type { Metadata } from 'next'
import localFont from 'next/font/local'
import './globals.css'
import React from 'react'
import { LiffProvider } from './LiffProvider'

const geistSans = localFont({
  src: './fonts/GeistVF.woff',
  variable: '--font-geist-sans',
  weight: '100 900'
})
const geistMono = localFont({
  src: './fonts/GeistMonoVF.woff',
  variable: '--font-geist-mono',
  weight: '100 900'
})

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
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
        <LiffProvider liffId={process.env.NEXT_PUBLIC_LIFF_ID || ''}>{children}</LiffProvider>
      </body>
    </html>
  )
}
