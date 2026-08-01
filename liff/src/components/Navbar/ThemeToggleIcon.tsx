'use client'

import React, { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'

/**
 * 陽 / 陰 rather than a sun and moon glyph — one character says it, and it
 * keeps the bar free of icon clutter.
 */
export default function ThemeToggleIcon() {
  const [isMounted, setIsMounted] = useState(false)
  const [theme, setTheme] = useState<'light' | 'dark'>('light')

  useEffect(() => {
    setTheme(document.documentElement.classList.contains('dark') ? 'dark' : 'light')
    setIsMounted(true)
  }, [])

  useEffect(() => {
    if (!isMounted) return
    document.documentElement.classList.toggle('dark', theme === 'dark')
  }, [isMounted, theme])

  const toggleTheme = () => {
    const next = theme === 'light' ? 'dark' : 'light'
    localStorage.setItem('theme', next)
    setTheme(next)
  }

  // Hold the slot before hydration so the bar does not shift.
  if (!isMounted) return <div className="h-10 w-10" aria-hidden />

  return (
    <Button
      variant="ghost"
      size="icon"
      aria-label={theme === 'light' ? '切換至深色' : '切換至淺色'}
      onClick={toggleTheme}
      className="mincho text-[15px]"
    >
      {theme === 'light' ? '陽' : '陰'}
    </Button>
  )
}
