'use client'

import React, { useEffect, useState } from 'react'
import { Moon, Sun } from 'lucide-react'

import { Button } from '@/components/ui/button'

export default function ThemeToggleIcon() {
  const [isMounted, setIsMounted] = useState(false)
  const [theme, setTheme] = useState<'light' | 'dark'>('light')

  useEffect(() => {
    // The inline script in the layout already applied the class; mirror it into state.
    setTheme(document.documentElement.classList.contains('dark') ? 'dark' : 'light')
    setIsMounted(true)
  }, [])

  useEffect(() => {
    if (!isMounted) return
    document.documentElement.classList.toggle('dark', theme === 'dark')
  }, [isMounted, theme])

  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light'
    localStorage.setItem('theme', newTheme)
    setTheme(newTheme)
  }

  // Reserve the slot before hydration so the navbar does not shift.
  if (!isMounted) return <div className="h-10 w-10" aria-hidden />

  return (
    <Button
      variant="outline"
      size="icon"
      aria-label={theme === 'light' ? '切換至深色模式' : '切換至淺色模式'}
      onClick={toggleTheme}
    >
      {theme === 'light' ? <Sun size={17} /> : <Moon size={17} />}
    </Button>
  )
}
