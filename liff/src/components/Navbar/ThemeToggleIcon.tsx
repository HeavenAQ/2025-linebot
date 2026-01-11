'use client'

import React, { useEffect, useState } from 'react'
import { IoSunny, IoMoon } from 'react-icons/io5'

export default function ThemeToggleIcon() {
  const [isMounted, setIsMounted] = useState(false)
  const [theme, setTheme] = useState<'light' | 'dark'>('light')

  useEffect(() => {
    if (typeof window !== 'undefined') {
      const storedTheme = localStorage.getItem('theme') as 'light' | 'dark' | null
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches

      if (storedTheme) {
        setTheme(storedTheme)
      } else if (prefersDark) {
        setTheme('dark')
      } else {
        setTheme('light')
      }
    }
    setIsMounted(true)
  }, [])

  useEffect(() => {
    const root = document.documentElement
    if (theme === 'dark') {
      root.classList.add('dark')
    } else {
      root.classList.remove('dark')
    }
  }, [theme])

  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light'
    localStorage.setItem('theme', newTheme)
    setTheme(newTheme)
  }

  if (!isMounted) return null

  return (
    <div className="inline-flex items-center rounded-lg bg-orange-300 p-[1px] duration-200 dark:bg-zinc-600">
      <button
        aria-label="Toggle Theme"
        className="cursor-pointer rounded-lg p-2 text-black dark:text-zinc-100"
        onClick={toggleTheme}
      >
        {theme === 'light' ? <IoSunny /> : <IoMoon />}
      </button>
    </div>
  )
}
