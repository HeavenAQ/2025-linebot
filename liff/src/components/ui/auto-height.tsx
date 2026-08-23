'use client'

import React, { useEffect, useRef, useState, type ReactNode } from 'react'

interface AutoHeightProps {
  children: ReactNode
  className?: string
}

/**
 * Animates its own height as the content inside it changes size.
 *
 * The caption under the video player is the reason this exists: each AI
 * correction is a different length, so the box grows and shrinks as playback
 * moves from one to the next, and everything below it jumps. A fixed height
 * would either clip the long corrections or leave a gap under the short ones,
 * so instead the height is measured and animated to.
 *
 * The measured height is applied on the frame after the content changes; until
 * the first measurement the element is left to size itself, so it renders at
 * the right height even if the observer never runs.
 */
export default function AutoHeight({ children, className }: AutoHeightProps) {
  const contentRef = useRef<HTMLDivElement>(null)
  const [height, setHeight] = useState<number | null>(null)

  useEffect(() => {
    const content = contentRef.current
    if (!content || typeof ResizeObserver === 'undefined') return
    const observer = new ResizeObserver(entries => {
      for (const entry of entries) setHeight(entry.contentRect.height)
    })
    observer.observe(content)
    return () => observer.disconnect()
  }, [])

  return (
    <div
      className={className ? `auto-height ${className}` : 'auto-height'}
      style={{ height: height === null ? undefined : height }}
    >
      <div ref={contentRef}>{children}</div>
    </div>
  )
}
