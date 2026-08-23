'use client'

import React, { useEffect } from 'react'
import { Check } from 'lucide-react'

interface ToastProps {
  /** The message to show. Empty means nothing is showing. */
  message: string
  onDismiss: () => void
  /** How long the message stays up, in milliseconds. */
  duration?: number
}

/**
 * A brief confirmation that something was saved.
 *
 * It sits at the bottom of the screen because that is where a thumb already
 * is on a phone, and it never takes pointer events: it confirms an action, it
 * is not one. Screen readers get it through the live region, which stays in
 * the tree whether or not a message is up so the announcement is not missed.
 */
export default function Toast({ message, onDismiss, duration = 2600 }: ToastProps) {
  useEffect(() => {
    if (!message) return
    const timer = setTimeout(onDismiss, duration)
    return () => clearTimeout(timer)
  }, [duration, message, onDismiss])

  return (
    <div
      role="status"
      aria-live="polite"
      className="pointer-events-none fixed inset-x-0 bottom-6 z-50 flex justify-center px-4"
    >
      {message ? (
        <span className="toast-enter flex items-center gap-2 rounded-full bg-neutral-900 px-4 py-2.5 text-sm text-white shadow-lg">
          <Check size={15} className="shrink-0" aria-hidden />
          {message}
        </span>
      ) : null}
    </div>
  )
}
