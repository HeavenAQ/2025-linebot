import React from 'react'
import Link from 'next/link'

/**
 * Wordmark in mincho with a vermilion seal-dot, and the Latin name set small
 * and wide beneath it — the way a Japanese studio sets a masthead.
 */
const Logo = () => {
  return (
    <Link href="/" className="group inline-flex items-baseline gap-2.5 rounded-lg">
      <span className="mincho text-[17px] leading-none">動作分析</span>
      <span aria-hidden className="h-[3px] w-[3px] shrink-0 self-center bg-highlight" />
      <span className="caption-latin leading-none">Badminton</span>
    </Link>
  )
}

export default Logo
