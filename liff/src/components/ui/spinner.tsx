import React from 'react'

export default function Spinner() {
  return (
    <div className="flex h-screen items-center justify-center">
      <div className="h-16 w-16 animate-spin rounded-full border-b-2 border-t-2 border-zinc-700"></div>
    </div>
  )
}
