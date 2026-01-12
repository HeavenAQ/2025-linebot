'use client'
import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import Spinner from '@/components/ui/spinner'
import { useLiff } from './LiffProvider'

export default function Home() {
  const router = useRouter()
  const { liff, profile, liffError } = useLiff()

  useEffect(() => {
    // Wait for LIFF to process callback params on this endpoint, then navigate.
    if (!liff) return
    if (liff.isLoggedIn() || profile || liffError) {
      // Clean up LIFF code/state params on the endpoint to avoid re-processing on refresh
      if (typeof window !== 'undefined' && window.location.search.includes('code=')) {
        const url = new URL(window.location.href)
        url.search = ''
        window.history.replaceState(null, '', url.toString())
      }
      router.replace('/personal')
    }
  }, [liff, profile, liffError, router])

  return <Spinner />
}
