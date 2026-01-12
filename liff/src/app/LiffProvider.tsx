'use client'
import React, {
  createContext,
  FC,
  PropsWithChildren,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState
} from 'react'

import { Profile } from '@liff/get-profile'
import { Liff } from '@line/liff'

const LiffContext = createContext<{
  liff: Liff | null
  profile: Profile | null
  liffError: string | null
}>({ liff: null, profile: null, liffError: null })

export const useLiff = () => useContext(LiffContext)

export const LiffProvider: FC<PropsWithChildren<{ liffId: string }>> = ({ children, liffId }) => {
  const [liff, setLiff] = useState<Liff | null>(null)
  const [profile, setProfile] = useState<Profile | null>(null)
  const [liffError, setLiffError] = useState<string | null>(null)
  const initializedRef = useRef(false)

  const initLiff = useCallback(async () => {
    if (initializedRef.current) return
    try {
      if (!liffId) {
        setLiffError('Missing NEXT_PUBLIC_LIFF_ID. Set it to your LIFF app ID.')
        return
      }
      const liffModule = await import('@line/liff')
      const liff = liffModule.default
      console.log('LIFF init...')

      // init LIFF
      await liff.init({
        liffId: liffId,
        withLoginOnExternalBrowser: true
      })

      // Ensure login state and prevent infinite loops with a one-shot guard.
      if (!liff.isLoggedIn()) {
        const guardKey = 'liff-login-initiated'
        const alreadyInitiated = typeof window !== 'undefined' && sessionStorage.getItem(guardKey)
        if (!alreadyInitiated) {
          sessionStorage.setItem(guardKey, '1')
          // Redirect back to a stable endpoint URL to satisfy LIFF expectations
          const explicitRedirect =
            typeof window !== 'undefined'
              ? (process.env.NEXT_PUBLIC_LIFF_REDIRECT_URI || window.location.href)
              : undefined
          if (!explicitRedirect) throw new Error('Missing redirectUri for LIFF login')
          liff.login({ redirectUri: explicitRedirect })
          return
        } else {
          // We already attempted login but still not logged in; avoid looping
          setLiffError('LIFF login could not be completed. Please try again or check LIFF settings.')
        }
      } else {
        // Clear guard once logged in
        if (typeof window !== 'undefined') sessionStorage.removeItem('liff-login-initiated')

        // update profile (only when logged in)
        try {
          const prof = await liff.getProfile()
          setProfile(prof)
          console.log(prof.pictureUrl)
        } catch (e) {
          console.warn('Failed to get LIFF profile:', e)
        }
        console.log(liff.getDecodedIDToken())

        setLiff(liff)
        initializedRef.current = true
      }
    } catch (error) {
      console.log('LIFF init failed.')
      setLiffError((error as Error).toString())
    }
  }, [liffId])

  // init Liff
  useEffect(() => {
    console.log('LIFF init start...')
    initLiff()
  }, [initLiff])

  return (
    <LiffContext.Provider
      value={{
        liff,
        profile,
        liffError
      }}
    >
      {children}
    </LiffContext.Provider>
  )
}
