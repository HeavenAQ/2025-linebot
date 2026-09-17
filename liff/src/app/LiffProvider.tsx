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

import { setExpiredTokenHandler, setTokenSources } from '@/lib/api/client'
import ExperimentGate from '@/components/ExperimentGate'

const LiffContext = createContext<{
  liff: Liff | null
  profile: Profile | null
  liffError: string | null
  sessionExpired: boolean
}>({ liff: null, profile: null, liffError: null, sessionExpired: false })

export const useLiff = () => useContext(LiffContext)

/**
 * A LINE user ID to stand in for a real login during local development.
 *
 * Set NEXT_PUBLIC_DEV_USER_ID in liff/.env.local to a real learner's ID and the
 * pages render that learner's data straight from the local backend. Left unset,
 * nothing changes and the normal LIFF login runs.
 */
const devUserId = process.env.NEXT_PUBLIC_DEV_USER_ID?.trim()

/**
 * A real LINE ID token to authenticate local development against a backend.
 *
 * The backend only trusts tokens it can verify with LINE, so the dev bypass
 * cannot mint one — paste a token from a real session (liff.getIDToken() in the
 * LINE in-app browser) into NEXT_PUBLIC_DEV_ID_TOKEN to read live data. Left
 * unset, backend calls come back 401 and only the static UI renders.
 */
const devIdToken = process.env.NEXT_PUBLIC_DEV_ID_TOKEN?.trim()

/**
 * What a learner sees when LINE login cannot be completed. The technical cause
 * goes to the console; the page never shows SDK errors or credential details.
 */
const LOGIN_UNAVAILABLE = '無法完成 LINE 登入，請關閉此頁後重新從 LINE 開啟。'

export const LiffProvider: FC<PropsWithChildren<{ liffId: string }>> = ({ children, liffId }) => {
  const [liff, setLiff] = useState<Liff | null>(null)
  const [profile, setProfile] = useState<Profile | null>(null)
  const [liffError, setLiffError] = useState<string | null>(null)
  const [sessionExpired, setSessionExpired] = useState(false)
  const initializedRef = useRef(false)

  const initLiff = useCallback(async () => {
    if (initializedRef.current) return
    try {
      // Local design work cannot go through LIFF: liff.login() hands off to
      // LINE, which redirects back to the endpoint URL registered in the LINE
      // console — the production site — so localhost bounces away before it
      // renders anything. With a user ID set, stand in for LIFF and render
      // against whatever backend NEXT_PUBLIC_BACKEND_BASE_URL points at.
      // Guarded on NODE_ENV so a production build can never take this path.
      if (process.env.NODE_ENV !== 'production' && devUserId) {
        setProfile({
          userId: devUserId,
          displayName: process.env.NEXT_PUBLIC_DEV_DISPLAY_NAME || '開發測試帳號'
        } as Profile)
        setLiff({ isLoggedIn: () => true, login: () => undefined } as unknown as Liff)
        setTokenSources({ idToken: () => devIdToken ?? null, accessToken: () => null })
        initializedRef.current = true
        console.info('LIFF bypassed for local development; user:', devUserId)
        return
      }

      if (!liffId) {
        console.error('Missing NEXT_PUBLIC_LIFF_ID. Set it to your LIFF app ID.')
        setLiffError(LOGIN_UNAVAILABLE)
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
              ? process.env.NEXT_PUBLIC_LIFF_REDIRECT_URI || window.location.href
              : undefined
          if (!explicitRedirect) throw new Error('Missing redirectUri for LIFF login')
          liff.login({ redirectUri: explicitRedirect })
          return
        } else {
          // We already attempted login but still not logged in; avoid looping
          console.error('LIFF login could not be completed; check the LIFF settings.')
          setLiffError(LOGIN_UNAVAILABLE)
        }
      } else {
        // Clear guard once logged in
        if (typeof window !== 'undefined') {
          sessionStorage.removeItem('liff-login-initiated')
        }

        // update profile (only when logged in)
        try {
          const prof = await liff.getProfile()
          setProfile(prof)
          console.log(prof.pictureUrl)
        } catch (e) {
          console.warn('Failed to get LIFF profile:', e)
        }
        // Every backend call proves who is asking with one of these tokens, so
        // hand over the getters rather than the strings they return right now.
        // The ID token is sent while fresh; within a minute of its hourly
        // expiry calls switch to the longer-lived access token.
        setTokenSources({
          idToken: () => liff.getIDToken(),
          accessToken: () => liff.getAccessToken()
        })
        // getIDToken() needs the openid scope, and returns null without it.
        // The access token still identifies the learner, but with neither the
        // backend sees an anonymous caller and refuses everything, which looks
        // like a login failure rather than a missing scope — so say which it is.
        if (!liff.getIDToken()) {
          console.warn('LIFF returned no ID token; the openid scope may not be granted')
          if (!liff.getAccessToken()) {
            console.error('LIFF returned neither an ID token nor an access token')
            setLiffError(LOGIN_UNAVAILABLE)
          }
        }

        // The access token outlives the hour-long ID token, but it expires too,
        // and LIFF refreshes neither. Once the backend refuses the credential
        // we hold, this does NOT log the learner back in: liff.login()
        // redirects to LINE and back to the endpoint URL registered in the
        // console, which in an ordinary browser bounces them away from the page
        // they were on. The session is marked stale instead, and the app asks
        // them to reopen it from LINE.
        setExpiredTokenHandler(() => setSessionExpired(true))

        setLiff(liff)
        initializedRef.current = true
      }
    } catch (error) {
      console.error('LIFF init failed:', error)
      setLiffError(LOGIN_UNAVAILABLE)
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
        liffError,
        sessionExpired
      }}
    >
      <ExperimentGate authenticated={Boolean(liff && profile) && !sessionExpired}
        loginError={liffError || (sessionExpired ? '登入已失效，請重新從 LINE 開啟此頁。' : null)}>
        {children}
      </ExperimentGate>
    </LiffContext.Provider>
  )
}
