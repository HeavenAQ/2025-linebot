'use client'
import React, {
  createContext,
  FC,
  PropsWithChildren,
  useCallback,
  useContext,
  useEffect,
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

  const initLiff = useCallback(async () => {
    try {
      const liffModule = await import('@line/liff')
      const liff = liffModule.default
      console.log('LIFF init...')

      // init LIFF
      await liff.init({
        liffId: liffId,
        withLoginOnExternalBrowser: true
      })

      // update profile
      await liff.getProfile().then(profile => {
        setProfile(profile)
        console.log(profile.pictureUrl)
      })
      console.log(liff.getDecodedIDToken())

      setLiff(liff)
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
