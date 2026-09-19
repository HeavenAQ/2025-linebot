'use client'

import React, { useCallback, useEffect, useState, type ReactNode } from 'react'
import {
  authorizedFetch,
  rateLimitedMessage,
  setRegistrationRequiredHandler
} from '@/lib/api/client'
import ProfileForm from '@/components/ProfileForm'
import { hasExperimentRegistration } from '@/lib/registration'
import { gateView, SLOW_LOADING_MS, type RegistrationStatus } from '@/lib/gateView'

const screen =
  'mx-auto flex min-h-screen max-w-md flex-col justify-center gap-5 px-6 py-12 text-center'
const button =
  'min-h-12 rounded-lg bg-primary px-4 py-3 text-primary-foreground disabled:opacity-50'

/** UI guidance only: LINE events and every learner API enforce the same gate. */
export default function ExperimentGate({
  children,
  authenticated,
  loginError,
  onReauthenticate,
  recovering = false
}: {
  children: ReactNode
  authenticated: boolean
  loginError: string | null
  onReauthenticate: () => void
  recovering?: boolean
}) {
  const [status, setStatus] = useState<RegistrationStatus>('checking')
  const [error, setError] = useState('')
  const [slow, setSlow] = useState(false)

  const check = useCallback(async () => {
    setStatus('checking')
    try {
      const response = await authorizedFetch('/api/db/user')
      const body = await response.json().catch(() => null)
      if (response.ok) {
        setStatus(hasExperimentRegistration(body) ? 'ready' : 'required')
        return
      }
      if (response.status === 403 && body?.code === 'registration_required') {
        setStatus('required')
        return
      }
      const limited = await rateLimitedMessage(response)
      setError(
        limited ??
          (response.status === 401
            ? 'LINE 登入驗證未通過，請重新登入。'
            : '暫時無法確認登記資料，請稍後再試。')
      )
      setStatus('error')
    } catch {
      setError('無法連線確認登記資料，請檢查網路後重試。')
      setStatus('error')
    }
  }, [])

  useEffect(() => {
    setRegistrationRequiredHandler(() => setStatus('required'))
    return () => setRegistrationRequiredHandler(null)
  }, [])

  useEffect(() => {
    if (authenticated) void check()
    else setStatus('checking')
  }, [authenticated, check])

  const view = gateView({ authenticated, loginError, status, recovering })

  // Offer re-login only if loading is unusually slow, so a normal load is just
  // a spinner.
  useEffect(() => {
    setSlow(false)
    if (view !== 'loading' || recovering) return
    const timer = window.setTimeout(() => setSlow(true), SLOW_LOADING_MS)
    return () => window.clearTimeout(timer)
  }, [view, recovering])

  if (view === 'ready') return children

  if (view === 'loading') {
    return (
      <main className={screen} aria-busy="true">
        <div className="mx-auto h-12 w-12 animate-spin rounded-full border-[3px] border-border border-t-primary" />
        <h1 className="text-xl font-semibold">正在載入</h1>
        <p role="status" aria-live="polite" className="text-muted-foreground">
          {recovering
            ? '正在重新登入 LINE…'
            : authenticated
              ? '正在確認你的學習資料…'
              : '正在連線 LINE…'}
        </p>
        {slow && (
          <>
            <p className="text-sm text-muted-foreground">
              載入時間比平常久，可以重新登入 LINE 再試一次。已登記的實驗編號與姓名會保留。
            </p>
            <button type="button" className={button} onClick={onReauthenticate}>
              重新登入 LINE
            </button>
          </>
        )}
      </main>
    )
  }

  if (view === 'login-error') {
    return (
      <main className={screen}>
        <h1 className="text-2xl font-semibold">請重新登入 LINE</h1>
        <p role="alert">{loginError}</p>
        <p>已登記的實驗編號與姓名會保留，不需要重新登記。</p>
        <button type="button" className={button} onClick={onReauthenticate}>
          重新登入 LINE
        </button>
      </main>
    )
  }

  if (view === 'error') {
    return (
      <main className={screen}>
        <h1 className="text-xl font-semibold">暫時無法載入</h1>
        <p role="alert">{error}</p>
        <button type="button" className={button} onClick={() => void check()}>
          重新確認
        </button>
      </main>
    )
  }

  // First use: the student registers here rather than in the LINE chat, so
  // they never have to leave the page they just opened.
  return (
    <main className="mx-auto flex min-h-screen max-w-md flex-col justify-center gap-5 px-6 py-12">
      <div className="space-y-2 text-center">
        <h1 className="text-2xl font-semibold">請先完成實驗登記</h1>
        <p className="text-muted-foreground">
          填寫實驗編號與真實姓名後，就可以使用所有功能。
        </p>
      </div>
      <ProfileForm submitLabel="完成登記" onSaved={() => setStatus('ready')} />
    </main>
  )
}
