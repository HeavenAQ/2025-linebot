'use client'

import { useCallback, useEffect, useState, type ReactNode } from 'react'
import {
  authorizedFetch,
  rateLimitedMessage,
  setRegistrationRequiredHandler
} from '@/lib/api/client'
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
  loginError
}: {
  children: ReactNode
  authenticated: boolean
  loginError: string | null
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
            ? '登入已失效，請重新從 LINE 開啟此頁。'
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

  const view = gateView({ authenticated, loginError, status })

  // Offer a way out only if loading is unusually slow, so a normal load is
  // just a spinner.
  useEffect(() => {
    setSlow(false)
    if (view !== 'loading') return
    const timer = window.setTimeout(() => setSlow(true), SLOW_LOADING_MS)
    return () => window.clearTimeout(timer)
  }, [view])

  if (view === 'ready') return children

  if (view === 'loading') {
    return (
      <main className={screen} aria-busy="true">
        <div className="mx-auto h-12 w-12 animate-spin rounded-full border-[3px] border-border border-t-primary" />
        <h1 className="text-xl font-semibold">正在載入</h1>
        <p role="status" aria-live="polite" className="text-muted-foreground">
          {authenticated ? '正在確認你的學習資料…' : '正在連線 LINE…'}
        </p>
        {slow && (
          <>
            <p className="text-sm text-muted-foreground">
              載入時間比平常久，可以重新整理再試一次。
            </p>
            <button type="button" className={button} onClick={() => window.location.reload()}>
              重新整理
            </button>
          </>
        )}
      </main>
    )
  }

  if (view === 'login-error' || view === 'error') {
    return (
      <main className={screen}>
        <h1 className="text-xl font-semibold">
          {view === 'login-error' ? '無法完成 LINE 登入' : '暫時無法載入'}
        </h1>
        <p role="alert">{view === 'login-error' ? loginError : error}</p>
        {view === 'error' && (
          <button type="button" className={button} onClick={() => void check()}>
            重新確認
          </button>
        )}
      </main>
    )
  }

  return (
    <main className={screen}>
      <h1 className="text-2xl font-semibold">請先完成實驗登記</h1>
      <p>請回到機器人的 LINE 聊天室，傳送「實驗編號 姓名」，完成後才能使用所有功能。</p>
      <div className="rounded-xl border p-5">
        <p className="text-sm text-muted-foreground">輸入範例（請換成自己的編號與真實姓名）</p>
        <p className="mt-3 text-2xl font-semibold">01 王小明</p>
        <p className="mt-3 text-sm">編號和姓名中間加一個空格即可。</p>
      </div>
      <p role="status" aria-live="polite">
        尚未完成登記。請在 LINE 傳送編號與姓名，不是在此頁填寫。
      </p>
      <button type="button" className={button} onClick={() => void check()}>
        已在 LINE 登記，重新確認
      </button>
    </main>
  )
}
