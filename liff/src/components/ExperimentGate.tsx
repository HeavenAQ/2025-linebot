'use client'

import React, { useCallback, useEffect, useState, type ReactNode } from 'react'
import {
  authorizedFetch,
  rateLimitedMessage,
  setRegistrationRequiredHandler
} from '@/lib/api/client'
import { hasExperimentRegistration } from '@/lib/registration'

type Status = 'checking' | 'required' | 'ready' | 'error'

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
  const [status, setStatus] = useState<Status>('checking')
  const [error, setError] = useState('')

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

  if (!authenticated || loginError)
    return (
      <main className="mx-auto flex min-h-screen max-w-md flex-col justify-center gap-5 px-6 py-12 text-center">
        <h1 className="text-2xl font-semibold">
          {loginError ? '請重新登入 LINE' : '正在確認 LINE 登入'}
        </h1>
        <p role="status" aria-live="polite">
          {recovering
            ? '正在開啟 LINE 登入…'
            : loginError || '正在取得 LINE 登入身分；若等待太久，請按下方按鈕。'}
        </p>
        <p>已登記的實驗編號與姓名會保留，不需要重新登記。</p>
        <button
          type="button"
          disabled={recovering}
          onClick={onReauthenticate}
          className="min-h-12 rounded-lg bg-primary px-4 py-3 text-primary-foreground disabled:opacity-50"
        >
          重新登入 LINE
        </button>
      </main>
    )
  if (status === 'ready') return children
  return (
    <main className="mx-auto flex min-h-screen max-w-md flex-col justify-center gap-5 px-6 py-12 text-center">
      <h1 className="text-2xl font-semibold">請先完成實驗登記</h1>
      <p>請回到機器人的 LINE 聊天室，傳送「實驗編號 姓名」，完成後才能使用所有功能。</p>
      <div className="rounded-xl border p-5">
        <p className="text-sm text-muted-foreground">輸入範例（請換成自己的編號與真實姓名）</p>
        <p className="mt-3 text-2xl font-semibold">01 王小明</p>
        <p className="mt-3 text-sm">編號和姓名中間加一個空格即可。</p>
      </div>
      <p role="status" aria-live="polite">
        {loginError ||
          (status === 'error'
            ? error
            : status === 'checking'
              ? authenticated
                ? '正在確認登記資料…'
                : '正在確認 LINE 登入…'
              : '尚未完成登記。請在 LINE 傳送編號與姓名，不是在此頁填寫。')}
      </p>
      <button
        type="button"
        disabled={!authenticated || status === 'checking'}
        className="min-h-12 rounded-lg bg-primary px-4 py-3 text-primary-foreground disabled:opacity-50"
        onClick={() => void check()}
      >
        已在 LINE 登記，重新確認
      </button>
    </main>
  )
}
