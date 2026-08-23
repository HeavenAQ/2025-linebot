'use client'

import { useEffect, useState } from 'react'

import type { Skill } from '@/lib/types'
import { authorizedFetch } from '@/lib/api/client'

/**
 * The label the backend puts where a raw score payload used to be.
 *
 * Kept in step with `db.ScoreRecordLabel` in the Go server, which does the
 * actual redaction; this is only how the web app recognises one to show it as a
 * record rather than as something the learner typed.
 */
export const SCORE_RECORD_LABEL = '我的動作評分紀錄'

export interface ChatMessage {
  role: string
  text: string
  skill: string
  conversation_id?: string
  timestamp?: string
}

interface SkillSummary {
  summary: string
  messages: ChatMessage[]
  loading: boolean
  error: string
}

/** How much of the conversation the summary is allowed to draw on. */
const SUMMARY_MESSAGE_LIMIT = 10

/**
 * The learner's AI summary for one skill, plus the conversation it came from.
 *
 * Both the landing page and the chat page show this summary, and the backend
 * caches one per learner per day — so they have to ask for it identically or
 * they would take turns overwriting each other's cached answer. Sharing the
 * request is the point of this hook; the messages come back too, so the chat
 * page does not have to fetch the same history twice.
 */
export function useSkillSummary(userId: string | undefined, skill: Skill): SkillSummary {
  const [summary, setSummary] = useState('')
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!userId || !skill) return
    let cancelled = false
    setLoading(true)
    setError('')
    setSummary('')
    setMessages([])

    const run = async () => {
      const query = new URLSearchParams({ user_id: userId, skill })
      const historyResponse = await authorizedFetch(`/api/chat/history?${query.toString()}`)
      if (!historyResponse.ok) throw new Error(historyResponse.statusText)
      const historyJson = await historyResponse.json()
      const history: ChatMessage[] = Array.isArray(historyJson.data) ? historyJson.data : []
      if (cancelled) return
      setMessages(history)

      const content = history
        .slice(-SUMMARY_MESSAGE_LIMIT)
        .map(message => message.text)
        .filter(Boolean)
        .join('\n')
      const summaryResponse = await authorizedFetch(`/api/chat/summarize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content, user_id: userId, skill })
      })
      if (!summaryResponse.ok) throw new Error(summaryResponse.statusText)
      const summaryJson = await summaryResponse.json()
      if (!cancelled) setSummary(summaryJson.summary || '')
    }

    run()
      .catch(() => {
        if (!cancelled) setError('目前無法取得 AI 總結，請稍後再試。')
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [skill, userId])

  return { summary, messages, loading, error }
}
