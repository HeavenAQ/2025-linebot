'use client'

import React, { useState, useEffect } from 'react'
import { useLiff } from '../LiffProvider'
import type { UserData } from '@/types'
import { Alert } from '@/components/ui/alert'
import { Card } from '@/components/ui/card'
import { PageContainer } from '@/components/ui/page'
import Spinner from '@/components/ui/spinner'
import SkillChips from '@/components/report/SkillChips'
import { Skill } from '@/lib/types'
import { getBackendBaseUrl } from '@/utils/env'

type ChatMessage = {
  role: string
  text: string
  skill: string
  conversation_id?: string
  timestamp?: string
}

export default function GptChatPage() {
  const [userData, setUserData] = useState<UserData | null>(null)
  const [loading, setLoading] = useState(true)
  const [selectedSkill, setSelectedSkill] = useState<Skill>('serve')
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([])
  const [summary, setSummary] = useState('') // Stores GPT summary
  const { liff, profile } = useLiff()

  useEffect(() => {
    if (liff) {
      if (!liff.isLoggedIn || !profile) {
        liff.login()
        return
      }
    }

    const fetchChatHistory = async (userId: string, skill: string) => {
      setLoading(true)
      setChatHistory([])
      setSummary('')
      try {
        const qs = new URLSearchParams({ user_id: userId, skill })
        const base = getBackendBaseUrl()
        const response = await fetch(`${base}/api/chat/history?${qs.toString()}`)
        if (!response.ok) throw new Error(`Failed to fetch chat history: ${response.statusText}`)

        const json = await response.json()
        const messages: ChatMessage[] = Array.isArray(json.data) ? json.data : []
        setChatHistory(messages)

        // Summarize the latest exchanges only.
        const lastMessages = messages
          .slice(-10)
          .map(m => m.text)
          .filter(Boolean)
        if (lastMessages.length > 0) {
          const body = { content: lastMessages.join('\n'), user_id: userId, skill }
          const sumRes = await fetch(`${base}/api/chat/summarize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
          })
          setSummary(sumRes.ok ? (await sumRes.json()).summary || '' : '')
        } else {
          setSummary('')
        }
      } catch (error) {
        console.error('Error fetching chat history:', error)
      } finally {
        setLoading(false)
      }
    }

    const fetchData = async () => {
      try {
        const base = getBackendBaseUrl()
        const response = await fetch(`${base}/api/db/user?user_id=${profile?.userId}`)
        if (!response.ok) throw new Error(`Failed to fetch user data: ${response.statusText}`)
        const data = await response.json()
        setUserData(data)

        if (profile?.userId) await fetchChatHistory(profile.userId, selectedSkill)
      } catch (err) {
        if (err instanceof Error) console.log(err.message)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [liff, profile, selectedSkill])

  const availableSkills = Object.keys(userData?.gpt_conversation_ids || {}) as Skill[]

  if (loading && !userData) return <Spinner fullscreen />

  if (!userData || availableSkills.length === 0) {
    return (
      <PageContainer className="pt-8">
        <Alert title="還沒有教練對話">在 LINE 裡向教練機器人提問，紀錄與重點會整理到這裡。</Alert>
      </PageContainer>
    )
  }

  return (
    <>
      <SkillChips skills={availableSkills} value={selectedSkill} onChange={setSelectedSkill} />

      <PageContainer className="pt-5">
        <h1 className="eyebrow">重點整理</h1>
        {loading ? (
          <Spinner />
        ) : summary ? (
          <p className="mt-2 whitespace-pre-line text-[15px] leading-7">{summary}</p>
        ) : (
          <p className="mt-2 text-sm text-muted-foreground">這個技能還沒有可整理的對話。</p>
        )}

        <h2 className="eyebrow mt-8">對話紀錄</h2>
        {!loading && chatHistory.length === 0 ? (
          <p className="mt-2 text-sm text-muted-foreground">尚無訊息。</p>
        ) : (
          <div className="mt-3 space-y-2.5">
            {chatHistory.map((message, idx) => {
              const isUser = message.role === 'user'
              return (
                <div
                  key={`${message.timestamp || 't'}-${idx}`}
                  className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}
                >
                  {isUser ? (
                    <p className="max-w-[85%] break-words rounded-lg bg-primary px-3.5 py-2.5 text-sm leading-6 text-primary-foreground">
                      {message.text}
                    </p>
                  ) : (
                    <Card className="max-w-[85%] break-words px-3.5 py-2.5 text-sm leading-6">
                      {message.text}
                    </Card>
                  )}
                </div>
              )
            })}
          </div>
        )}
      </PageContainer>
    </>
  )
}
