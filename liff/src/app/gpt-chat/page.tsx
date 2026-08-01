'use client'

import React, { useState, useEffect } from 'react'
import { useLiff } from '../LiffProvider'
import type { UserData } from '@/types'
import { Alert } from '@/components/ui/alert'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { PageContainer } from '@/components/ui/page'
import { SelectField } from '@/components/ui/select'
import Spinner from '@/components/ui/spinner'
import { Skill, SkillNameMap } from '@/lib/types'
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

        // Prepare latest up to 10 messages' content for summarization
        const lastMessages = messages
          .slice(-10)
          .map(m => m.text)
          .filter(Boolean)
        if (lastMessages.length > 0) {
          const body = { content: lastMessages.join('\n'), user_id: userId, skill }
          const base = getBackendBaseUrl()
          const sumRes = await fetch(`${base}/api/chat/summarize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
          })
          if (sumRes.ok) {
            const sumJson = await sumRes.json()
            setSummary(sumJson.summary || '')
          } else {
            setSummary('')
          }
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
        if (!response.ok) {
          throw new Error(`Failed to fetch user data: ${response.statusText}`)
        }
        const data = await response.json()
        setUserData(data)

        // Load chat history for the selected skill
        if (profile?.userId) {
          await fetchChatHistory(profile.userId, selectedSkill)
        }
      } catch (err) {
        if (err instanceof Error) console.log(err.message)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [liff, profile, selectedSkill])

  const availableSkills = Object.keys(userData?.gpt_conversation_ids || {}) as Skill[]

  if (loading) {
    return <Spinner fullscreen />
  }

  if (!userData) {
    return (
      <PageContainer className="pt-6">
        <Alert variant="info" title="目前尚無聊天紀錄">
          與教練機器人對話後，摘要與紀錄會顯示在這裡。
        </Alert>
      </PageContainer>
    )
  }

  return (
    <PageContainer className="pt-6">
      <main className="space-y-5">
        <Card className="enter">
          <CardHeader>
            <CardTitle>動作學習總結</CardTitle>
            <SelectField
              label="選擇技能"
              className="mt-2 max-w-[12rem]"
              value={selectedSkill}
              onChange={e => setSelectedSkill(e.target.value as Skill)}
            >
              {availableSkills.map(skill => (
                <option key={skill} value={skill}>
                  {SkillNameMap[skill as keyof typeof SkillNameMap] || skill}
                </option>
              ))}
            </SelectField>
          </CardHeader>
          <CardContent>
            {summary ? (
              <p className="whitespace-pre-line break-words text-sm leading-7">{summary}</p>
            ) : (
              <p className="text-sm text-muted-foreground">這個技能還沒有可摘要的對話。</p>
            )}
          </CardContent>
        </Card>

        <Card className="enter">
          <CardHeader>
            <CardTitle>聊天記錄</CardTitle>
          </CardHeader>
          <CardContent>
            {chatHistory.length === 0 ? (
              <p className="text-sm text-muted-foreground">尚無訊息。</p>
            ) : (
              <div className="max-h-[60vh] space-y-3 overflow-y-auto pr-1">
                {chatHistory.map((message, idx) => {
                  const isUser = message.role === 'user'
                  return (
                    <div
                      key={`${message.timestamp || 't'}-${idx}`}
                      className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}
                    >
                      <div
                        className={`max-w-[80%] break-words px-4 py-3 text-[13px] leading-7 ${
                          isUser
                            ? 'bg-primary text-primary-foreground'
                            : 'border border-border bg-card text-foreground'
                        }`}
                      >
                        {message.text}
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </CardContent>
        </Card>
      </main>
    </PageContainer>
  )
}
