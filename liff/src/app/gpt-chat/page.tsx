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
import { SCORE_RECORD_LABEL, useSkillSummary } from '@/lib/useSkillSummary'
import SkillSummary from '@/components/SkillSummary'
import { authorizedFetch } from '@/lib/api/client'

export default function GptChatPage() {
  const [userData, setUserData] = useState<UserData | null>(null)
  const [loading, setLoading] = useState(true)
  const [selectedSkill, setSelectedSkill] = useState<Skill>('serve')
  const { liff, profile } = useLiff()
  const {
    summary,
    messages: chatHistory,
    loading: summaryLoading,
    error: summaryError
  } = useSkillSummary(profile?.userId, selectedSkill)

  useEffect(() => {
    if (liff) {
      if (!liff.isLoggedIn || !profile) {
        liff.login()
        return
      }
    }

    const fetchData = async () => {
      try {
        const response = await authorizedFetch(`/api/db/user?user_id=${profile?.userId}`)
        if (!response.ok) {
          throw new Error(`Failed to fetch user data: ${response.statusText}`)
        }
        const data = await response.json()
        setUserData(data)
      } catch (err) {
        if (err instanceof Error) console.log(err.message)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [liff, profile])

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
        <SelectField
          label="選擇技能"
          className="max-w-[12rem]"
          value={selectedSkill}
          onChange={e => setSelectedSkill(e.target.value as Skill)}
        >
          {availableSkills.map(skill => (
            <option key={skill} value={skill}>
              {SkillNameMap[skill as keyof typeof SkillNameMap] || skill}
            </option>
          ))}
        </SelectField>

        <SkillSummary
          skill={selectedSkill}
          summary={summary}
          loading={summaryLoading}
          error={summaryError}
        />

        <Card className="animate-fade-down">
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
                  // Not something the learner wrote: it marks where their score
                  // record went to the coach, so it reads as a note, not a turn.
                  if (message.text === SCORE_RECORD_LABEL) {
                    return (
                      <p key={`${message.timestamp || 't'}-${idx}`} className="flex justify-center">
                        <span className="rounded-full bg-muted px-3 py-1 text-[11px] text-muted-foreground">
                          {SCORE_RECORD_LABEL}
                        </span>
                      </p>
                    )
                  }
                  return (
                    <div
                      key={`${message.timestamp || 't'}-${idx}`}
                      className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}
                    >
                      <div
                        className={`max-w-[80%] break-words rounded-xl px-3.5 py-2.5 text-sm leading-6 ${
                          isUser
                            ? 'rounded-br-sm bg-primary text-primary-foreground'
                            : 'rounded-bl-sm bg-muted text-foreground'
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
