'use client'

import React, { useState, useEffect } from 'react'
import { useLiff } from '../LiffProvider'
import type { UserData } from '@/types'
import Spinner from '@/components/ui/spinner'
import { Skill, SkillNameMap } from '@/lib/types'

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
        const response = await fetch(`/api/chat/history?${qs.toString()}`)
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
          const sumRes = await fetch('/api/chat/summarize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
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
        const response = await fetch(`/api/db/user?user_id=${profile?.userId}`)
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
    return <Spinner />
  }

  if (!userData) {
    return <div className="text-center mt-10 font-semibold text-red-400 border border-red-500 max-w-72 mx-auto rounded-lg h-32 pt-12">
      目前尚無聊天紀錄
    </div>
  }

  return (
    <div className="mx-auto mt-10 w-10/12 max-w-[800px] duration-200 fade-in">
      {/* User Thread Summary */}
      <h1 className="h1-heading mb-4">動作學習總結</h1>
      <section className="mb-10 rounded-lg border border-gray-300 p-5">
        <div className="mb-5">
          <label className="mr-3 font-semibold">選擇技能：</label>
          <select
            value={selectedSkill}
            onChange={e => {
              setSelectedSkill(e.target.value as Skill)
            }}
            className="rounded border px-2 py-1 font-semibold text-zinc-700"
          >
            {availableSkills.map(skill => (
              <option key={skill} value={skill}>
                {SkillNameMap[skill as keyof typeof SkillNameMap] || skill}
              </option>
            ))}{' '}
          </select>
        </div>
        <hr className="my-5 border-gray-300" />
        <div className="space-y-4">
          <div className="font-semibold">總結：</div>
          <div className="break-words">{summary}</div>
        </div>
      </section>

      {/* Chat History */}
      <h1 className="h1-heading mb-4">聊天記錄</h1>
      <section className="mb-10 h-[80vh] overflow-y-scroll rounded-lg border border-gray-300 p-5">
        <div className="space-y-4">
          {chatHistory.map((message, idx) => (
            <div
              key={`${message.timestamp || 't'}-${idx}`}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`rounded-lg p-3 ${message.role === 'user' ? 'bg-teal-600 text-white' : 'bg-gray-200 text-black'
                  } max-w-xs break-words`}
              >
                {message.text}
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}
