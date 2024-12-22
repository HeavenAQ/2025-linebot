'use client'

import React, { useState, useEffect, Dispatch, SetStateAction } from 'react'
import { useLiff } from '../../app/LiffProvider'
import { UserData } from '../api/db/getUserData/types'
import Spinner from '@/components/ui/spinner'
import { Skill, SkillNameMap } from '@/lib/types'

interface Message {
  id: string
  role: string
  text: string
  content: { text: { value: string } }[]
}

const openaiApiKey = process.env.NEXT_PUBLIC_OPENAI_API_KEY || ''
const assistantId = process.env.NEXT_PUBLIC_OPENAI_ASSISTANT_ID || ''

const summarizeWithGPT = async (
  userProfileID: string | undefined,
  content: string,
  setSummary: Dispatch<SetStateAction<string>>
) => {
  if (!content) {
    setSummary('No content to summarize.')
    return
  }

  if (!userProfileID) {
    setSummary('No profile ID available.')
    return
  }

  const today = new Date().toISOString().split('T')[0] // Format: YYYY-MM-DD

  try {
    // Check Firestore for an existing summary
    const existingSummary = await fetchExistingSummary(userProfileID, today)
    if (existingSummary) {
      setSummary(existingSummary)
      console.log('Existing summary found.')
      return
    }

    console.log('No existing summary found. Proceeding to GPT.')

    // Generate a new summary using GPT
    const newSummary = await generateSummaryWithGPT(content)

    // Save the summary to Firestore
    if (newSummary) {
      await saveSummaryToFirestore(userProfileID, today, newSummary)
      setSummary(newSummary)
    } else {
      setSummary('No summarized response available.')
    }
  } catch (error) {
    console.error('Error summarizing with GPT:', error)
    setSummary('Error retrieving summary.')
  }
}

// --- Helper Functions ---

/**
 * Fetch existing summary from Firestore via API
 */
const fetchExistingSummary = async (
  userProfileID: string,
  date: string
): Promise<string | null> => {
  try {
    const response = await fetch(`/api/db/gptSummaries?profileId=${userProfileID}&date=${date}`, {
      method: 'GET'
    })

    if (!response.ok) {
      throw new Error('Error fetching existing summary')
    }

    const data = await response.json()
    return data.summary || null
  } catch (error) {
    console.error('Error fetching existing summary:', error)
    return null
  }
}

/**
 * Generate a new summary using OpenAI's GPT
 */
const generateSummaryWithGPT = async (content: string): Promise<string | null> => {
  try {
    // Step 1: Create a new thread
    const threadId = await createThread()

    // Step 2: Add a message to the thread
    await addMessageToThread(
      threadId,
      `Summarize this learning summary under 100 words: ${content}`
    )

    // Step 3: Run the thread
    const runId = await runThread(threadId)

    // Step 4: Wait for the run to complete
    await pollRunStatus(threadId, runId)

    // Step 5: Fetch the summarized message
    return await fetchSummarizedMessage(threadId)
  } catch (error) {
    console.error('Error generating summary with GPT:', error)
    return null
  }
}

/**
 * Save the summary to Firestore via API
 */
const saveSummaryToFirestore = async (userProfileID: string, date: string, summary: string) => {
  try {
    await fetch('/api/db/gptSummaries', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ userProfileID, date, summary })
    })
    console.log('Summary saved to Firestore.')
  } catch (error) {
    console.error('Error saving summary to Firestore:', error)
  }
}

// --- GPT API Helper Functions ---

/**
 * Create a new thread
 */
const createThread = async (): Promise<string> => {
  const response = await fetch('https://api.openai.com/v1/threads', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${openaiApiKey}`,
      'OpenAI-Beta': 'assistants=v2'
    },
    body: JSON.stringify({})
  })

  if (!response.ok) {
    throw new Error(`Failed to create thread: ${response.statusText}`)
  }

  const data = await response.json()
  return data.id
}

/**
 * Add a message to the thread
 */
const addMessageToThread = async (threadId: string, content: string) => {
  const response = await fetch(`https://api.openai.com/v1/threads/${threadId}/messages`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${openaiApiKey}`,
      'OpenAI-Beta': 'assistants=v2'
    },
    body: JSON.stringify({ role: 'user', content })
  })

  if (!response.ok) {
    throw new Error(`Failed to add message: ${response.statusText}`)
  }
}

/**
 * Run the thread
 */
const runThread = async (threadId: string): Promise<string> => {
  const response = await fetch(`https://api.openai.com/v1/threads/${threadId}/runs`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${openaiApiKey}`,
      'OpenAI-Beta': 'assistants=v2'
    },
    body: JSON.stringify({ assistant_id: assistantId })
  })

  if (!response.ok) {
    throw new Error(`Failed to start run: ${response.statusText}`)
  }

  const data = await response.json()
  return data.id
}

/**
 * Poll the run step for completion status
 */
const pollRunStatus = async (threadId: string, runId: string) => {
  const startTime = Date.now()
  const timeout = 30 * 1000 // 30 seconds

  while (Date.now() - startTime < timeout) {
    const response = await fetch(`https://api.openai.com/v1/threads/${threadId}/runs/${runId}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${openaiApiKey}`,
        'OpenAI-Beta': 'assistants=v2'
      }
    })

    if (!response.ok) {
      throw new Error(`Failed to check step status: ${response.statusText}`)
    }

    const data = await response.json()

    if (data.status === 'completed') {
      return
    }

    if (data.status === 'failed') {
      throw new Error('Run step failed.')
    }

    await new Promise(resolve => setTimeout(resolve, 1000)) // Wait 1 second before retrying
  }

  throw new Error('Run step timeout.')
}

/**
 * Fetch the summarized message
 */
const fetchSummarizedMessage = async (threadId: string): Promise<string | null> => {
  const response = await fetch(`https://api.openai.com/v1/threads/${threadId}/messages`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${openaiApiKey}`,
      'OpenAI-Beta': 'assistants=v2'
    }
  })

  if (!response.ok) {
    throw new Error(`Failed to fetch messages: ${response.statusText}`)
  }

  const data = await response.json()
  return data.data[0]?.content[0]?.text?.value || null
}
export default function GptChatPage() {
  const [userData, setUserData] = useState<UserData | null>(null)
  const [loading, setLoading] = useState(true)
  const [selectedSkill, setSelectedSkill] = useState('Serve') // Default skill
  const [chatHistory, setChatHistory] = useState([]) // Stores GPT chat history
  const [summary, setSummary] = useState('') // Stores GPT summary
  const { liff, profile } = useLiff()

  useEffect(() => {
    if (liff) {
      if (!liff.isLoggedIn || !profile) {
        liff.login()
        return
      }
    }

    const fetchChatHistory = async (threadID: string) => {
      setLoading(true)
      setChatHistory([]) // Clear existing chat history
      setSummary('') // Clear existing summary
      try {
        const response = await fetch(`https://api.openai.com/v1/threads/${threadID}/messages`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${openaiApiKey}`,
            'OpenAI-Beta': 'assistants=v2'
          }
        })

        if (!response.ok) {
          throw new Error(`Failed to fetch chat history: ${response.statusText}`)
        }

        const data = await response.json()

        // Map the messages to a format for the UI
        const messages = data.data.reverse().map((message: Message) => ({
          id: message.id,
          role: message.role,
          text: message.content[0]?.text?.value || ''
        }))

        setChatHistory(messages)

        // Generate summary from assistant messages
        const assistantMessages = messages
          .filter((message: Message) => message.role !== 'user')
          .map((message: Message) => message.text)

        const generatedSummary = assistantMessages.join(' ')
        setSummary(generatedSummary || 'No summary available.')

        // Summarize with GPT
        if (generatedSummary) {
          await summarizeWithGPT(profile?.userId, generatedSummary, setSummary)
        }
      } catch (error) {
        console.error('Error fetching chat history:', error)
      } finally {
        setLoading(false)
      }
    }

    const fetchData = async () => {
      try {
        const response = await fetch(`/api/db/getUserData?userId=${profile?.userId}`)
        if (!response.ok) {
          throw new Error(`Failed to fetch user data: ${response.statusText}`)
        }
        const data = await response.json()
        setUserData(data)

        // Automatically load chat history for the default skill
        const threadID = data.GPTThreadIDs?.[selectedSkill]
        if (threadID) {
          await fetchChatHistory(threadID)
        }
      } catch (err) {
        if (err instanceof Error) console.log(err.message)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [liff, profile, selectedSkill])

  const availableSkills = Object.keys(userData?.GPTThreadIDs || {})

  if (loading) {
    return <Spinner />
  }

  if (!userData) {
    return <div className="text-center">No data available</div>
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
          {chatHistory.map((message: Message) => (
            <div
              key={message.id}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`rounded-lg p-3 ${
                  message.role === 'user' ? 'bg-teal-600 text-white' : 'bg-gray-200 text-black'
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
