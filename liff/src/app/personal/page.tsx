'use client'

import React, { useEffect, useMemo, useState } from 'react'
import { BarChart3, Columns2 } from 'lucide-react'

import { useLiff } from '../LiffProvider'
import type { PlaybackResponse, UserData } from '@/types'
import { Alert } from '@/components/ui/alert'
import { Card } from '@/components/ui/card'
import { PageContainer } from '@/components/ui/page'
import { Segmented } from '@/components/ui/segmented'
import Spinner from '@/components/ui/spinner'
import CriterionList from '@/components/report/CriterionList'
import ScoreBand, { type Session } from '@/components/report/ScoreBand'
import SkillChips from '@/components/report/SkillChips'
import VideoComparison from '@/components/VideoComparison'
import { Skill, SkillNameMap } from '@/lib/types'
import { fetchUserDataSafe } from '@/lib/api/fetchUserDataSafe'
import { fetchPlayback } from '@/lib/api/fetchPlayback'

const TAB_OPTIONS = [
  { value: 'scores', label: '成績分析', icon: BarChart3 },
  { value: 'comparison', label: '影片比較', icon: Columns2 }
] as const

type TabValue = (typeof TAB_OPTIONS)[number]['value']

/** Sessions oldest-first, which is the order the tape reads in. */
const sessionsFor = (userData: UserData, skill: Skill): Session[] =>
  Object.keys(userData.portfolio[skill])
    .sort()
    .map(date => ({
      date,
      grade: userData.portfolio[skill][date].grading_outcome.total_grade
    }))

export default function PersonalPage() {
  const [userData, setUserData] = useState<UserData | null>(null)
  const [loading, setLoading] = useState(true)
  const [selectedSkill, setSelectedSkill] = useState<Skill>('serve')
  const [selectedDate, setSelectedDate] = useState('')
  const [playback, setPlayback] = useState<PlaybackResponse | null>(null)
  const [playbackError, setPlaybackError] = useState('')
  const [playbackLoading, setPlaybackLoading] = useState(false)
  const [activeTab, setActiveTab] = useState<TabValue>('scores')
  const { liff, profile } = useLiff()

  useEffect(() => {
    if (!liff) return
    if (!profile?.userId) return

    const fetchData = async () => {
      try {
        const result = await fetchUserDataSafe(profile.userId)
        if (!result.ok) throw result.error
        const data = result.data
        setUserData(data)
        const firstSkill = (Object.keys(SkillNameMap) as Skill[]).find(
          skill => Object.keys(data.portfolio[skill]).length > 0
        )
        if (firstSkill) {
          setSelectedSkill(firstSkill)
          setSelectedDate(sessionsFor(data, firstSkill).at(-1)?.date ?? '')
        }
      } catch (err) {
        if (err instanceof Error) console.log(err.message)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [liff, profile])

  const availableSkills = useMemo(
    () =>
      userData
        ? (Object.keys(SkillNameMap) as Skill[]).filter(
            skill => Object.keys(userData.portfolio[skill]).length > 0
          )
        : [],
    [userData]
  )

  const sessions = useMemo(
    () => (userData ? sessionsFor(userData, selectedSkill) : []),
    [selectedSkill, userData]
  )

  useEffect(() => {
    if (activeTab !== 'comparison' || !profile?.userId || !selectedDate) {
      setPlayback(null)
      return
    }
    let cancelled = false
    setPlayback(null)
    setPlaybackLoading(true)
    setPlaybackError('')
    fetchPlayback(profile.userId, selectedSkill, selectedDate)
      .then(value => {
        if (!cancelled) setPlayback(value)
      })
      .catch(error => {
        if (!cancelled) {
          setPlayback(null)
          setPlaybackError(error instanceof Error ? error.message : '無法載入影片')
        }
      })
      .finally(() => {
        if (!cancelled) setPlaybackLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [activeTab, profile?.userId, selectedDate, selectedSkill])

  if (loading) return <Spinner fullscreen />

  if (!userData) {
    return (
      <PageContainer className="pt-8">
        <Alert title="查不到學習資料">請從 LINE 重新開啟這個頁面。</Alert>
      </PageContainer>
    )
  }

  if (availableSkills.length === 0) {
    return (
      <PageContainer className="pt-8">
        <Alert title="還沒有動作分析">上傳一段練習影片，分析完成後這裡會顯示評分。</Alert>
      </PageContainer>
    )
  }

  const outcome = userData.portfolio[selectedSkill][selectedDate]?.grading_outcome
  const details = Array.isArray(outcome?.grading_details) ? outcome.grading_details : []

  return (
    <>
      {/* Skill first: it decides everything below it. */}
      <SkillChips
        skills={availableSkills}
        value={selectedSkill}
        onChange={skill => {
          setSelectedSkill(skill)
          setSelectedDate(sessionsFor(userData, skill).at(-1)?.date ?? '')
        }}
      />

      <ScoreBand
        sessions={sessions}
        selectedDate={selectedDate}
        onSelectDate={setSelectedDate}
        skillName={SkillNameMap[selectedSkill]}
      />

      <PageContainer className="pt-5">
        <Segmented
          role="tablist"
          label="檢視方式"
          options={TAB_OPTIONS}
          value={activeTab}
          onChange={setActiveTab}
        />

        {activeTab === 'scores' && (
          <div role="tabpanel" className="pt-5">
            <h2 className="eyebrow">動作細節</h2>
            <Card className="mt-2 px-4">
              <CriterionList details={details} />
            </Card>
          </div>
        )}

        {activeTab === 'comparison' && (
          <div role="tabpanel" className="pt-5">
            {playbackLoading && <Spinner />}
            {!playbackLoading && playback && <VideoComparison playback={playback} />}
            {!playbackLoading && playbackError && (
              <Alert variant="fault" title="無法載入影片">
                {playbackError}
              </Alert>
            )}
          </div>
        )}
      </PageContainer>
    </>
  )
}
