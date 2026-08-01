'use client'

import { Line, LineChart, CartesianGrid, XAxis, YAxis } from 'recharts'
import React, { useEffect, useMemo, useState } from 'react'
import { BarChart3, Columns2 } from 'lucide-react'

import { useLiff } from '../LiffProvider'
import type { GradingDetail, PlaybackResponse, UserData } from '@/types'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent
} from '@/components/ui/chart'
import Spinner from '@/components/ui/spinner'
import { Alert } from '@/components/ui/alert'
import { PageContainer } from '@/components/ui/page'
import { Segmented } from '@/components/ui/segmented'
import { SelectField } from '@/components/ui/select'
import { Skill, SkillNameMap } from '@/lib/types'
import { fetchUserDataSafe } from '@/lib/api/fetchUserDataSafe'
import { fetchPlayback } from '@/lib/api/fetchPlayback'
import VideoComparison from '@/components/VideoComparison'

const TAB_OPTIONS = [
  { value: 'scores', label: '成績分析', icon: BarChart3 },
  { value: 'comparison', label: '影片比較', icon: Columns2 }
] as const

type TabValue = (typeof TAB_OPTIONS)[number]['value']

const MAX_CRITERION = 20

const chartConfig = {
  totalGrade: { label: '總分', color: 'hsl(var(--chart-1))' }
} satisfies ChartConfig

/** Sort the "YYYY-MM-DD-HH-mm" keys chronologically. */
const chronological = (a: string, b: string) => {
  const parse = (v: string) => {
    const [y, mo, d, h, mi] = v.split('-').map(Number)
    return new Date(y, (mo ?? 1) - 1, d ?? 1, h ?? 0, mi ?? 0).getTime()
  }
  return parse(a) - parse(b)
}

/**
 * Criteria as labelled rows. The labels are Chinese phrases that never fit
 * inside a bar at phone width, so the name sits above its own bar and the
 * scores align down the right edge.
 */
const Criteria = ({ details }: { details: readonly GradingDetail[] }) => {
  if (details.length === 0) {
    return <p className="text-[13px] text-muted-foreground">這次分析沒有細項評分。</p>
  }

  const weakest = details.reduce((low, d) => (d.grade < low.grade ? d : low), details[0])

  return (
    <ul className="space-y-4">
      {details.map((detail, i) => {
        const ratio = Math.max(0, Math.min(1, detail.grade / MAX_CRITERION))
        const isWeakest = details.length > 1 && detail === weakest
        return (
          <li key={`${detail.description}-${i}`}>
            <div className="flex items-baseline justify-between gap-3">
              <span className="min-w-0 text-sm leading-snug">
                {detail.description}
                {isWeakest && (
                  <span className="ml-2 whitespace-nowrap text-[11px] font-medium text-destructive">
                    最需改進
                  </span>
                )}
              </span>
              <span className="num shrink-0 font-data text-sm font-semibold">
                {detail.grade.toFixed(1)}
                <span className="ml-0.5 text-[11px] font-normal text-muted-foreground">
                  /{MAX_CRITERION}
                </span>
              </span>
            </div>
            <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-muted">
              <div
                className={`h-full rounded-full ${isWeakest ? 'bg-destructive' : 'bg-primary'}`}
                style={{ width: `${ratio * 100}%` }}
              />
            </div>
          </li>
        )
      })}
    </ul>
  )
}

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
    if (!liff || !profile?.userId) return

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
          setSelectedDate(Object.keys(data.portfolio[firstSkill]).sort().reverse()[0] || '')
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

  const availableDates = useMemo(
    () => (userData ? Object.keys(userData.portfolio[selectedSkill]).sort().reverse() : []),
    [selectedSkill, userData]
  )

  /** Oldest first, for the trend line. Driven by the page's skill, not its own. */
  const trend = useMemo(() => {
    if (!userData) return []
    return Object.keys(userData.portfolio[selectedSkill])
      .sort(chronological)
      .map(date => ({
        date,
        totalGrade: Number(
          userData.portfolio[selectedSkill][date].grading_outcome.total_grade.toFixed(2)
        )
      }))
  }, [selectedSkill, userData])

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
      <PageContainer className="pt-6">
        <Alert variant="warning" title="查不到學習資料">
          請從 LINE 重新開啟這個頁面。
        </Alert>
      </PageContainer>
    )
  }

  if (availableSkills.length === 0) {
    return (
      <PageContainer className="pt-6">
        <Alert title="還沒有動作分析">上傳一段練習影片，分析完成後這裡會顯示評分。</Alert>
      </PageContainer>
    )
  }

  const outcome = userData.portfolio[selectedSkill][selectedDate]?.grading_outcome
  const details = Array.isArray(outcome?.grading_details) ? outcome.grading_details : []
  const total = outcome?.total_grade
  const currentIndex = trend.findIndex(t => t.date === selectedDate)
  const previous = currentIndex > 0 ? trend[currentIndex - 1] : undefined
  const delta = previous && total !== undefined ? total - previous.totalGrade : undefined

  return (
    <PageContainer className="pt-6">
      <main className="space-y-6">
        <div className="flex gap-3">
          <SelectField
            label="技能"
            className="flex-1"
            value={selectedSkill}
            onChange={event => {
              const skill = event.target.value as Skill
              setSelectedSkill(skill)
              setSelectedDate(Object.keys(userData.portfolio[skill]).sort().reverse()[0] || '')
            }}
          >
            {availableSkills.map(skill => (
              <option key={skill} value={skill}>
                {SkillNameMap[skill]}
              </option>
            ))}
          </SelectField>
          <SelectField
            label="分析日期"
            className="flex-[1.4]"
            value={selectedDate}
            onChange={event => setSelectedDate(event.target.value)}
          >
            {availableDates.map(date => (
              <option key={date} value={date}>
                {date}
              </option>
            ))}
          </SelectField>
        </div>

        {/* The score sits on the page itself, not in a card — it is the answer,
            not one more item in a list of panels. */}
        {total !== undefined && (
          <div className="flex items-end gap-4 border-b border-border pb-6">
            <p className="num font-data text-figure">{total.toFixed(1)}</p>
            <div className="mb-1.5">
              <p className="text-[13px] text-muted-foreground">總分 / 100</p>
              {delta !== undefined && (
                <p
                  className={`num text-sm font-semibold ${
                    delta >= 0 ? 'text-primary' : 'text-destructive'
                  }`}
                >
                  {delta >= 0 ? '+' : ''}
                  {delta.toFixed(1)}
                  <span className="ml-1.5 font-normal text-muted-foreground">較上次</span>
                </p>
              )}
            </div>
          </div>
        )}

        <Segmented
          role="tablist"
          label="學習歷程檢視"
          options={TAB_OPTIONS}
          value={activeTab}
          onChange={setActiveTab}
        />

        {activeTab === 'scores' && (
          <div role="tabpanel" className="space-y-6">
            <section>
              <h2 className="eyebrow mb-3">動作細節</h2>
              <Criteria details={details} />
            </section>

            {trend.length > 1 && (
              <Card>
                <CardHeader>
                  <CardTitle>{SkillNameMap[selectedSkill]}　歷次總分</CardTitle>
                </CardHeader>
                <CardContent>
                  <ChartContainer config={chartConfig}>
                    <LineChart
                      accessibilityLayer
                      data={trend}
                      width={500}
                      height={500}
                      margin={{ left: -16, right: 10, top: 6 }}
                    >
                      <CartesianGrid vertical={false} strokeDasharray="3 4" />
                      <XAxis
                        dataKey="date"
                        type="category"
                        tickLine={false}
                        axisLine={false}
                        tickMargin={10}
                        tickFormatter={(value: string) => value.slice(5, 10).replace('-', '/')}
                        fontSize={11}
                      />
                      <YAxis tickLine={false} axisLine={false} domain={[0, 100]} fontSize={11} />
                      <ChartTooltip cursor={false} content={<ChartTooltipContent hideLabel />} />
                      <Line
                        dataKey="totalGrade"
                        type="monotone"
                        stroke="var(--color-totalGrade)"
                        strokeWidth={2}
                        dot={{ r: 3, strokeWidth: 0, fill: 'var(--color-totalGrade)' }}
                        activeDot={{ r: 5 }}
                      />
                    </LineChart>
                  </ChartContainer>
                </CardContent>
              </Card>
            )}
          </div>
        )}

        {activeTab === 'comparison' && (
          <div role="tabpanel" className="space-y-5">
            {playbackLoading && <Spinner />}
            {!playbackLoading && playback && <VideoComparison playback={playback} />}
            {!playbackLoading && playbackError && (
              <Alert variant="warning" title="無法載入影片">
                {playbackError}
              </Alert>
            )}
          </div>
        )}
      </main>
    </PageContainer>
  )
}
