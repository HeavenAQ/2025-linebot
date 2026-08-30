'use client'

import { Line, LineChart, CartesianGrid, XAxis, YAxis } from 'recharts'
import React, { useEffect, useMemo, useState } from 'react'
import { BarChart3, Columns2, NotebookPen } from 'lucide-react'

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
import { useSkillSummary } from '@/lib/useSkillSummary'
import { resolveWorkFocus, type WorkFocus } from '@/lib/workLink'
import SkillSummary from '@/components/SkillSummary'
import WeeklyReview from '@/components/WeeklyReview'
import VideoComparison from '@/components/VideoComparison'

const TAB_OPTIONS = [
  { value: 'scores', label: '成績分析', icon: BarChart3 },
  { value: 'comparison', label: '影片比較', icon: Columns2 },
  { value: 'review', label: '每週回顧', icon: NotebookPen }
] as const

type TabValue = (typeof TAB_OPTIONS)[number]['value']

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

  // Weakest by share of its own maximum, not by raw score: 5.0 out of 5 is full
  // marks while 4.0 out of 30 is the problem, and comparing the numbers alone
  // would say the opposite.
  const share = (d: GradingDetail) => d.grade / (d.maximum > 0 ? d.maximum : 20)
  const weakest = details.reduce((low, d) => (share(d) < share(low) ? d : low), details[0])
  // "Weakest" is only useful coaching language when the criterion actually
  // needs work.  Without this gate an all-perfect attempt still painted one
  // row red and called it 最需改進 merely because it was first in the list.
  const weakestNeedsImprovement = share(weakest) < 0.8

  return (
    <ul className="space-y-4">
      {details.map((detail, i) => {
        // Each criterion carries its own maximum -- serve alone runs 5, 5, 30,
        // 10, 30, 20 -- so a shared cap both misdrew the bars and printed
        // impossible scores like 30.0/20.
        const maximum = detail.maximum > 0 ? detail.maximum : 20
        const ratio = Math.max(0, Math.min(1, detail.grade / maximum))
        const isWeakest = details.length > 1 && detail === weakest && weakestNeedsImprovement
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
                  /{maximum}
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
  const [userDataError, setUserDataError] = useState('')
  const [loading, setLoading] = useState(true)
  const [selectedSkill, setSelectedSkill] = useState<Skill>('serve')
  const [selectedDate, setSelectedDate] = useState('')
  const [playback, setPlayback] = useState<PlaybackResponse | null>(null)
  const [playbackError, setPlaybackError] = useState('')
  const [playbackLoading, setPlaybackLoading] = useState(false)
  const [activeTab, setActiveTab] = useState<TabValue>('scores')
  const [focusWork, setFocusWork] = useState<WorkFocus | null>(null)
  const { liff, profile, liffError, sessionExpired } = useLiff()
  const aiSummary = useSkillSummary(profile?.userId, selectedSkill)

  // The bot links straight to a tab (?tab=review from 每週回顧), and a portfolio
  // card names the attempt it shows as well, so a student arriving from LINE
  // lands where they were sent rather than on the default. The attempt can only
  // be checked once the portfolio is here, which is why this waits for userData
  // instead of running on mount.
  useEffect(() => {
    if (typeof window === 'undefined') return
    const search = window.location.search
    const requested = new URLSearchParams(search).get('tab')
    if (TAB_OPTIONS.some(option => option.value === requested)) {
      setActiveTab(requested as TabValue)
    }
    if (!userData) return
    const focus = resolveWorkFocus(search, userData.portfolio)
    if (!focus) return
    setSelectedSkill(focus.skill)
    setSelectedDate(focus.date)
    setFocusWork(focus)
  }, [userData])

  useEffect(() => {
    if (liffError) {
      setUserDataError('LINE 登入失敗，請重新整理頁面後再試一次。')
      setLoading(false)
      return
    }
    if (!liff || !profile?.userId) return

    const fetchData = async () => {
      try {
        setUserDataError('')
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
        if (err instanceof Error) console.error(err.message)
        setUserDataError('無法讀取帳戶資料，請稍後再試。')
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [liff, liffError, profile])

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
        <Alert
          variant="warning"
          title={sessionExpired ? '登入已逾時' : liffError ? 'LINE 登入失敗' : '無法載入學習資料'}
        >
          {sessionExpired
            ? // Reloading will not help: LIFF hands back the same expired token
              // until the app is opened from LINE again.
              '這個頁面開太久了，請從 LINE 重新開啟一次。'
            : userDataError || '請重新整理頁面後再試一次。'}
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
                {date} ·{' '}
                {userData.portfolio[selectedSkill][date].handedness === 'left' ? '左手' : '右手'}
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

        <SkillSummary
          skill={selectedSkill}
          summary={aiSummary.summary}
          loading={aiSummary.loading}
          error={aiSummary.error}
        />

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

        {activeTab === 'review' && profile?.userId && (
          <div role="tabpanel">
            <WeeklyReview userId={profile.userId} userData={userData} focusWork={focusWork} />
          </div>
        )}
      </main>
    </PageContainer>
  )
}
