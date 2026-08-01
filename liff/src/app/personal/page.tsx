'use client'

import { Line, LineChart, Bar, BarChart, CartesianGrid, LabelList, XAxis, YAxis } from 'recharts'
import React, { useEffect, useMemo, useState } from 'react'
import { BarChart3, Columns2 } from 'lucide-react'
import { useLiff } from '../LiffProvider'
import type { GradingDetail, PlaybackResponse, UserData } from '@/types'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

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
import { Select, SelectField } from '@/components/ui/select'
import { Skill, SkillNameMap } from '@/lib/types'
import { fetchUserDataSafe } from '@/lib/api/fetchUserDataSafe'
import { fetchPlayback } from '@/lib/api/fetchPlayback'
import VideoComparison from '@/components/VideoComparison'

const TAB_OPTIONS = [
  { value: 'scores', label: '成績分析', icon: BarChart3 },
  { value: 'comparison', label: '影片比較', icon: Columns2 }
] as const

type TabValue = (typeof TAB_OPTIONS)[number]['value']

interface MovementDetailBarChartProps {
  userData: UserData
  selectedDate: string
  selectedSkill: Skill
}

const MovementDetailBarChart = ({
  userData,
  selectedDate,
  selectedSkill
}: MovementDetailBarChartProps) => {
  // Get grading details for the selected skill and date
  const gradingDetails =
    selectedSkill &&
    selectedDate &&
    userData.portfolio[selectedSkill][selectedDate]?.grading_outcome?.grading_details

  const chartData =
    gradingDetails !== ''
      ? gradingDetails?.map((detail: GradingDetail) => ({
          description: detail.description,
          grade: detail.grade.toFixed(2)
        }))
      : []

  const chartConfig = {
    grade: {
      label: '得分',
      color: 'hsl(var(--chart-1))'
    },
    label: {
      color: 'hsl(var(--primary-foreground))'
    }
  }

  return (
    <Card className="animate-fade-down">
      <CardHeader>
        <CardTitle>動作細節評分</CardTitle>
        <CardDescription>
          {SkillNameMap[selectedSkill]} · {selectedDate}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer config={chartConfig}>
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{
              right: 30
            }}
            width={500}
            height={chartData.length * 50} // Dynamic height based on data
          >
            <CartesianGrid horizontal={false} />
            <YAxis
              dataKey="description"
              type="category"
              tickLine={false}
              tickMargin={10}
              axisLine={false}
              hide
            />
            <XAxis domain={[0, 20]} type="number" hide />
            <ChartTooltip cursor={false} content={<ChartTooltipContent indicator="line" />} />
            <Bar dataKey="grade" layout="vertical" fill="var(--color-grade)" radius={6}>
              <LabelList
                dataKey="description"
                position="insideLeft"
                offset={8}
                className="fill-[--color-label]"
                fontSize={12}
              />
              <LabelList
                dataKey="grade"
                position="right"
                offset={8}
                className="fill-foreground"
                fontSize={12}
              />
            </Bar>
          </BarChart>
        </ChartContainer>
      </CardContent>
    </Card>
  )
}

interface PersonalProgressChartProps {
  userData: UserData
}
const PersonalProgressChart = ({ userData }: PersonalProgressChartProps) => {
  const chartConfig = {
    totalGrade: {
      label: '成績',
      color: 'hsl(var(--chart-1))'
    }
  } satisfies ChartConfig
  const [selectedSkill, setSelectedSkill] = useState<Skill>('serve') // Default skill
  const availableSkills = Object.keys(SkillNameMap) as Skill[]

  return (
    <Card className="animate-fade-down">
      <CardHeader>
        <CardTitle>每週進步</CardTitle>
        <CardDescription>依技能查看歷次總分趨勢</CardDescription>
        <Select
          value={selectedSkill}
          onChange={e => setSelectedSkill(e.target.value as Skill)}
          aria-label="選擇技能"
          className="mt-2 max-w-[12rem]"
        >
          {availableSkills
            .filter(
              skill => userData.portfolio[skill] && Object.keys(userData.portfolio[skill]).length > 0
            ) // Only show skills with records
            .map(skill => (
              <option key={skill} value={skill}>
                {SkillNameMap[skill as keyof typeof SkillNameMap] || skill}
              </option>
            ))}
        </Select>
      </CardHeader>
      <CardContent>
        <ChartContainer config={chartConfig}>
          <LineChart
            accessibilityLayer
            data={Object.keys(userData.portfolio[selectedSkill])
              .sort((a, b) => {
                const [yearA, monthA, dayA, hourA, minuteA] = a.split('-').map(Number)
                const [yearB, monthB, dayB, hourB, minuteB] = b.split('-').map(Number)

                const dateA = new Date(yearA, monthA - 1, dayA, hourA, minuteA) // Adjust month (0-based index)
                const dateB = new Date(yearB, monthB - 1, dayB, hourB, minuteB)

                return dateA.getTime() - dateB.getTime() // Ascending order
              })
              .map(date => ({
                date,
                totalGrade:
                  userData.portfolio[selectedSkill][date].grading_outcome.total_grade.toFixed(2)
              }))}
            width={500}
            height={500}
          >
            <CartesianGrid vertical={false} />
            <XAxis
              dataKey="date"
              type="category"
              tickLine={false}
              axisLine={false}
              tickMargin={9}
              angle={-35}
              tickFormatter={(value: string) => value.slice(5, 10).replace('-', '/')} // Format dates to month
              dx={-8}
              dy={5}
            />
            <YAxis
              tickLine={false}
              axisLine={false}
              tickFormatter={value => `${value}`}
              domain={[0, 100]}
              width={40}
              dx={-10}
            />
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

  if (loading) {
    return <Spinner fullscreen />
  }

  if (!userData) {
    return (
      <PageContainer className="pt-6">
        <Alert variant="warning" title="尚無資料">
          目前查不到您的學習資料，請稍後再試。
        </Alert>
      </PageContainer>
    )
  }

  if (availableSkills.length === 0) {
    return (
      <PageContainer className="pt-6">
        <Alert variant="info" title="尚無動作分析記錄">
          上傳一段練習影片後，這裡就會顯示評分與比較。
        </Alert>
      </PageContainer>
    )
  }

  return (
    <PageContainer className="pt-6">
      <main className="space-y-5">
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
            className="flex-[1.5]"
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

        <Segmented
          role="tablist"
          label="學習歷程檢視"
          options={TAB_OPTIONS}
          value={activeTab}
          onChange={setActiveTab}
        />

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

        {activeTab === 'scores' && (
          <div role="tabpanel" className="space-y-5">
            <MovementDetailBarChart
              userData={userData}
              selectedDate={selectedDate}
              selectedSkill={selectedSkill}
            />
            <PersonalProgressChart userData={userData} />
          </div>
        )}
      </main>
    </PageContainer>
  )
}
