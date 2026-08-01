'use client'
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from 'recharts'
import React, { useEffect, useMemo, useState } from 'react'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

import {
  ChartConfig,
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent
} from '@/components/ui/chart'
import { Alert } from '@/components/ui/alert'
import { PageContainer } from '@/components/ui/page'
import { Select } from '@/components/ui/select'
import Spinner from '@/components/ui/spinner'
import { Skill, SkillNameMap } from '@/lib/types'
import { useLiff } from '../LiffProvider'
import { fetchClassStats, fetchUserStats } from '@/lib/api/fetchStats'
import type { StatsByDate } from '@/schemas/stats.schema'

const ClassProgressChart = () => {
  const chartConfig = {
    classTotalGrade: {
      label: '班級',
      color: 'hsl(var(--chart-1))'
    },
    personalTotalGrade: {
      label: '個人',
      color: 'hsl(var(--chart-2))'
    }
  } satisfies ChartConfig
  const [selectedSkill, setSelectedSkill] = useState<Skill>('serve')
  const { profile } = useLiff()
  const [loading, setLoading] = useState(true)
  const [classStatsByDate, setClassStatsByDate] = useState<StatsByDate | null>(null)
  const [personalStatsByDate, setPersonalStatsByDate] = useState<StatsByDate | null>(null)

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      if (!profile?.userId) return
      setLoading(true)
      try {
        const [clsRes, usrRes] = await Promise.allSettled([
          fetchClassStats(selectedSkill),
          fetchUserStats(profile.userId, selectedSkill)
        ])
        if (!cancelled) {
          if (clsRes.status === 'fulfilled') setClassStatsByDate(clsRes.value)
          else setClassStatsByDate(null)
          if (usrRes.status === 'fulfilled') setPersonalStatsByDate(usrRes.value)
          else setPersonalStatsByDate(null)
        }
      } catch (e) {
        console.error(e)
        if (!cancelled) {
          setClassStatsByDate(null)
          setPersonalStatsByDate(null)
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    load()
    return () => {
      cancelled = true
    }
  }, [profile?.userId, selectedSkill])

  const chartData = useMemo(() => {
    // Use backend-provided dates. Prefer class dates as base timeline.
    const baseDatesRaw =
      (classStatsByDate && Object.keys(classStatsByDate)) ||
      (personalStatsByDate && Object.keys(personalStatsByDate)) ||
      []
    // Sort ascending (YYYY-MM-DD sorts chronologically) and take the latest 6
    const dates = baseDatesRaw.sort().slice(-6)
    return dates.map(date => ({
      date,
      classTotalGrade:
        classStatsByDate?.[date]?.avg !== undefined
          ? Number(classStatsByDate[date]!.avg.toFixed(2))
          : undefined,
      personalTotalGrade:
        personalStatsByDate?.[date]?.avg !== undefined
          ? Number(personalStatsByDate[date]!.avg.toFixed(2))
          : undefined
    }))
  }, [classStatsByDate, personalStatsByDate])

  return (
    <Card className="enter">
      <CardHeader>
        <CardTitle>班級學習概況</CardTitle>
        <CardDescription>比較個人成績與班級平均</CardDescription>
        <Select
          value={selectedSkill}
          onChange={e => setSelectedSkill(e.target.value as Skill)}
          aria-label="選擇技能"
          className="mt-2 max-w-[12rem]"
        >
          {Object.keys(SkillNameMap).map(skill => (
            <option key={skill} value={skill}>
              {SkillNameMap[skill as keyof typeof SkillNameMap] || skill}
            </option>
          ))}
        </Select>
      </CardHeader>
      <CardContent>
        {loading ? (
          <Spinner />
        ) : chartData.length === 0 ? (
          <Alert variant="info" title="尚無班級資料">
            這個技能還沒有累積足夠的分析記錄。
          </Alert>
        ) : (
          <ChartContainer config={chartConfig}>
            <LineChart accessibilityLayer data={chartData} width={500} height={500}>
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
              <ChartLegend content={<ChartLegendContent />} />
              <Line
                dataKey="classTotalGrade"
                type="monotone"
                stroke="var(--color-classTotalGrade)"
                strokeWidth={2}
                dot={{ r: 3, strokeWidth: 0, fill: 'var(--color-classTotalGrade)' }}
                activeDot={{ r: 5 }}
              />
              <Line
                dataKey="personalTotalGrade"
                type="monotone"
                stroke="var(--color-personalTotalGrade)"
                strokeWidth={2}
                dot={{ r: 3, strokeWidth: 0, fill: 'var(--color-personalTotalGrade)' }}
                activeDot={{ r: 5 }}
              />
            </LineChart>
          </ChartContainer>
        )}
      </CardContent>
    </Card>
  )
}

export default function ClassPage() {
  const { liff, profile } = useLiff()

  if (!liff || !profile) {
    return (
      <PageContainer className="pt-6">
        <Alert variant="warning" title="尚未取得個人資料">
          請重新從 LINE 開啟此頁面。
        </Alert>
      </PageContainer>
    )
  }

  return (
    <PageContainer className="pt-6">
      <main>
        <ClassProgressChart />
      </main>
    </PageContainer>
  )
}
