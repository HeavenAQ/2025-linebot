'use client'
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from 'recharts'
import React, { useEffect, useMemo, useState } from 'react'

import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent
} from '@/components/ui/chart'
import { Alert } from '@/components/ui/alert'
import { Card } from '@/components/ui/card'
import { PageContainer } from '@/components/ui/page'
import Spinner from '@/components/ui/spinner'
import SkillChips from '@/components/report/SkillChips'
import { Skill, SkillNameMap } from '@/lib/types'
import { useLiff } from '../LiffProvider'
import { fetchClassStats, fetchUserStats } from '@/lib/api/fetchStats'
import type { StatsByDate } from '@/schemas/stats.schema'

const chartConfig = {
  personalTotalGrade: { label: '你', color: 'hsl(var(--chart-1))' },
  classTotalGrade: { label: '班級平均', color: 'hsl(var(--muted-foreground))' }
} satisfies ChartConfig

const ALL_SKILLS = Object.keys(SkillNameMap) as Skill[]

export default function ClassPage() {
  const { liff, profile } = useLiff()
  const [selectedSkill, setSelectedSkill] = useState<Skill>('serve')
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
          setClassStatsByDate(clsRes.status === 'fulfilled' ? clsRes.value : null)
          setPersonalStatsByDate(usrRes.status === 'fulfilled' ? usrRes.value : null)
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
    // Backend-provided dates; class timeline is the base when present.
    const baseDatesRaw =
      (classStatsByDate && Object.keys(classStatsByDate)) ||
      (personalStatsByDate && Object.keys(personalStatsByDate)) ||
      []
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

  // Latest point where both series exist, so the gap statement is comparable.
  const latestPair = useMemo(
    () =>
      [...chartData]
        .reverse()
        .find(d => d.personalTotalGrade !== undefined && d.classTotalGrade !== undefined),
    [chartData]
  )
  const gap =
    latestPair && latestPair.personalTotalGrade! - latestPair.classTotalGrade!

  if (!liff || !profile) {
    return (
      <PageContainer className="pt-8">
        <Alert title="尚未取得個人資料">請從 LINE 重新開啟這個頁面。</Alert>
      </PageContainer>
    )
  }

  return (
    <>
      <SkillChips skills={ALL_SKILLS} value={selectedSkill} onChange={setSelectedSkill} />

      <PageContainer className="pt-5">
        <h1 className="eyebrow">班級對照</h1>
        <p className="mt-1 text-sm leading-6 text-muted-foreground">
          你與班級平均在{SkillNameMap[selectedSkill]}上的最近六次表現。
        </p>

        {loading ? (
          <Spinner />
        ) : chartData.length === 0 ? (
          <Alert className="mt-4" title="這個技能還沒有班級資料">
            等班上累積幾次分析後就會出現對照曲線。
          </Alert>
        ) : (
          <>
            {gap !== undefined && (
              <p className="num mt-4 font-data text-metric">
                {gap >= 0 ? '+' : ''}
                {gap.toFixed(1)}
                <span className="ml-2 font-sans text-sm font-medium text-muted-foreground">
                  {gap >= 0 ? '高於班級平均' : '低於班級平均'}
                </span>
              </p>
            )}

            <Card className="mt-4 p-4">
              <div className="mb-3 flex items-center gap-4 text-xs font-medium">
                <span className="flex items-center gap-1.5">
                  <span className="h-0.5 w-4 bg-[hsl(var(--chart-1))]" />你
                </span>
                <span className="flex items-center gap-1.5 text-muted-foreground">
                  <span className="h-0.5 w-4 bg-[hsl(var(--muted-foreground))]" />
                  班級平均
                </span>
              </div>
              <ChartContainer config={chartConfig}>
                <LineChart accessibilityLayer data={chartData} width={500} height={500}>
                  <CartesianGrid vertical={false} strokeDasharray="2 4" />
                  <XAxis
                    dataKey="date"
                    type="category"
                    tickLine={false}
                    axisLine={false}
                    tickMargin={10}
                    tickFormatter={(value: string) => value.slice(5, 10).replace('-', '/')}
                    fontSize={11}
                  />
                  <YAxis
                    tickLine={false}
                    axisLine={false}
                    domain={[0, 100]}
                    width={30}
                    fontSize={11}
                  />
                  <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
                  <Line
                    dataKey="classTotalGrade"
                    type="monotone"
                    stroke="var(--color-classTotalGrade)"
                    strokeWidth={1.5}
                    strokeDasharray="3 3"
                    dot={false}
                  />
                  <Line
                    dataKey="personalTotalGrade"
                    type="monotone"
                    stroke="var(--color-personalTotalGrade)"
                    strokeWidth={2.5}
                    dot={{ r: 3, strokeWidth: 0, fill: 'var(--color-personalTotalGrade)' }}
                    activeDot={{ r: 5 }}
                  />
                </LineChart>
              </ChartContainer>
            </Card>
          </>
        )}
      </PageContainer>
    </>
  )
}
