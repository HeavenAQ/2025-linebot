'use client'
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from 'recharts'
import React, { useEffect, useMemo, useState } from 'react'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent
} from '@/components/ui/chart'
import { mPlusRounded1c } from '@/components/Fonts/M_PLUS_Rounded_1c'
import { Skill, SkillNameMap } from '@/lib/types'
import Spinner from '@/components/ui/spinner'
import { useLiff } from '../LiffProvider'
import { fetchClassStats, fetchUserStats } from '@/lib/api/fetchStats'
import type { StatsByDate } from '@/schemas/stats.schema'


const ClassProgressChart = ({ onLoadingChange }: { onLoadingChange: (_: boolean) => void }) => {
  const chartConfig = {
    classTotalGrade: {
      label: '班級',
      color: 'hsl(var(--chart-1))'
    },
    personalTotalGrade: {
      label: '個人',
      color: 'hsl(var(--chart-5))'
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
      onLoadingChange(true)
      try {
        const [clsRes, usrRes] = await Promise.allSettled([
          fetchClassStats(selectedSkill),
          fetchUserStats(profile.userId, selectedSkill),
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
        if (!cancelled) {
          setLoading(false)
          onLoadingChange(false)
        }
      }
    }
    load()
    return () => {
      cancelled = true
    }
  }, [profile?.userId, selectedSkill])

  const chartData = useMemo(() => {
    // Use backend-provided dates. Prefer class dates as base timeline.
    const baseDatesRaw = (classStatsByDate && Object.keys(classStatsByDate))
      || (personalStatsByDate && Object.keys(personalStatsByDate))
      || []
    // Sort ascending (YYYY-MM-DD sorts chronologically) and take the latest 6
    const dates = baseDatesRaw.sort().slice(-6)
    return dates.map(date => ({
      date,
      classTotalGrade: classStatsByDate?.[date]?.avg !== undefined
        ? Number(classStatsByDate[date]!.avg.toFixed(2))
        : undefined,
      personalTotalGrade: personalStatsByDate?.[date]?.avg !== undefined
        ? Number(personalStatsByDate[date]!.avg.toFixed(2))
        : undefined,
    }))
  }, [classStatsByDate, personalStatsByDate])

  return (
    <Card
      className={`${mPlusRounded1c.className} mx-auto mb-10 mt-5 w-10/12 max-w-[800px] animate-fade-down`}
    >
      <CardHeader>
        <CardTitle className="mb-3">班級學習概況</CardTitle>
        <CardDescription>
          <div>
            <select
              value={selectedSkill}
              onChange={e => {
                setSelectedSkill(e.target.value as Skill)
              }}
              className="rounded border px-2 py-1 text-zinc-700"
            >
              {Object.keys(SkillNameMap).map(skill => (
                <option key={skill} value={skill}>
                  {SkillNameMap[skill as keyof typeof SkillNameMap] || skill}
                </option>
              ))}
            </select>
          </div>
        </CardDescription>
      </CardHeader>
      <CardContent>
        {loading ? (
          <Spinner />
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
              <Line
                dataKey="classTotalGrade"
                type="monotone"
                stroke="var(--color-classTotalGrade)"
                strokeWidth={2}
                dot={true}
              />
              <Line
                dataKey="personalTotalGrade"
                type="monotone"
                stroke="var(--color-personalTotalGrade)"
                strokeWidth={2}
                dot={true}
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
  const [loading, setLoading] = useState(true)

  if (!liff || !profile) return <div>Profile not found</div>

  return (
    <>
      {loading && <Spinner />}
      <div className={loading ? 'hidden' : ''}>
        <ClassProgressChart onLoadingChange={setLoading} />
      </div>
    </>
  )
}
