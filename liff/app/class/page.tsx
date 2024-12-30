'use client'
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from 'recharts'
import React, { useEffect, useState } from 'react'
import { UserData } from '../api/db/getUserData/types'

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

interface ClassProgressChartProps {
  classData: Record<string, UserData>
  personalData: UserData
}
const ClassProgressChart = ({ classData, personalData }: ClassProgressChartProps) => {
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
  const [selectedSkill, setSelectedSkill] = useState<Skill>('Serve') // Default skill

  const classAndPersonalAverage = () => {
    const curUserPortfolios = personalData.Portfolio[selectedSkill]
    const curUserPortfolioKeys = Object.keys(curUserPortfolios)

    // Temporary map to store max grade for each date per user
    const personalMax: Record<string, number> = {}
    curUserPortfolioKeys.forEach(key => {
      const date = key.slice(0, 10)
      const totalGrade = curUserPortfolios[key].GradingOutcome.TotalGrade

      // Track the maximum grade for this user on this date
      personalMax[date] = Math.max(personalMax[date] || 0, totalGrade)
    })

    return Object.entries(
      Object.values(classData).reduce(
        (acc, cur) => {
          const curUserPortfolios = cur.Portfolio[selectedSkill]
          const curUserPortfolioKeys = Object.keys(curUserPortfolios)

          // Temporary map to store max grade for each date per user
          const userMaxGrades: Record<string, number> = {}
          curUserPortfolioKeys.forEach(key => {
            const date = key.slice(0, 10)
            const totalGrade = curUserPortfolios[key].GradingOutcome.TotalGrade

            // Track the maximum grade for this user on this date
            userMaxGrades[date] = Math.max(userMaxGrades[date] || 0, totalGrade)
          })

          // Aggregate max grades for this user into the global accumulator
          Object.entries(userMaxGrades).forEach(([date, maxGrade]) => {
            acc[date] ??= { total: 0, count: 0 } // Initialize the record for this date
            acc[date].total += maxGrade // Add the user's max grade for this date
            acc[date].count += 1 // Increment the count of users contributing to this date
          })
          return acc
        },
        {} as Record<string, { total: number; count: number }>
      )
    )
      .map(([date, { total, count }]) => ({
        date,
        classTotalGrade: (total / count).toFixed(2),
        personalTotalGrade: personalMax[date].toFixed(2)
      }))
      .sort((a, b) => a.date.localeCompare(b.date))
  }

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
        <ChartContainer config={chartConfig}>
          <LineChart accessibilityLayer data={classAndPersonalAverage()} width={500} height={500}>
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
      </CardContent>
    </Card>
  )
}

export default function ClassPage() {
  const [dbData, setDbData] = useState<Record<string, UserData> | null>(null)
  const [personalData, setPersonalData] = useState<UserData | null>(null)
  const [loading, setLoading] = useState(true)
  const { liff, profile } = useLiff()

  useEffect(() => {
    if (!profile) return

    const fetchData = async () => {
      try {
        const response = await fetch(`/api/db/listUsersData`)
        const personalData = await fetch(`/api/db/getUserData?userId=${profile?.userId}`)
        if (!response.ok) {
          throw new Error(`Failed to fetch user data: ${response.statusText}`)
        }

        setDbData(await response.json())
        setPersonalData(await personalData.json())
      } catch (err) {
        if (err instanceof Error) console.log(err.message)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [profile])

  if (loading || !dbData || !personalData) return <Spinner />

  if (!liff || !profile) return <div>Profile not found</div>

  return (
    <>
      <ClassProgressChart classData={dbData} personalData={personalData} />
    </>
  )
}
