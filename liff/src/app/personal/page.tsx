'use client'

import { Line, LineChart, Bar, BarChart, CartesianGrid, LabelList, XAxis, YAxis } from 'recharts'
import React, { useEffect, useState, Dispatch, SetStateAction } from 'react'
import { useLiff } from '../LiffProvider'
import type { GradingDetail, UserData } from '@/types'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent
} from '@/components/ui/chart'
import { mPlusRounded1c } from '@/components/Fonts/M_PLUS_Rounded_1c'
import Spinner from '@/components/ui/spinner'
import { Skill, SkillNameMap } from '@/lib/types'
import { getBackendBaseUrl } from '@/utils/env'

interface MovementDetailBarChartProps {
  userData: UserData
  selectedDate: string
  setSelectedDate: Dispatch<SetStateAction<string>>
}

const MovementDetailBarChart = ({
  userData,
  selectedDate,
  setSelectedDate
}: MovementDetailBarChartProps) => {
  const [selectedSkill, setSelectedSkill] = useState<Skill>('serve') // Default skill

  // Get the available skills and dates
  const availableSkills = Object.keys(SkillNameMap) as Skill[]
  const availableDates =
    selectedSkill && userData.portfolio[selectedSkill]
      ? Object.keys(userData.portfolio[selectedSkill])
      : []

  // Get grading details for the selected skill and date
  const gradingDetails =
    selectedSkill &&
    selectedDate &&
    userData.portfolio[selectedSkill][selectedDate]?.grading_outcome?.grading_details

  const chartData = (gradingDetails !== "") ?
    gradingDetails?.map((detail: GradingDetail) => ({
      description: detail.description,
      grade: detail.grade.toFixed(2)
    })) : []

  const chartConfig = {
    grade: {
      label: '得分',
      color: 'hsl(var(--chart-1))'
    },
    label: {
      color: 'hsl(var(--background))'
    }
  }

  return (
    <Card
      className={`${mPlusRounded1c.className} mx-auto mt-5 w-10/12 max-w-[800px] animate-fade-down`}
    >
      <CardHeader>
        <CardTitle className="mb-3">動作細節評分</CardTitle>
        <CardDescription className="flex items-center justify-start gap-3">
          {/* Skill Selector */}
          <div>
            <select
              value={selectedSkill}
              onChange={e => {
                setSelectedSkill(e.target.value as Skill)
                setSelectedDate('') // Reset date when skill changes
              }}
              className="rounded border px-2 py-1 text-zinc-700"
            >
              {availableSkills
                .filter(
                  skill =>
                    userData.portfolio[skill] && Object.keys(userData.portfolio[skill]).length > 0
                ) // Only show skills with records
                .map(skill => (
                  <option key={skill} value={skill}>
                    {SkillNameMap[skill as keyof typeof SkillNameMap] || skill}
                  </option>
                ))}{' '}
            </select>
          </div>
          {/* Date Selector */}
          <div>
            <select
              value={selectedDate}
              onChange={e => setSelectedDate(e.target.value)}
              className="rounded border px-2 py-1 text-zinc-700"
            >
              {availableDates.map(date => (
                <option key={date} value={date}>
                  {date}
                </option>
              ))}
            </select>
          </div>
        </CardDescription>
      </CardHeader>
      <CardContent>
        {/* Chart */}
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
            <Bar
              dataKey="grade"
              layout="vertical"
              fill="var(--color-grade)"
              fillOpacity={0.7}
              radius={4}
            >
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
    <Card
      className={`${mPlusRounded1c.className} mx-auto mb-10 mt-5 w-10/12 max-w-[800px] animate-fade-down`}
    >
      <CardHeader>
        <CardTitle className="mb-3">每週進步</CardTitle>
        <CardDescription>
          <div>
            <select
              value={selectedSkill}
              onChange={e => {
                setSelectedSkill(e.target.value as Skill)
              }}
              className="rounded border px-2 py-1 text-zinc-700"
            >
              {availableSkills
                .filter(
                  skill =>
                    userData.portfolio[skill] && Object.keys(userData.portfolio[skill]).length > 0
                ) // Only show skills with records
                .map(skill => (
                  <option key={skill} value={skill}>
                    {SkillNameMap[skill as keyof typeof SkillNameMap] || skill}
                  </option>
                ))}{' '}
            </select>
          </div>
        </CardDescription>
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
              dot={true}
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
  const [selectedDate, setSelectedDate] = useState('') // Default date
  const { liff, profile } = useLiff()

  useEffect(() => {
    if (!liff) return
    if (!profile?.userId) return

    const fetchData = async () => {
      try {
        const base = getBackendBaseUrl()
        const response = await fetch(`${base}/api/db/user?user_id=${profile?.userId}`)
        if (!response.ok) {
          throw new Error(`Failed to fetch user data: ${response.statusText}`)
        }
        const data = await response.json()
        setUserData(data)

        // Automatically set the first available date for the default skill
        const dates = data.portfolio?.serve ? Object.keys(data.portfolio.serve) : []
        setSelectedDate(dates[0] || '') // Set to the first date or an empty string
      } catch (err) {
        if (err instanceof Error) console.log(err.message)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [liff, profile])

  if (loading) {
    return <Spinner />
  }

  if (!userData) {
    return <div className="mt-10 text-center font-semibold">No data available</div>
  }

  return (
    <>
      <MovementDetailBarChart
        userData={userData}
        selectedDate={selectedDate}
        setSelectedDate={setSelectedDate}
      />
      <PersonalProgressChart userData={userData} />
    </>
  )
}
