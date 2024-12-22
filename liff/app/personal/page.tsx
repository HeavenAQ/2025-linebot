'use client'

import { Bar, BarChart, CartesianGrid, LabelList, XAxis, YAxis } from 'recharts'
import React, { useEffect, useState } from 'react'
import { useLiff } from '../../app/LiffProvider'
import { UserData } from '../api/db/getUserData/types'

import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle
} from '@/components/ui/card'

import { ChartContainer, ChartTooltip, ChartTooltipContent } from '@/components/ui/chart'
import { mPlusRounded1c } from '@/components/Fonts/M_PLUS_Rounded_1c'

export default function Component() {
  const [userData, setUserData] = useState<UserData | null>(null)
  const [loading, setLoading] = useState(true)
  const { liff, profile } = useLiff()

  useEffect(() => {
    if (liff) {
      if (!liff.isLoggedIn || !profile) {
        liff.login()
        return
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
      } catch (err) {
        if (err instanceof Error) console.log(err.message)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [liff, profile])

  if (loading) {
    return <div className="text-center">Loading...</div>
  }

  if (!userData) {
    return <div className="text-center">No data available</div>
  }

  // Extract grading details
  const gradingDetails =
    userData.Portfolio?.Serve &&
    Object.values(userData.Portfolio.Serve)[0]?.GradingOutcome?.GradingDetails

  const chartData =
    gradingDetails?.map(detail => ({
      description: detail.Description,
      grade: detail.Grade.toFixed(2)
    })) || []

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
        <CardTitle>發球</CardTitle>
        <CardDescription>得分細節</CardDescription>
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
