'use client'
import { Line, LineChart, CartesianGrid, XAxis, YAxis } from 'recharts'
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

interface ClassProgressChartProps {
  userData: UserData
}
// const PersonalProgressChart = ({ userData }: ClassProgressChartProps) => {
//   const chartConfig = {
//     totalGrade: {
//       label: '成績',
//       color: 'hsl(var(--chart-1))'
//     }
//   } satisfies ChartConfig
//   const [selectedSkill, setSelectedSkill] = useState<Skill>('Serve') // Default skill
//   const availableSkills = Object.keys(userData.Portfolio || {})
//
//   return (
//     <Card
//       className={`${mPlusRounded1c.className} mx-auto mb-10 mt-5 w-10/12 max-w-[800px] animate-fade-down`}
//     >
//       <CardHeader>
//         <CardTitle className="mb-3">每週進步</CardTitle>
//         <CardDescription>
//           <div>
//             <select
//               value={selectedSkill}
//               onChange={e => {
//                 setSelectedSkill(e.target.value as Skill)
//               }}
//               className="rounded border px-2 py-1"
//             >
//               {availableSkills
//                 .filter(
//                   skill =>
//                     userData.Portfolio[skill] && Object.keys(userData.Portfolio[skill]).length > 0
//                 ) // Only show skills with records
//                 .map(skill => (
//                   <option key={skill} value={skill}>
//                     {SkillNameMap[skill as keyof typeof SkillNameMap] || skill}
//                   </option>
//                 ))}{' '}
//             </select>
//           </div>
//         </CardDescription>
//       </CardHeader>
//       <CardContent>
//         <ChartContainer config={chartConfig}>
//           <LineChart
//             accessibilityLayer
//             data={Object.keys(userData.Portfolio[selectedSkill])
//               .sort((a, b) => {
//                 const [yearA, monthA, dayA, hourA, minuteA] = a.split('-').map(Number)
//                 const [yearB, monthB, dayB, hourB, minuteB] = b.split('-').map(Number)
//
//                 const dateA = new Date(yearA, monthA - 1, dayA, hourA, minuteA) // Adjust month (0-based index)
//                 const dateB = new Date(yearB, monthB - 1, dayB, hourB, minuteB)
//
//                 return dateA.getTime() - dateB.getTime() // Ascending order
//               })
//               .map(date => ({
//                 date,
//                 totalGrade:
//                   userData.Portfolio[selectedSkill][date].GradingOutcome.TotalGrade.toFixed(2)
//               }))}
//             width={500}
//             height={500}
//           >
//             <CartesianGrid vertical={false} />
//             <XAxis
//               dataKey="date"
//               type="category"
//               tickLine={false}
//               axisLine={false}
//               tickMargin={9}
//               angle={-35}
//               tickFormatter={(value: string) => value.slice(5, 10).replace('-', '/')} // Format dates to month
//               dx={-8}
//               dy={5}
//             />
//             <YAxis
//               tickLine={false}
//               axisLine={false}
//               tickFormatter={value => `${value}`}
//               domain={[0, 100]}
//               width={40}
//               dx={-10}
//             />
//             <ChartTooltip cursor={false} content={<ChartTooltipContent hideLabel />} />
//             <Line
//               dataKey="totalGrade"
//               type="monotone"
//               stroke="var(--color-totalGrade)"
//               strokeWidth={2}
//               dot={true}
//             />
//           </LineChart>
//         </ChartContainer>
//       </CardContent>
//     </Card>
//   )
// }

export default function ClassPage() {
  const [dbData, setDbData] = useState<Record<string, UserData> | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch(`/api/db/listUsersData`)
        if (!response.ok) {
          throw new Error(`Failed to fetch user data: ${response.statusText}`)
        }
        const data = await response.json()
        setDbData(data)
      } catch (err) {
        if (err instanceof Error) console.log(err.message)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [])

  return <></>
}
