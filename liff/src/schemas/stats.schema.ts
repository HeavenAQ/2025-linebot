import { z } from 'zod'

// Accept several common variants and normalize to { date, value }
const BasePointVariants = z.union([
  // canonical
  z.object({ date: z.string(), value: z.number() }).transform(v => ({ date: v.date, value: v.value })),
  // alternate value keys
  z.object({ date: z.string(), avg: z.number() }).transform(v => ({ date: v.date, value: v.avg })),
  z.object({ date: z.string(), score: z.number() }).transform(v => ({ date: v.date, value: v.score })),
  // alternate date keys
  z.object({ day: z.string(), value: z.number() }).transform(v => ({ date: v.day, value: v.value })),
  z.object({ d: z.string(), value: z.number() }).transform(v => ({ date: v.d, value: v.value })),
  // alternate combos
  z.object({ day: z.string(), avg: z.number() }).transform(v => ({ date: v.day, value: v.avg })),
  z.object({ d: z.string(), avg: z.number() }).transform(v => ({ date: v.d, value: v.avg })),
  z.object({ day: z.string(), score: z.number() }).transform(v => ({ date: v.day, value: v.score })),
  z.object({ d: z.string(), score: z.number() }).transform(v => ({ date: v.d, value: v.score })),
])

export const StatsPointSchema = BasePointVariants.transform(v => ({
  date: String(v.date).slice(0, 10),
  value: v.value,
}))

export const StatsSeriesSchema = z.array(StatsPointSchema)

export type StatsPoint = z.infer<typeof StatsPointSchema>
export type StatsSeries = z.infer<typeof StatsSeriesSchema>

// Summary shape returned by backend for both user and class
export const StatsSummarySchema = z.object({
  avg: z.number(),
  max: z.number(),
  min: z.number(),
  std: z.number(),
})
export type StatsSummary = z.infer<typeof StatsSummarySchema>

// Map of YYYY-MM-DD -> StatsSummary
export const StatsByDateSchema = z.record(z.string(), StatsSummarySchema)
export type StatsByDate = z.infer<typeof StatsByDateSchema>
