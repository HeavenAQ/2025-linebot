import { z } from 'zod'

// Summary shape returned by backend for both user and class
const StatsSummarySchema = z.object({
  avg: z.number(),
  max: z.number(),
  min: z.number(),
  std: z.number()
})

// Map of YYYY-MM-DD -> StatsSummary
export const StatsByDateSchema = z.record(z.string(), StatsSummarySchema)
export type StatsByDate = z.infer<typeof StatsByDateSchema>
