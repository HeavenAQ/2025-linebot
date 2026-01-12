import { ErrorResponseSchema } from '@/schemas/error.schema'
import { StatsByDateSchema, type StatsByDate } from '@/schemas/stats.schema'
import { getBackendBaseUrl } from '@/utils/env'

export async function fetchClassStats(skill: string): Promise<StatsByDate> {
  const base = getBackendBaseUrl()
  const url = `${base}/api/db/stats/class?skill=${encodeURIComponent(skill)}`
  const res = await fetch(url, { method: 'GET', cache: 'no-store' })
  const json = await res.json()
  if (!res.ok) {
    const parsed = ErrorResponseSchema.safeParse(json)
    const message = parsed.success ? parsed.data.error : 'Unknown error'
    throw new Error(`Class stats upstream failure: ${message}`)
  }
  return StatsByDateSchema.parse(json)
}

export async function fetchUserStats(userId: string, skill: string): Promise<StatsByDate> {
  const base = getBackendBaseUrl()
  const url = `${base}/api/db/stats/users/${encodeURIComponent(userId)}?skill=${encodeURIComponent(skill)}`
  const res = await fetch(url, { method: 'GET', cache: 'no-store' })
  const json = await res.json()
  if (!res.ok) {
    const parsed = ErrorResponseSchema.safeParse(json)
    const message = parsed.success ? parsed.data.error : 'Unknown error'
    throw new Error(`User stats upstream failure: ${message}`)
  }
  return StatsByDateSchema.parse(json)
}
