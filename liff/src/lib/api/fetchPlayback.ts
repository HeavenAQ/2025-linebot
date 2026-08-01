import { PlaybackResponseSchema, type PlaybackResponse } from '@/schemas/userData.schema'
import { getBackendBaseUrl } from '@/utils/env'
import type { Skill } from '@/lib/types'

export async function fetchPlayback(
  userId: string,
  skill: Skill,
  workDate: string
): Promise<PlaybackResponse> {
  const query = new URLSearchParams({ user_id: userId, skill, work_date: workDate })
  const response = await fetch(`${getBackendBaseUrl()}/api/db/playback?${query.toString()}`)
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as { error?: string } | null
    throw new Error(body?.error || `Unable to load playback (${response.status})`)
  }
  return PlaybackResponseSchema.parse(await response.json())
}
