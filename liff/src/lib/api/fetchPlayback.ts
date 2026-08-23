import { PlaybackResponseSchema, type PlaybackResponse } from '@/schemas/userData.schema'
import { authorizedFetch } from '@/lib/api/client'
import type { Skill } from '@/lib/types'

export async function fetchPlayback(
  userId: string,
  skill: Skill,
  workDate: string
): Promise<PlaybackResponse> {
  const query = new URLSearchParams({ user_id: userId, skill, work_date: workDate })
  const response = await authorizedFetch(`/api/db/playback?${query.toString()}`)
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as { error?: string } | null
    if (response.status === 409) {
      throw new Error('此筆舊版分析沒有同步比較影片，請重新上傳影片以產生新的比較結果。')
    }
    throw new Error(body?.error || `Unable to load playback (${response.status})`)
  }
  return PlaybackResponseSchema.parse(await response.json())
}
