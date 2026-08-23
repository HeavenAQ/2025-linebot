import { getBackendBaseUrl } from '@/utils/env'

export interface WeeklyReflection {
  user_id: string
  week: string
  note: string
  updated_at: string
}

/** Every week a learner has written about, keyed by ISO week label. */
export async function fetchWeeklyReflections(
  userId: string
): Promise<Record<string, WeeklyReflection>> {
  const query = new URLSearchParams({ user_id: userId })
  const response = await fetch(`${getBackendBaseUrl()}/api/db/weekly-reflections?${query}`)
  if (!response.ok) throw new Error(`無法讀取反思紀錄 (${response.status})`)
  const json = await response.json()
  return (json.data ?? {}) as Record<string, WeeklyReflection>
}

export async function saveWeeklyReflection(
  userId: string,
  week: string,
  note: string
): Promise<WeeklyReflection> {
  const response = await fetch(`${getBackendBaseUrl()}/api/db/weekly-reflection`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: userId, week, note })
  })
  if (!response.ok) {
    if (response.status === 413) throw new Error('反思內容太長，請精簡後再儲存。')
    throw new Error(`儲存失敗 (${response.status})`)
  }
  return (await response.json()) as WeeklyReflection
}
