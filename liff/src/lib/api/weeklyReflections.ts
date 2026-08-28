import { authorizedFetch } from '@/lib/api/client'

export interface WeeklyReflection {
  user_id: string
  week: string
  note: string
  /** The learner's own 課前檢視要點; empty on a week written before it existed. */
  preview: string
  updated_at: string
}

/** The two notes a week's record holds. The value is the field the API takes. */
export type WeeklyNoteField = 'note' | 'preview'

/** Names the note in what the student reads, so an error says which one. */
const NOTE_LABEL: Record<WeeklyNoteField, string> = {
  note: '反思',
  preview: '課前檢視要點'
}

/** Every week a learner has written about, keyed by ISO week label. */
export async function fetchWeeklyReflections(
  userId: string
): Promise<Record<string, WeeklyReflection>> {
  const query = new URLSearchParams({ user_id: userId })
  const response = await authorizedFetch(`/api/db/weekly-reflections?${query}`)
  if (!response.ok) throw new Error(`無法讀取每週紀錄 (${response.status})`)
  const json = await response.json()
  return (json.data ?? {}) as Record<string, WeeklyReflection>
}

/**
 * Saves one of the week's notes. Only the named field is sent, and the server
 * writes only what it is sent, so saving a reflection cannot blank a preview
 * written days earlier — or the other way round.
 */
export async function saveWeeklyNote(
  userId: string,
  week: string,
  field: WeeklyNoteField,
  text: string
): Promise<WeeklyReflection> {
  const response = await authorizedFetch(`/api/db/weekly-reflection`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: userId, week, [field]: text })
  })
  if (!response.ok) {
    if (response.status === 413) throw new Error(`${NOTE_LABEL[field]}內容太長，請精簡後再儲存。`)
    throw new Error(`儲存失敗 (${response.status})`)
  }
  return (await response.json()) as WeeklyReflection
}
