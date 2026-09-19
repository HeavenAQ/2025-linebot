import { authorizedFetch, RateLimitedError, rateLimitedMessage } from '@/lib/api/client'
import { UserDataSchema } from '@/schemas/userData.schema'
import type { UserData } from '@/types'

export interface ProfilePayload {
  real_name: string
  experiment_number: string
  handedness: 'left' | 'right'
}

/**
 * Saves the learner's own details. The same call registers someone using the
 * app for the first time and edits an existing profile, so it is the one
 * learner endpoint that works before registration.
 */
export async function saveProfile(payload: ProfilePayload): Promise<UserData> {
  const response = await authorizedFetch('/api/db/profile', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
  if (!response.ok) {
    const limited = await rateLimitedMessage(response)
    if (limited) throw new RateLimitedError(limited)
    const body = await response.json().catch(() => null)
    throw new Error(body?.error || `無法儲存資料 (${response.status})`)
  }
  return UserDataSchema.parse(await response.json())
}
