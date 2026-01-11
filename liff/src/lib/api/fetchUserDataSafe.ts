import { UserDataSchema, type UserData } from '@/schemas/userData.schema'
import { getBackendBaseUrl } from '@/utils/env'
import type { Result } from './result'
import { err, ok } from './result'
import { ErrorResponseSchema } from '@/schemas/error.schema'

export async function fetchUserDataSafe(userId: string): Promise<Result<UserData, Error>> {
  try {
    const base = getBackendBaseUrl()
    const qs = new URLSearchParams({ user_id: userId })
    const res = await fetch(
      `${base}/api/db/user?${qs.toString()}`,
      { method: 'GET' },
    )

    // Handle none 2XX errors
    if (!res.ok) {
      let message = "Unknown error occured"
      const raw = await res.json()
      const parsed = ErrorResponseSchema.safeParse(raw)

      if (parsed.success) {
        message = parsed.data.error
      }
      return err(new Error(`Upstream failure: ${message}`))
    }

    const json = await res.json()
    const parsed = UserDataSchema.parse(json)
    return ok(parsed)
  } catch (e) {

    console.log(e)
    const error = e instanceof Error ? e : new Error('Unknown error')
    return err(error)
  }
}

