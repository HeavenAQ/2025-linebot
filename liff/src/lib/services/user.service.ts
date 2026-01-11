import { fetchUserDataSafe } from '@/lib/api/fetchUserDataSafe'
import type { Result } from '@/lib/api/result'
import type { UserData } from '@/schemas/userData.schema'

export async function getUserData(userId: string): Promise<Result<UserData, Error>> {
  return fetchUserDataSafe(userId)
}

