import { NextResponse } from 'next/server'
import { getUserData } from '@/lib/services/user.service'
import { ApiError } from '@/schemas/apiResponse.schema'

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url)
  const userId = searchParams.get('user_id')
  if (!userId) {
    return NextResponse.json<ApiError>({ ok: false, error: 'Parameter userId is missing' }, { status: 400 })
  }

  console.log(`Getting data (user ID: ${userId})`)
  const result = await getUserData(userId)
  if (!result.ok) {
    return NextResponse.json<ApiError>({ ok: false, error: result.error.message }, { status: 502 })
  }
  return NextResponse.json(result.data, { status: 200 })
}
