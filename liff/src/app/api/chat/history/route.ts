import { NextResponse } from 'next/server'
import { getBackendBaseUrl } from '@/utils/env'

export async function GET(req: Request) {
  try {
    const { searchParams } = new URL(req.url)
    const userId = searchParams.get('user_id')
    const skill = searchParams.get('skill') || ''

    if (!userId) {
      return NextResponse.json({ error: 'missing user_id' }, { status: 400 })
    }

    const base = getBackendBaseUrl()
    const qs = new URLSearchParams({ user_id: userId })
    if (skill) qs.set('skill', skill)

    const upstream = await fetch(`${base}/api/chat/history?${qs.toString()}`, {
      method: 'GET',
    })

    if (!upstream.ok) {
      let message = 'failed to fetch chat history'
      try {
        const raw = await upstream.json()
        message = raw?.error || message
      } catch (e) {
        console.log(e)
      }
      return NextResponse.json({ error: message }, { status: upstream.status })
    }

    const json = await upstream.json()
    return NextResponse.json(json, { status: 200 })
  } catch (e) {
    const message = e instanceof Error ? e.message : 'unknown error'
    return NextResponse.json({ error: message }, { status: 500 })
  }
}

