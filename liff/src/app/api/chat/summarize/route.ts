import { NextResponse } from 'next/server'
import { getBackendBaseUrl } from '@/utils/env'

type SummarizeReq = { content?: string; user_id?: string; skill?: string }

export async function POST(req: Request) {
  try {
    const body = (await req.json()) as SummarizeReq
    const content = (body?.content || '').trim()
    const user_id = (body?.user_id || '').trim()
    const skill = (body?.skill || '').trim()
    if (!content || !user_id) {
      return NextResponse.json({ error: 'invalid body' }, { status: 400 })
    }

    const base = getBackendBaseUrl()
    const upstream = await fetch(`${base}/api/chat/summarize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content, user_id, skill }),
    })

    if (!upstream.ok) {
      let message = 'failed to summarize'
      try {
        const raw = await upstream.json()
        message = raw?.error || message
      } catch {}
      return NextResponse.json({ error: message }, { status: upstream.status })
    }

    const json = await upstream.json()
    return NextResponse.json(json, { status: 200 })
  } catch (e) {
    const message = e instanceof Error ? e.message : 'unknown error'
    return NextResponse.json({ error: message }, { status: 500 })
  }
}
