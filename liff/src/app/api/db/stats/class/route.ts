import { NextResponse } from 'next/server'
import { getBackendBaseUrl } from '@/utils/env'

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url)
  const skill = searchParams.get('skill')
  if (!skill) {
    return NextResponse.json({ error: 'missing skill' }, { status: 400 })
  }
  const base = getBackendBaseUrl()
  const upstream = `${base}/api/db/stats/class?skill=${encodeURIComponent(skill)}`
  try {
    const res = await fetch(upstream, { method: 'GET', cache: 'no-store' })
    const json = await res.json()
    return NextResponse.json(json, { status: res.status })
  } catch (e) {
    console.error('Upstream error:', e)
    return NextResponse.json({ error: 'Upstream error' }, { status: 502 })
  }
}
