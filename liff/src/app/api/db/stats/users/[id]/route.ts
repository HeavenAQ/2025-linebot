import { NextResponse } from 'next/server'
import { getBackendBaseUrl } from '@/utils/env'

export async function GET(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { searchParams } = new URL(req.url)
  const skill = searchParams.get('skill')
  const { id } = await params
  if (!id || !skill) {
    return NextResponse.json({ error: 'missing id or skill' }, { status: 400 })
  }
  const base = getBackendBaseUrl()
  const upstream = `${base}/api/db/stats/users/${encodeURIComponent(id)}?skill=${encodeURIComponent(skill)}`
  try {
    const res = await fetch(upstream, { method: 'GET', cache: 'no-store' })
    const json = await res.json()
    return NextResponse.json(json, { status: res.status })
  } catch (e) {
    console.error('Upstream error:', e)
    return NextResponse.json({ error: 'Upstream error' }, { status: 502 })
  }
}
