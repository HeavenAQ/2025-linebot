import { db } from '@/lib/firebaseAdmin'

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url)
  const profileId = searchParams.get('profileId')
  const date = searchParams.get('date')

  if (!profileId || !date) {
    return new Response(JSON.stringify({ error: 'Missing profileId or date' }), { status: 400 })
  }

  try {
    const summariesRef = db.collection('gpt_summaries')
    const docSnapshot = await summariesRef.doc(profileId).get()

    if (docSnapshot.exists) {
      const userSummaries = docSnapshot.data()
      if (userSummaries && userSummaries[date]) {
        return new Response(JSON.stringify({ summary: userSummaries[date] }), { status: 200 })
      }
    }

    return new Response(JSON.stringify({ summary: null }), { status: 200 })
  } catch (error) {
    console.error('Error fetching summary:', error)
    return new Response(JSON.stringify({ error: 'Internal server error' }), { status: 500 })
  }
}

export async function POST(req: Request) {
  const { userProfileID, date, summary } = await req.json()

  if (!userProfileID || !date || !summary) {
    return new Response(JSON.stringify({ error: 'Missing required fields' }), { status: 400 })
  }

  try {
    const summariesRef = db.collection('gpt_summaries')
    await summariesRef.doc(userProfileID).set({ [date]: summary }, { merge: true })

    return new Response(JSON.stringify({ message: 'Summary saved successfully' }), { status: 200 })
  } catch (error) {
    console.error('Error saving summary:', error)
    return new Response(JSON.stringify({ error: 'Internal server error' }), { status: 500 })
  }
}
