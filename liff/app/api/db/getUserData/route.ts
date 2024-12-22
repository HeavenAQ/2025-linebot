import { NextResponse } from 'next/server'
import { db } from '@/lib/firebaseAdmin'

export async function GET(req: Request) {
  // Extract userId from the query string
  const { searchParams } = new URL(req.url)
  const userId = searchParams.get('userId')

  if (!userId) {
    return NextResponse.json({ error: 'Missing userId' }, { status: 400 })
  }

  try {
    // Fetch the user document
    const userDoc = await db.collection('users').doc(userId).get()

    // Handle the case where the document doesn't exist
    if (!userDoc.exists) {
      return NextResponse.json({ error: 'User not found' }, { status: 404 })
    }

    // Cast the Firestore document data to the UserData type
    const userData = userDoc.data()

    // Respond with the UserData
    return NextResponse.json(userData, { status: 200 })
  } catch (error) {
    console.error('Error fetching user data:', error)

    // Respond with an error message
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 })
  }
}
