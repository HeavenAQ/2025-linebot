import { NextResponse } from 'next/server'
import { db } from '@/lib/firebaseAdmin'

export async function GET(req: Request) {
  // Extract userId from the query string

  try {
    // Fetch the user document
    const userDoc = await db.collection('users').get()

    // Handle the case where the document doesn't exist
    if (userDoc.empty) {
      return NextResponse.json({ error: 'User not found' }, { status: 404 })
    }

    // Cast the Firestore document data to the UserData type
    const userData = userDoc.docs.map(doc => doc.data())

    // Respond with the UserData
    return NextResponse.json(userData, { status: 200 })
  } catch (error) {
    console.error('Error fetching user data:', error)

    // Respond with an error message
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 })
  }
}
