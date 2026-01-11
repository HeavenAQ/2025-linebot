import * as admin from 'firebase-admin'

// Initialize Firebase Admin SDK exactly once
if (!admin.apps.length) {
  const raw = process.env.GOOGLE_SERVICE_ACCOUNT_KEY
  if (!raw) {
    throw new Error('GOOGLE_SERVICE_ACCOUNT_KEY is not defined')
  }

  let credentialInput: any = raw
  try {
    // Support JSON string or base64-encoded JSON; otherwise treat as path
    const trimmed = raw.trim()
    if (trimmed.startsWith('{')) {
      credentialInput = JSON.parse(trimmed)
    } else if (/^[A-Za-z0-9+/=]+$/.test(trimmed)) {
      // Might be base64 JSON
      const decoded = Buffer.from(trimmed, 'base64').toString('utf8')
      const maybeJSON = decoded.trim()
      if (maybeJSON.startsWith('{')) {
        credentialInput = JSON.parse(maybeJSON)
      }
    }
  } catch (e) {
    // Fall back to treating raw as a file path
    credentialInput = raw
  }

  admin.initializeApp({
    credential: admin.credential.cert(credentialInput)
  })
}

const db = admin.firestore()
export { db }
