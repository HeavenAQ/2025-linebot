import * as admin from 'firebase-admin'

// Initialize Firebase Admin SDK only one
if (!admin.apps.length) {
  // Get the service account key from the environment variable. Not found -> throw an error
  const serviceAccountKey = process.env.GOOGLE_SERVICE_ACCOUNT_KEY
  if (serviceAccountKey === undefined) {
    throw new Error('GOOGLE_SERVICE_ACCOUNT_KEY is not defined')
  }

  // init the Firebase Admin SDK
  admin.initializeApp({
    credential: admin.credential.cert(serviceAccountKey)
  })
}

const db = admin.firestore()
export { db }
