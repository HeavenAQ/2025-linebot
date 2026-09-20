import { getBackendBaseUrl } from '@/utils/env'
import { rateLimitMessage, selectCredential } from '@/lib/api/credentials'

/**
 * The backend identifies callers by their LINE credentials, not by the user ID
 * in the request — anyone can send anyone's ID. LiffProvider registers sources
 * for the ID token and the access token once login completes, and every
 * backend call carries one of them (see selectCredential for which).
 *
 * They are sources rather than fixed strings so that a token replaced during
 * the session is picked up, instead of the page pinning whatever existed at
 * login.
 */
type TokenSource = () => string | null

interface TokenSources {
  idToken: TokenSource
  accessToken: TokenSource
}

const noToken: TokenSource = () => null

let tokenSources: TokenSources = { idToken: noToken, accessToken: noToken }
let onExpired: (() => void) | null = null
let onRegistrationRequired: (() => void) | null = null

export function setRegistrationRequiredHandler(handler: (() => void) | null): void {
  onRegistrationRequired = handler
}

export function setTokenSources(sources: TokenSources | null): void {
  tokenSources = sources ?? { idToken: noToken, accessToken: noToken }
}

/**
 * Registers what to do when the backend refuses a credential we actually held.
 *
 * An expired ID token alone no longer ends the session: once it is within a
 * minute of expiry, calls switch to the longer-lived LIFF access token. This
 * fires when that is refused too (or there is none to switch to), which
 * reloading will not fix — LIFF keeps handing back the same spent tokens.
 */
export function setExpiredTokenHandler(handler: (() => void) | null): void {
  onExpired = handler
}

/** Thrown for a 429, carrying the backend's own explanation as the message. */
export class RateLimitedError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'RateLimitedError'
  }
}

/**
 * The message to show for a rate-limited response, or null for any other.
 * Reads a clone, so the caller can still consume the body.
 */
export async function rateLimitedMessage(response: Response): Promise<string | null> {
  if (response.status !== 429) return null
  const body = await response
    .clone()
    .json()
    .catch(() => null)
  return rateLimitMessage(body, response.headers.get('Retry-After'))
}

/** Fetch a backend path with the caller's identity attached. */
export async function authorizedFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers)
  const credential = selectCredential(
    tokenSources.idToken(),
    tokenSources.accessToken(),
    Date.now()
  )
  // Exactly one credential goes out, as the backend expects: a stale ID token
  // must not ride along with the access token that replaced it.
  headers.delete('Authorization')
  headers.delete('X-Line-Access-Token')
  if (credential) headers.set(credential.header, credential.value)
  const response = await fetch(`${getBackendBaseUrl()}${path}`, { ...init, headers })
  // Only a credential we sent and the backend refused means an expired session.
  // A 401 with no token at all is the ordinary state of a page whose LIFF login
  // has not finished yet — calling it an expired session there strands the
  // learner on "reopen from LINE" instead of letting the login they are mid-way
  // through complete. 403 is left alone either way: the caller is known and
  // refused.
  if (response.status === 401 && credential) onExpired?.()
  if (response.status === 403) {
    const body = await response.clone().json().catch(() => null)
    if (body?.code === 'registration_required') onRegistrationRequired?.()
  }
  return response
}
