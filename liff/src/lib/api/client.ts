import { getBackendBaseUrl } from '@/utils/env'

/**
 * The backend identifies callers by their LINE ID token, not by the user ID in
 * the request — anyone can send anyone's ID. LiffProvider registers a source
 * for that token once login completes, and every backend call carries it.
 *
 * It is a source rather than a fixed string so that a token replaced during the
 * session is picked up, instead of the page pinning whatever existed at login.
 */
type IdTokenSource = () => string | null

let idTokenSource: IdTokenSource = () => null
let onExpired: (() => void) | null = null

export function setIdTokenSource(source: IdTokenSource | null): void {
  idTokenSource = source ?? (() => null)
}

/**
 * Registers what to do when the backend refuses a token we actually held.
 *
 * LINE ID tokens last an hour and LIFF has no way to refresh one: getIDToken()
 * keeps returning the token from login long after it has expired, so a page
 * left open starts failing every call and reloading will not fix it.
 */
export function setExpiredTokenHandler(handler: (() => void) | null): void {
  onExpired = handler
}

/** Fetch a backend path with the caller's identity attached. */
export async function authorizedFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers)
  const token = idTokenSource()
  if (token) headers.set('Authorization', `Bearer ${token}`)
  const response = await fetch(`${getBackendBaseUrl()}${path}`, { ...init, headers })
  // Only a token we sent and the backend refused means an expired session. A
  // 401 with no token is the ordinary state of a page whose LIFF login has not
  // finished yet — calling it an expired session there strands the learner on
  // "reopen from LINE" instead of letting the login they are mid-way through
  // complete. 403 is left alone either way: the caller is known and refused.
  if (response.status === 401 && token) onExpired?.()
  return response
}
