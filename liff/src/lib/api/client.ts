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
 * Registers what to do when the backend refuses our token.
 *
 * LINE ID tokens last an hour and LIFF has no way to refresh one: getIDToken()
 * keeps returning the token from login long after it has expired. A page left
 * open therefore starts failing every call, and the only way back is to log in
 * again — which the provider does, rather than leaving the learner looking at
 * an error that reloading will not fix.
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
  // 401 is the backend saying it could not verify the token, which for a page
  // that had been working means it has expired. 403 is a different thing — the
  // caller is known and refused — so it is left alone.
  if (response.status === 401) onExpired?.()
  return response
}
