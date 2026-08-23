import { getBackendBaseUrl } from '@/utils/env'

/**
 * The backend identifies callers by their LINE ID token, not by the user ID in
 * the request — anyone can send anyone's ID. LiffProvider registers a source
 * for that token once login completes, and every backend call carries it.
 *
 * It is a source rather than a fixed string because ID tokens expire: reading
 * it per request lets LIFF hand back a refreshed one instead of the page
 * pinning whatever token happened to exist at login.
 */
type IdTokenSource = () => string | null

let idTokenSource: IdTokenSource = () => null

export function setIdTokenSource(source: IdTokenSource | null): void {
  idTokenSource = source ?? (() => null)
}

/** Fetch a backend path with the caller's identity attached. */
export async function authorizedFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers)
  const token = idTokenSource()
  if (token) headers.set('Authorization', `Bearer ${token}`)
  return fetch(`${getBackendBaseUrl()}${path}`, { ...init, headers })
}
