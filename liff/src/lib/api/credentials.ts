/**
 * Which LINE credential a backend call carries.
 *
 * A LINE ID token lasts about an hour and LIFF cannot refresh one: getIDToken()
 * keeps handing back the token from login after it has expired. The LIFF access
 * token outlives it, and the backend accepts either, so a page left open keeps
 * working by switching to the access token once the ID token is (nearly) spent.
 *
 * The ID token is preferred while it is fresh: it is what the backend has always
 * verified, and it proves identity without a round trip to LINE's API.
 */
export type Credential =
  | { kind: 'id_token'; header: 'Authorization'; value: string }
  | { kind: 'access_token'; header: 'X-Line-Access-Token'; value: string }

/**
 * How long an ID token must still have to be sent. Covers clock skew between
 * this device and the backend, and the time the request spends in flight.
 */
export const ID_TOKEN_EXPIRY_MARGIN_SECONDS = 60

/**
 * The `exp` claim of a JWT, in Unix seconds, or null when it cannot be read.
 *
 * Decoded without verification — this only decides which credential to send;
 * the backend verifies whatever arrives.
 */
export function jwtExpiry(token: string): number | null {
  const parts = token.split('.')
  if (parts.length !== 3 || !parts[1]) return null
  try {
    const base64 = parts[1].replace(/-/g, '+').replace(/_/g, '/')
    const padded = base64 + '='.repeat((4 - (base64.length % 4)) % 4)
    const bytes = Uint8Array.from(atob(padded), char => char.charCodeAt(0))
    // The payload carries the learner's display name, so decode it as UTF-8.
    const payload: unknown = JSON.parse(new TextDecoder().decode(bytes))
    if (!payload || typeof payload !== 'object') return null
    const exp = (payload as { exp?: unknown }).exp
    return typeof exp === 'number' && Number.isFinite(exp) ? exp : null
  } catch {
    return null
  }
}

/**
 * The credential to attach, or null when there is none to send.
 *
 * An ID token whose expiry cannot be read counts as expired: sending it would
 * only earn a 401, while the access token may still work.
 */
export function selectCredential(
  idToken: string | null,
  accessToken: string | null,
  nowMs: number
): Credential | null {
  if (idToken) {
    const exp = jwtExpiry(idToken)
    if (exp !== null && exp - nowMs / 1000 > ID_TOKEN_EXPIRY_MARGIN_SECONDS) {
      return { kind: 'id_token', header: 'Authorization', value: `Bearer ${idToken}` }
    }
  }
  if (accessToken) {
    return { kind: 'access_token', header: 'X-Line-Access-Token', value: accessToken }
  }
  return null
}

/** What the learner is told when the backend rate-limits them without saying why. */
export const RATE_LIMITED_FALLBACK = '操作太頻繁，請稍後再試。'

/**
 * The message for a 429, preferring the backend's own `{ "error": "..." }`.
 *
 * A Retry-After in seconds is added to the fallback only; the backend's message
 * is shown as written, since it may already say how long to wait.
 */
export function rateLimitMessage(body: unknown, retryAfter: string | null): string {
  const error = (body as { error?: unknown } | null)?.error
  if (typeof error === 'string' && error.trim()) return error.trim()
  const seconds = retryAfter !== null && /^\d+$/.test(retryAfter.trim()) ? Number(retryAfter) : NaN
  return Number.isFinite(seconds) && seconds > 0
    ? `操作太頻繁，請於 ${seconds} 秒後再試。`
    : RATE_LIMITED_FALLBACK
}
