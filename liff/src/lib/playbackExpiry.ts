import type { PlaybackResponse } from '@/schemas/userData.schema'

/**
 * When to ask the backend for fresh signed playback URLs.
 *
 * The videos are V4-signed Cloud Storage URLs that stop working after an hour,
 * so a comparison left open longer starts failing with 403s on the next seek or
 * buffer. The page re-fetches playback a little before the first of them
 * expires, which hands back the same analysis with newly signed URLs.
 */

/** How long before expiry to refresh: a margin for a slow request and clock skew. */
export const REFRESH_MARGIN_SECONDS = 5 * 60

/**
 * The shortest gap between two refreshes. If the backend ever hands back URLs
 * already inside the margin, this keeps the page from re-fetching in a loop.
 */
export const MIN_REFRESH_INTERVAL_MS = 30 * 1000

/** The largest delay setTimeout honours; anything longer fires immediately. */
const MAX_TIMEOUT_MS = 2 ** 31 - 1

type Media = Pick<PlaybackResponse['student_video'], 'signed_url' | 'signed_url_expires_at_unix'>

/**
 * The earliest expiry, in Unix seconds, among the media this playback can show,
 * or null when none of them says when it expires.
 *
 * Media without a URL are skipped (a missing overlay has nothing to expire), as
 * is an expiry of 0, which is what an older backend's omitted field parses to.
 */
export function earliestSignedUrlExpiry(
  playback: Pick<
    PlaybackResponse,
    'student_video' | 'feedback_video' | 'skeleton_overlay_video' | 'expert'
  >
): number | null {
  const media: (Media | undefined)[] = [
    playback.student_video,
    playback.feedback_video,
    playback.skeleton_overlay_video,
    playback.expert?.video
  ]
  let earliest: number | null = null
  for (const item of media) {
    if (!item?.signed_url) continue
    const expiry = item.signed_url_expires_at_unix
    if (!Number.isFinite(expiry) || expiry <= 0) continue
    if (earliest === null || expiry < earliest) earliest = expiry
  }
  return earliest
}

/**
 * Milliseconds until playback should be refreshed, or null if it never needs to.
 *
 * Zero means now: the URLs are already inside the margin or expired. A refresh
 * that happened recently pushes the answer out to the minimum interval.
 */
export function refreshDelayMs(
  expiresAtUnix: number | null,
  nowMs: number,
  lastRefreshAtMs: number | null = null,
  marginSeconds: number = REFRESH_MARGIN_SECONDS
): number | null {
  if (expiresAtUnix === null) return null
  let delay = Math.max(0, (expiresAtUnix - marginSeconds) * 1000 - nowMs)
  if (lastRefreshAtMs !== null) {
    delay = Math.max(delay, lastRefreshAtMs + MIN_REFRESH_INTERVAL_MS - nowMs)
  }
  return Math.min(MAX_TIMEOUT_MS, Math.max(0, delay))
}

/**
 * Whether the same analysis came back with only its URLs re-signed, as opposed
 * to a different attempt being shown. Only then may players keep their place.
 */
export function isSameAnalysis(
  previous: Pick<PlaybackResponse, 'analysis_id' | 'student_video'> | null,
  next: Pick<PlaybackResponse, 'analysis_id' | 'student_video'> | null
): boolean {
  if (!previous || !next) return false
  return (
    previous.analysis_id === next.analysis_id &&
    previous.student_video.object_path === next.student_video.object_path
  )
}
