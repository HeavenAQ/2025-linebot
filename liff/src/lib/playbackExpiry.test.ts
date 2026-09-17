import assert from 'node:assert/strict'
import test from 'node:test'

import {
  earliestSignedUrlExpiry,
  isSameAnalysis,
  MIN_REFRESH_INTERVAL_MS,
  refreshDelayMs,
  REFRESH_MARGIN_SECONDS
} from './playbackExpiry.ts'
import type { PlaybackResponse } from '@/schemas/userData.schema.ts'

const media = (signed_url: string, signed_url_expires_at_unix: number, object_path = '') =>
  ({ signed_url, signed_url_expires_at_unix, object_path }) as PlaybackResponse['student_video']

const playback = (
  overrides: Partial<
    Record<'student' | 'feedback' | 'overlay' | 'expert', ReturnType<typeof media>>
  >
) =>
  ({
    analysis_id: 'a1',
    student_video: overrides.student ?? media('https://storage.googleapis.com/s', 2000),
    feedback_video: overrides.feedback ?? media('', 0),
    skeleton_overlay_video: overrides.overlay ?? media('', 0),
    expert: { video: overrides.expert ?? media('https://storage.googleapis.com/e', 3000) }
  }) as PlaybackResponse

test('finds the first signed URL to expire across all media', () => {
  assert.equal(earliestSignedUrlExpiry(playback({})), 2000)
  assert.equal(
    earliestSignedUrlExpiry(playback({ overlay: media('https://storage.googleapis.com/o', 1500) })),
    1500
  )
  assert.equal(
    earliestSignedUrlExpiry(playback({ expert: media('https://storage.googleapis.com/e', 1000) })),
    1000
  )
})

test('ignores media with no URL or no stated expiry', () => {
  // An empty overlay still parses with an expiry field; it has nothing to expire.
  assert.equal(earliestSignedUrlExpiry(playback({ feedback: media('', 10) })), 2000)
  assert.equal(
    earliestSignedUrlExpiry(
      playback({ student: media('https://x/s', 0), expert: media('https://x/e', 0) })
    ),
    null
  )
})

test('schedules the refresh a margin before expiry', () => {
  const now = 1_000_000
  const expiry = now / 1000 + 3600
  assert.equal(refreshDelayMs(expiry, now), (3600 - REFRESH_MARGIN_SECONDS) * 1000)
  assert.equal(refreshDelayMs(expiry, now, null, 60), (3600 - 60) * 1000)
  assert.equal(refreshDelayMs(null, now), null)
})

test('refreshes immediately inside the margin or after expiry', () => {
  const now = 1_000_000
  assert.equal(refreshDelayMs(now / 1000 + REFRESH_MARGIN_SECONDS, now), 0)
  assert.equal(refreshDelayMs(now / 1000 + 10, now), 0)
  assert.equal(refreshDelayMs(now / 1000 - 600, now), 0)
})

test('never refreshes again sooner than the minimum interval', () => {
  const now = 1_000_000
  const expired = now / 1000 - 1
  assert.equal(refreshDelayMs(expired, now, now - 1000), MIN_REFRESH_INTERVAL_MS - 1000)
  assert.equal(refreshDelayMs(expired, now, now - MIN_REFRESH_INTERVAL_MS - 1), 0)
  // A refresh long ago does not delay a far-off expiry.
  assert.equal(
    refreshDelayMs(now / 1000 + 3600, now, now - 1000),
    (3600 - REFRESH_MARGIN_SECONDS) * 1000
  )
})

test('caps the delay to what setTimeout can wait', () => {
  assert.equal(refreshDelayMs(1e12, 0), 2 ** 31 - 1)
})

test('recognizes a re-signed response for the same analysis', () => {
  const before = playback({ student: media('https://x/s?sig=1', 2000, 'users/u/a1.mp4') })
  const after = {
    ...before,
    student_video: media('https://x/s?sig=2', 5600, 'users/u/a1.mp4')
  } as PlaybackResponse
  assert.equal(isSameAnalysis(before, after), true)
  assert.equal(isSameAnalysis(before, { ...after, analysis_id: 'a2' }), false)
  assert.equal(
    isSameAnalysis(before, { ...after, student_video: media('https://x/s', 1, 'other.mp4') }),
    false
  )
  assert.equal(isSameAnalysis(null, after), false)
  assert.equal(isSameAnalysis(before, null), false)
})
