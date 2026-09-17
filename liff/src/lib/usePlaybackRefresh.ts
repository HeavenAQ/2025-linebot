'use client'

import { useCallback, useEffect, useRef, type Dispatch, type SetStateAction } from 'react'

import type { PlaybackResponse } from '@/schemas/userData.schema'
import { earliestSignedUrlExpiry, isSameAnalysis, refreshDelayMs } from '@/lib/playbackExpiry'

/**
 * Keeps the signed video URLs of the playback on screen from expiring.
 *
 * Re-fetches a few minutes before the earliest URL expires, again as soon as
 * the page comes back into view if that moment passed while it was hidden
 * (timers do not run reliably in a backgrounded LINE webview), and once when a
 * video fails to load. The refresh is silent: the current playback stays on
 * screen, and is replaced only if the answer is still the same analysis — the
 * learner may have moved to another attempt while it was in flight.
 *
 * Returns the handler for a <video> `error` event.
 */
export function usePlaybackRefresh(
  playback: PlaybackResponse | null,
  setPlayback: Dispatch<SetStateAction<PlaybackResponse | null>>,
  load: () => Promise<PlaybackResponse>
): () => void {
  // The latest loader, so a re-render does not reschedule the timer.
  const loadRef = useRef(load)
  useEffect(() => {
    loadRef.current = load
  })
  const inFlightRef = useRef(false)
  const lastRefreshAtRef = useRef<number | null>(null)
  // The analysis a video error has already refreshed. One retry per analysis:
  // a video that fails with fresh URLs too is not failing for its signature.
  const errorRetriedForRef = useRef<string | null>(null)

  const refresh = useCallback(() => {
    if (inFlightRef.current) return
    inFlightRef.current = true
    lastRefreshAtRef.current = Date.now()
    loadRef
      .current()
      .then(next => {
        setPlayback(current => (isSameAnalysis(current, next) ? next : current))
      })
      .catch(error => {
        // The URLs on screen may still have minutes left; keep them.
        console.warn('Failed to refresh playback URLs:', error)
      })
      .finally(() => {
        inFlightRef.current = false
      })
  }, [setPlayback])

  const expiry = playback ? earliestSignedUrlExpiry(playback) : null

  useEffect(() => {
    if (expiry === null) return
    const delay = refreshDelayMs(expiry, Date.now(), lastRefreshAtRef.current)
    if (delay === null) return
    const scheduled = () => {
      errorRetriedForRef.current = null
      refresh()
    }
    const timer = window.setTimeout(scheduled, delay)
    const onVisible = () => {
      if (document.hidden) return
      if (refreshDelayMs(expiry, Date.now(), lastRefreshAtRef.current) === 0) scheduled()
    }
    document.addEventListener('visibilitychange', onVisible)
    return () => {
      clearTimeout(timer)
      document.removeEventListener('visibilitychange', onVisible)
    }
  }, [expiry, refresh])

  const analysisId = playback?.analysis_id ?? null
  return useCallback(() => {
    if (analysisId === null || errorRetriedForRef.current === analysisId) return
    errorRetriedForRef.current = analysisId
    refresh()
  }, [analysisId, refresh])
}
