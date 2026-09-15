'use client'

import { useCallback, useEffect, useMemo, useRef, useState, type RefObject } from 'react'
import {
  advanceCheckpointLoop,
  checkpointRange,
  type ClipRange,
  type RangeMarker
} from '@/lib/checkpointPlayback'

type Selection = { id: string; label: string; student: ClipRange; expert: ClipRange }

export function useCheckpointLoop(
  student: RefObject<HTMLVideoElement | null>,
  expert: RefObject<HTMLVideoElement | null>,
  playing: RefObject<boolean>,
  view: 'both' | 'student' | 'expert',
  onPlaying: (_value: boolean) => void
) {
  const [selection, setSelection] = useState<Selection | null>(null)
  const selected = useRef<Selection | null>(null)
  const stop = useCallback(() => {
    selected.current = null
    setSelection(null)
  }, [])
  const tracks = useCallback(() => {
    const value = selected.current
    if (!value) return []
    return [
      ...(view !== 'expert' && student.current
        ? [{ video: student.current, range: value.student }]
        : []),
      ...(view !== 'student' && expert.current
        ? [{ video: expert.current, range: value.expert }]
        : [])
    ]
  }, [expert, student, view])
  const start = useCallback(
    (
      id: string,
      label: string,
      sm: RangeMarker,
      em: RangeMarker,
      studentEnd: number,
      expertStart: number,
      expertEnd: number
    ) => {
      const value = {
        id,
        label,
        student: checkpointRange(sm, 0, studentEnd),
        expert: checkpointRange(em, expertStart, expertEnd)
      }
      selected.current = value
      setSelection(value)
      playing.current = true
      onPlaying(true)
      for (const { video, range } of tracks()) {
        video.currentTime = range.start
        void video.play().catch(() => undefined)
      }
    },
    [onPlaying, playing, tracks]
  )
  useEffect(() => {
    if (!selection) return
    const restartLoadedTrack = (event: Event) => {
      const track = tracks().find(item => item.video === event.target)
      if (!track) return
      track.video.currentTime = track.range.start
      if (playing.current) void track.video.play().catch(() => undefined)
    }
    const videos = [student.current, expert.current]
    videos.forEach(video => video?.addEventListener('loadedmetadata', restartLoadedTrack))
    let frame = requestAnimationFrame(function tick() {
      if (playing.current) advanceCheckpointLoop(tracks())
      frame = requestAnimationFrame(tick)
    })
    return () => {
      cancelAnimationFrame(frame)
      videos.forEach(video => video?.removeEventListener('loadedmetadata', restartLoadedTrack))
    }
  }, [expert, playing, selection, student, tracks])
  return useMemo(() => ({ selection, start, stop }), [selection, start, stop])
}
