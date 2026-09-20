export type RangeMarker = {
  id: string
  timestamp_seconds: number
  start_seconds?: number | null
  end_seconds?: number | null
}

export type ClipRange = { start: number; end: number; inferred: boolean }

// Older follow-through markers carry the initial-to-final scoring evidence
// interval. Replay the action after contact, not its initial-pose reference.
// Do not mutate the stored evidence or change other checkpoints.
export function checkpointReplayMarker(marker: RangeMarker, timeline: RangeMarker[]): RangeMarker {
  if (marker.id !== 'follow_through') return marker
  const contact = timeline.find(item => item.id === 'wrist_flick')?.timestamp_seconds
  if (
    contact === undefined ||
    !Number.isFinite(contact) ||
    !Number.isFinite(marker.start_seconds) ||
    !Number.isFinite(marker.end_seconds) ||
    contact >= marker.end_seconds! ||
    contact <= marker.start_seconds!
  ) {
    return marker
  }
  return { ...marker, start_seconds: contact }
}

// New analyses carry scored intervals. Legacy records only have a point:
// expose a short preview around it, explicitly marked as inferred in the UI.
export function checkpointRange(marker: RangeMarker, start: number, end: number): ClipRange {
  const measured = Number.isFinite(marker.start_seconds) && Number.isFinite(marker.end_seconds)
  const left = measured ? marker.start_seconds! : marker.timestamp_seconds - 0.35
  const right = measured ? marker.end_seconds! + 1 / 30 : marker.timestamp_seconds + 0.35
  const lo = Math.min(Math.max(start, left), Math.max(start, end - 1 / 30))
  return { start: lo, end: Math.min(end, Math.max(lo + 1 / 30, right)), inferred: !measured }
}

type LoopVideo = {
  currentTime: number
  ended: boolean
  paused: boolean
  seeking: boolean
  readyState: number
  pause(): void
  play(): Promise<void>
}

// The faster video waits at its own interval end. Rewind both only after both
// have reached their endpoints; seeking must settle before the next iteration.
export function advanceCheckpointLoop(tracks: { video: LoopVideo; range: ClipRange }[]) {
  if (!tracks.length || tracks.some(({ video }) => video.seeking || video.readyState < 2)) return
  const reached = tracks.map(
    ({ video, range }) => video.ended || video.currentTime >= range.end - 1 / 120
  )
  if (reached.every(Boolean)) {
    for (const { video, range } of tracks) {
      video.currentTime = range.start
      void video.play().catch(() => undefined)
    }
  } else {
    tracks.forEach(({ video }, i) => {
      if (reached[i]) video.pause()
      else if (video.paused) void video.play().catch(() => undefined)
    })
  }
}
