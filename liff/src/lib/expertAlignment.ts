import type { PhaseMarker } from '@/types'

/**
 * Maps the student's motion progress onto the expert video's clock.
 *
 * The two performances do not share a tempo: a learner may spend 40% of the
 * stroke winding up where the expert spends 25%, so stretching the expert's
 * motion window evenly across the student's progress drifts out of sync
 * between checkpoints. Anchoring the map on matching checkpoints and
 * interpolating within each segment keeps every phase aligned — the playback
 * counterpart of the segmental warping used when scoring.
 */

export interface AlignmentAnchor {
  /** Student progress through the motion, 0–1. */
  position: number
  /** The matching moment in the expert video, in seconds. */
  seconds: number
}

const clamp01 = (value: number) => Math.min(1, Math.max(0, value))

const isFiniteNumber = (value: number) => typeof value === 'number' && Number.isFinite(value)

/**
 * Builds the anchors for one analysis: the motion window's ends, plus every
 * checkpoint pair that survives as a strictly increasing correspondence.
 *
 * Pairs are dropped rather than reordered when they disagree with what came
 * before, because a non-monotonic anchor would rewind the expert mid-play.
 * With no usable checkpoints this degrades to the two window ends, which is
 * exactly the old whole-window stretch.
 */
export function buildAlignmentAnchors(
  studentTimeline: readonly PhaseMarker[],
  expertTimeline: readonly PhaseMarker[],
  motionStart: number,
  motionEnd: number
): AlignmentAnchor[] {
  const anchors: AlignmentAnchor[] = [{ position: 0, seconds: motionStart }]
  if (studentTimeline.length === expertTimeline.length) {
    const pairs = studentTimeline
      .map((student, index) => ({
        position: clamp01(student.normalized_position),
        seconds: expertTimeline[index].timestamp_seconds
      }))
      .filter(pair => isFiniteNumber(pair.position) && isFiniteNumber(pair.seconds))
      .sort((a, b) => a.position - b.position)

    for (const pair of pairs) {
      const previous = anchors[anchors.length - 1]
      const seconds = Math.min(motionEnd, Math.max(motionStart, pair.seconds))
      if (pair.position > previous.position && seconds > previous.seconds) {
        anchors.push({ position: pair.position, seconds })
      }
    }
  }

  const last = anchors[anchors.length - 1]
  if (last.position < 1 && motionEnd > last.seconds) {
    anchors.push({ position: 1, seconds: motionEnd })
  } else if (last.position >= 1) {
    // A checkpoint already landed on the end of the motion; keep it as the tail.
    last.seconds = Math.max(last.seconds, motionEnd)
  }
  return anchors
}

const segmentAt = (anchors: readonly AlignmentAnchor[], position: number) => {
  for (let index = 1; index < anchors.length; index += 1) {
    if (position <= anchors[index].position) {
      return [anchors[index - 1], anchors[index]] as const
    }
  }
  return [anchors[anchors.length - 2], anchors[anchors.length - 1]] as const
}

/** Interpolates the expert timestamp for a student progress value. */
export function expertTimeAt(anchors: readonly AlignmentAnchor[], position: number): number {
  const target = clamp01(position)
  if (anchors.length === 0) return 0
  if (anchors.length === 1) return anchors[0].seconds
  const [previous, next] = segmentAt(anchors, target)
  const span = next.position - previous.position
  if (span <= 0) return next.seconds
  return (
    previous.seconds + ((target - previous.position) / span) * (next.seconds - previous.seconds)
  )
}

/** Slowest and fastest expert playback we will ask a browser for. */
const MIN_RATE = 0.25
const MAX_RATE = 4

/**
 * How fast the expert video must run to hold the current segment in step.
 *
 * Without this the expert plays at 1x and is yanked back into place whenever
 * drift is noticed, which reads as stutter; matching the segment's slope keeps
 * the correction continuous.
 */
export function expertRateAt(
  anchors: readonly AlignmentAnchor[],
  position: number,
  studentMotionDuration: number
): number {
  if (anchors.length < 2 || studentMotionDuration <= 0) return 1
  const [previous, next] = segmentAt(anchors, clamp01(position))
  const span = next.position - previous.position
  if (span <= 0) return 1
  const rate = (next.seconds - previous.seconds) / span / studentMotionDuration
  if (!Number.isFinite(rate) || rate <= 0) return 1
  return Math.min(MAX_RATE, Math.max(MIN_RATE, rate))
}

/**
 * The student progress that puts the expert at a given moment of its video.
 *
 * The inverse of `expertTimeAt`. The expert-only view scrubs on the expert's
 * own clock, but the student video is still what drives playback, so a point
 * picked on the expert's timeline has to come back as student progress.
 */
export function progressAtExpertTime(anchors: readonly AlignmentAnchor[], seconds: number): number {
  if (anchors.length === 0) return 0
  if (anchors.length === 1) return anchors[0].position
  const first = anchors[0]
  const last = anchors[anchors.length - 1]
  if (seconds <= first.seconds) return first.position
  if (seconds >= last.seconds) return last.position
  for (let index = 1; index < anchors.length; index += 1) {
    const previous = anchors[index - 1]
    const next = anchors[index]
    if (seconds <= next.seconds) {
      const span = next.seconds - previous.seconds
      if (span <= 0) return next.position
      return (
        previous.position +
        ((seconds - previous.seconds) / span) * (next.position - previous.position)
      )
    }
  }
  return last.position
}
