import type { AlignmentSample, PhaseMarker } from '@/types'

/**
 * Maps the student's motion progress onto the expert video's clock.
 *
 * The two performances do not share a tempo: a learner may spend 40% of the
 * stroke winding up where the expert spends 25%, so stretching the expert's
 * motion window evenly across the student's progress drifts out of sync
 * between checkpoints. Anchoring the map on matching checkpoints and
 * interpolating within each segment keeps every phase aligned — the playback
 * counterpart of the segmental warping used when scoring.
 *
 * The analysis now warps the poses between those checkpoints too and sends the
 * result as dense samples, because tempo is not constant inside a phase either.
 * Those samples are anchors like any other; the checkpoint pairs remain as the
 * fallback for analyses that carry no warp.
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
 * Grows the anchor list from correspondences ordered by student progress.
 *
 * Pairs are dropped rather than reordered when they disagree with what came
 * before, because a non-monotonic anchor would rewind the expert mid-play, and
 * a flat one would stall it. Both `expertTimeAt` and its inverse rely on the
 * result increasing strictly in each axis.
 */
function accumulateAnchors(
  pairs: readonly AlignmentAnchor[],
  motionStart: number,
  motionEnd: number
): AlignmentAnchor[] {
  const anchors: AlignmentAnchor[] = [{ position: 0, seconds: motionStart }]
  for (const pair of pairs) {
    const previous = anchors[anchors.length - 1]
    const seconds = Math.min(motionEnd, Math.max(motionStart, pair.seconds))
    if (pair.position > previous.position && seconds > previous.seconds) {
      anchors.push({ position: pair.position, seconds })
    }
  }

  const last = anchors[anchors.length - 1]
  if (last.position < 1 && motionEnd > last.seconds) {
    anchors.push({ position: 1, seconds: motionEnd })
  } else if (last.position >= 1) {
    // An anchor already landed on the end of the motion; keep it as the tail.
    last.seconds = Math.max(last.seconds, motionEnd)
  }
  return anchors
}

const usablePairs = (pairs: readonly AlignmentAnchor[]) =>
  pairs
    .filter(pair => isFiniteNumber(pair.position) && isFiniteNumber(pair.seconds))
    .sort((a, b) => a.position - b.position)

/**
 * Builds the anchors for one analysis, from the densest map it was given.
 *
 * The warped samples already carry the checkpoints as fixed points, so when
 * they are present they replace the checkpoint pairs rather than joining them.
 * Analyses recorded before the warp shipped, and any the service could not
 * warp, fall back to the checkpoints; with neither this degrades to the two
 * window ends, which is exactly the old whole-window stretch.
 */
export function buildAlignmentAnchors(
  studentTimeline: readonly PhaseMarker[],
  expertTimeline: readonly PhaseMarker[],
  motionStart: number,
  motionEnd: number,
  alignment: readonly AlignmentSample[] = []
): AlignmentAnchor[] {
  if (alignment.length > 0) {
    return accumulateAnchors(
      usablePairs(
        alignment.map(sample => ({
          position: clamp01(sample.normalized_position),
          seconds: sample.expert_seconds
        }))
      ),
      motionStart,
      motionEnd
    )
  }
  if (studentTimeline.length !== expertTimeline.length) {
    return accumulateAnchors([], motionStart, motionEnd)
  }
  return accumulateAnchors(
    usablePairs(
      studentTimeline.map((student, index) => ({
        position: clamp01(student.normalized_position),
        seconds: expertTimeline[index].timestamp_seconds
      }))
    ),
    motionStart,
    motionEnd
  )
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

export interface ExpertMotionWindow {
  start: number
  end: number
}

/**
 * The stretch of the expert clip that the learner's motion maps onto.
 *
 * `duration` is what the video element has loaded, which is zero until it has:
 * an expert matched out of the reference bank is served from a signed URL with
 * no probed metadata, so nothing is known about the clip up front. Clamping the
 * window to a zero duration collapses it to a point, which pins the expert on
 * its first frame and reads as a clip that never plays — so while the duration
 * is unknown the analysis's own window is what bounds it.
 */
export function expertMotionWindow(
  duration: number,
  configuredStart: number,
  configuredEnd: number
): ExpertMotionWindow {
  const known = isFiniteNumber(duration) && duration > 0
  const start = known
    ? Math.min(duration, Math.max(0, configuredStart))
    : Math.max(0, configuredStart)
  if (configuredEnd > start) {
    return { start, end: known ? Math.min(duration, configuredEnd) : configuredEnd }
  }
  return { start, end: known ? duration : start }
}
