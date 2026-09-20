const isFiniteNumber = (value: number) => typeof value === 'number' && Number.isFinite(value)

interface ExpertMotionWindow {
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
