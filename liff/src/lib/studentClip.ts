/**
 * Which render of the student's own stroke is playing.
 *
 * `coach` is the analysis render with the AI coach's corrections written into
 * it, which freezes on each one. `analysis` is the same skeleton overlay
 * without those pauses, so the stroke runs at its real speed.
 */
export type StudentSource = 'coach' | 'analysis'

interface ClipChoice {
  /** The student's pick in the player. */
  source: StudentSource
  /** A single checkpoint is being replayed on a loop. */
  loopActive: boolean
  /** The plain analysis render exists for this attempt. */
  hasAnalysisRender: boolean
}

/**
 * playsAnalysisRender reports whether the pause-free render should play.
 *
 * Replaying one checkpoint always uses it, whatever the student picked: the
 * coach's pauses sit inside the range being looped and would stall every lap.
 * Older attempts have no plain render at all, so the coach's version plays.
 */
export function playsAnalysisRender({
  source,
  loopActive,
  hasAnalysisRender
}: ClipChoice): boolean {
  return hasAnalysisRender && (source === 'analysis' || loopActive)
}

/** hasAnalysisRender reports whether this attempt has a playable plain render. */
export function hasAnalysisRender(playback: {
  skeleton_overlay_video?: { signed_url?: string } | null
}): boolean {
  return Boolean(playback.skeleton_overlay_video?.signed_url)
}
