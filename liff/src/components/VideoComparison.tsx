'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Captions, Maximize2, Pause, Play, RotateCcw } from 'lucide-react'

import AutoHeight from '@/components/ui/auto-height'
import { Button } from '@/components/ui/button'
import { Segmented } from '@/components/ui/segmented'
import { expertMotionWindow } from '@/lib/expertAlignment'
import type { CoachingCue, PhaseMarker, PlaybackResponse } from '@/types'

type ViewMode = 'both' | 'student' | 'expert'

const VIEW_OPTIONS = [
  { value: 'both', label: '雙畫面' },
  { value: 'student', label: '學員' },
  { value: 'expert', label: '專家' }
] as const satisfies readonly { value: ViewMode; label: string }[]

interface VideoComparisonProps {
  playback: PlaybackResponse
}

/** What the caption is saying right now: a correction, or the current phase. */
interface Caption {
  title: string
  body: string
  live: boolean
}

interface PauseInterval {
  start: number
  end: number
  duration: number
  position: number
  cue: CoachingCue
}

const clamp = (value: number) => Math.min(1, Math.max(0, value))

/** How often the scrubber's React state follows the playhead, in ms. */
const PROGRESS_STATE_INTERVAL = 66

/**
 * Slack for deciding a checkpoint has been reached, in normalized position.
 * A frame of the 64-frame timeline is about 0.016, so this is well inside one.
 */
const CHECKPOINT_REACHED_EPSILON = 0.002

/** Portrait phone footage, used until a video reports its real dimensions. */
const FALLBACK_RATIO = 3 / 4

/** A video's own aspect ratio, preferring what the element actually loaded. */
const videoRatio = (element: HTMLVideoElement, current: number) =>
  element.videoWidth > 0 && element.videoHeight > 0
    ? element.videoWidth / element.videoHeight
    : current

const metadataRatio = (width: number, height: number) =>
  width > 0 && height > 0 ? width / height : FALLBACK_RATIO

const formatTime = (seconds: number) => {
  if (!Number.isFinite(seconds)) return '0:00'
  const rounded = Math.max(0, Math.floor(seconds))
  return `${Math.floor(rounded / 60)}:${String(rounded % 60).padStart(2, '0')}`
}

export default function VideoComparison({ playback }: VideoComparisonProps) {
  const studentRef = useRef<HTMLVideoElement>(null)
  const expertRef = useRef<HTMLVideoElement>(null)
  const playingRef = useRef(false)
  const lastProgressAtRef = useRef(0)
  const previousViewModeRef = useRef<ViewMode>('both')
  const [playing, setPlaying] = useState(false)
  const [progress, setProgress] = useState(0)
  const [expertProgress, setExpertProgress] = useState(0)
  const [studentDuration, setStudentDuration] = useState(playback.student_video.duration_seconds)
  const [expertDuration, setExpertDuration] = useState(playback.expert.video.duration_seconds)
  const [viewMode, setViewMode] = useState<ViewMode>('both')
  const [captionsOn, setCaptionsOn] = useState(true)
  const [caption, setCaption] = useState<Caption | null>(null)
  const [studentRatio, setStudentRatio] = useState(() =>
    metadataRatio(playback.student_video.width, playback.student_video.height)
  )
  const [expertRatio, setExpertRatio] = useState(() =>
    metadataRatio(playback.expert.video.width, playback.expert.video.height)
  )
  const [activeCue, setActiveCue] = useState<CoachingCue | null>(playback.coaching_cues[0] ?? null)
  const expertOnly = viewMode === 'expert'

  const pauses = useMemo<PauseInterval[]>(() => {
    const unique = new Map<number, CoachingCue>()
    for (const cue of playback.coaching_cues) {
      const current = unique.get(cue.normalized_frame)
      if (!current || cue.pause_duration_seconds > current.pause_duration_seconds) {
        unique.set(cue.normalized_frame, cue)
      }
    }
    return [...unique.values()]
      .sort((a, b) => a.student_timestamp_seconds - b.student_timestamp_seconds)
      .map(cue => ({
        start: cue.student_timestamp_seconds,
        end: cue.student_timestamp_seconds + cue.pause_duration_seconds,
        duration: cue.pause_duration_seconds,
        position: clamp(cue.normalized_position),
        cue
      }))
  }, [playback.coaching_cues])

  const totalPauseDuration = useMemo(
    () => pauses.reduce((total, pause) => total + pause.duration, 0),
    [pauses]
  )
  const motionDuration = Math.max(0.01, studentDuration - totalPauseDuration)
  const { start: expertMotionStart, end: expertMotionEnd } = expertMotionWindow(
    expertDuration,
    playback.expert.motion_start_seconds,
    playback.expert.motion_end_seconds
  )
  const expertMotionSpan = Math.max(0.01, expertMotionEnd - expertMotionStart)

  const expertTimeFromMotionProgress = useCallback(
    (position: number) => expertMotionStart + clamp(position) * expertMotionSpan,
    [expertMotionSpan, expertMotionStart]
  )

  const pauseAtTime = useCallback(
    (time: number) => pauses.find(pause => time >= pause.start && time < pause.end),
    [pauses]
  )

  // The caption always says where the stroke is. A correction takes it over for
  // the two-second pause it occupies, and outside those the checkpoint the
  // playhead has reached holds the line — so it reads as a commentary that runs
  // to the end of the motion, rather than freezing on whichever correction
  // happened to come last.
  const updateCaption = useCallback(
    (studentTime: number, position: number) => {
      const live = pauseAtTime(studentTime)
      if (live) {
        setCaption({ title: live.cue.title, body: live.cue.feedback, live: true })
        return
      }
      // Checkpoints are listed in scoring order, not stroke order, so the one
      // reached most recently is the furthest along that the playhead has
      // passed — not the last one in the list.
      //
      // "Passed" needs slack. Seeking to a checkpoint sets a video time that
      // converts back a hair short of the position asked for — a seek to
      // 0.3650794 came back as 0.3650625 — and on an exact test the checkpoint
      // just jumped to counts as not yet reached, so the caption named the one
      // before it. The tolerance is a small fraction of a frame of 63.
      let reached: PhaseMarker | null = null
      for (const marker of playback.timeline) {
        if (marker.normalized_position > position + CHECKPOINT_REACHED_EPSILON) continue
        if (!reached || marker.normalized_position >= reached.normalized_position) reached = marker
      }
      setCaption(reached ? { title: reached.label, body: '', live: false } : null)
    },
    [pauseAtTime, playback.timeline]
  )

  const updateExpertCaption = useCallback(
    (seconds: number) => {
      let reached: PhaseMarker | null = null
      for (const marker of playback.expert.timeline) {
        if (marker.timestamp_seconds > seconds + 1 / 120) continue
        if (!reached || marker.timestamp_seconds >= reached.timestamp_seconds) reached = marker
      }
      setCaption(reached ? { title: reached.label, body: '', live: false } : null)
    },
    [playback.expert.timeline]
  )

  const motionProgressFromStudentTime = useCallback(
    (time: number) => {
      let completedPauseDuration = 0
      for (const pause of pauses) {
        if (time < pause.start) break
        if (time < pause.end) return pause.position
        completedPauseDuration += pause.duration
      }
      return clamp((time - completedPauseDuration) / motionDuration)
    },
    [motionDuration, pauses]
  )

  const studentTimeFromMotionProgress = useCallback(
    (position: number) => {
      let time = clamp(position) * motionDuration
      for (const pause of pauses) {
        if (pause.position < position - 0.0001) time += pause.duration
      }
      return Math.min(studentDuration, time)
    },
    [motionDuration, pauses, studentDuration]
  )

  const syncExpert = useCallback(
    (_position: number, studentTime: number) => {
      const expert = expertRef.current
      if (!expert || expertMotionEnd <= expertMotionStart) return
      expert.playbackRate = 1
      if (pauseAtTime(studentTime)) {
        expert.pause()
      } else if (playingRef.current && expert.paused) {
        void expert.play().catch(() => undefined)
      }
    },
    [expertMotionEnd, expertMotionStart, pauseAtTime]
  )

  const seek = useCallback(
    (position: number, cue?: CoachingCue) => {
      const next = clamp(position)
      const student = studentRef.current
      const expert = expertRef.current
      const studentTime = cue?.student_timestamp_seconds ?? studentTimeFromMotionProgress(next)
      if (student) student.currentTime = studentTime
      if (expert && expertMotionEnd > expertMotionStart) {
        expert.currentTime = expertTimeFromMotionProgress(next)
      }
      setProgress(next)
      setExpertProgress(next)
      if (cue) {
        setActiveCue(cue)
        setCaption({ title: cue.title, body: cue.feedback, live: true })
      } else {
        updateCaption(studentTime, next)
      }
    },
    [
      expertMotionEnd,
      expertMotionStart,
      expertTimeFromMotionProgress,
      studentTimeFromMotionProgress,
      updateCaption
    ]
  )

  const setPlayback = useCallback(
    (shouldPlay: boolean) => {
      const student = studentRef.current
      const expert = expertRef.current
      if (!expert || (!expertOnly && !student)) return
      playingRef.current = shouldPlay
      setPlaying(shouldPlay)
      if (!shouldPlay) {
        student?.pause()
        expert.pause()
        return
      }
      if (expertOnly) {
        student?.pause()
        expert.playbackRate = 1
        if (
          expert.currentTime < expertMotionStart ||
          expert.currentTime >= expertMotionEnd - 1 / 120
        ) {
          expert.currentTime = expertMotionStart
          setProgress(0)
          setExpertProgress(0)
          updateExpertCaption(expertMotionStart)
        }
        void expert.play().catch(() => {
          playingRef.current = false
          setPlaying(false)
        })
        return
      }
      if (!student) return
      if (student.ended || progress >= 0.999) seek(0)
      student.playbackRate = 1
      expert.playbackRate = 1
      if (expert.currentTime >= expertMotionEnd - 1 / 120) {
        expert.currentTime = expertMotionStart
      }
      void student.play().catch(() => {
        playingRef.current = false
        setPlaying(false)
      })
      if (!pauseAtTime(student.currentTime)) void expert.play().catch(() => undefined)
    },
    [
      expertMotionEnd,
      expertMotionStart,
      expertOnly,
      pauseAtTime,
      progress,
      seek,
      updateExpertCaption
    ]
  )

  // Following the playhead on `timeupdate` alone is too coarse to hold two
  // clips together: browsers fire it about four times a second, so the expert
  // spends most of playback correcting a drift it only just noticed. While
  // playing, sync runs every frame instead, and the scrubber's React state is
  // throttled separately so re-rendering does not ride at 60fps.
  const followPlayhead = useCallback(
    (force: boolean) => {
      if (expertOnly) {
        const expert = expertRef.current
        if (!expert || expertMotionEnd <= expertMotionStart) return
        const seconds = Math.min(expertMotionEnd, Math.max(expertMotionStart, expert.currentTime))
        const next = clamp((seconds - expertMotionStart) / expertMotionSpan)
        const now = performance.now()
        if (force || now - lastProgressAtRef.current >= PROGRESS_STATE_INTERVAL) {
          lastProgressAtRef.current = now
          setExpertProgress(next)
        }
        updateExpertCaption(seconds)
        if (expert.currentTime >= expertMotionEnd - 1 / 120) {
          expert.pause()
          expert.currentTime = expertMotionEnd
          playingRef.current = false
          setPlaying(false)
          setProgress(1)
          setExpertProgress(1)
        }
        return
      }
      const student = studentRef.current
      if (!student) return
      const next = motionProgressFromStudentTime(student.currentTime)
      const expert = expertRef.current
      const nextExpert = expert
        ? clamp((expert.currentTime - expertMotionStart) / expertMotionSpan)
        : expertProgress
      const now = performance.now()
      if (force || now - lastProgressAtRef.current >= PROGRESS_STATE_INTERVAL) {
        lastProgressAtRef.current = now
        setProgress(next)
        setExpertProgress(nextExpert)
      }
      syncExpert(next, student.currentTime)
      updateCaption(student.currentTime, next)
    },
    [
      expertMotionEnd,
      expertMotionSpan,
      expertMotionStart,
      expertOnly,
      expertProgress,
      motionProgressFromStudentTime,
      syncExpert,
      updateCaption,
      updateExpertCaption
    ]
  )

  useEffect(() => {
    if (!playing) return
    let frame = requestAnimationFrame(function step() {
      followPlayhead(false)
      frame = requestAnimationFrame(step)
    })
    return () => cancelAnimationFrame(frame)
  }, [followPlayhead, playing])

  useEffect(() => {
    playingRef.current = false
    setPlaying(false)
    setProgress(0)
    setExpertProgress(0)
    setStudentDuration(playback.student_video.duration_seconds)
    setExpertDuration(playback.expert.video.duration_seconds)
    setActiveCue(playback.coaching_cues[0] ?? null)
    setCaption(null)
    setStudentRatio(metadataRatio(playback.student_video.width, playback.student_video.height))
    setExpertRatio(metadataRatio(playback.expert.video.width, playback.expert.video.height))
  }, [playback])

  const showStudent = viewMode !== 'expert'
  const showExpert = viewMode !== 'student'

  // Watching the expert alone means reading the expert's stroke, so the
  // timeline switches to that expert's own checkpoints: same criteria as the
  // student's -- both timelines come from the skill's qualitative feedback
  // rules, marker for marker -- but timed where this expert reaches each one.
  // Analyses recorded before checkpoint alignment carry no expert timeline, and
  // those keep the student's axis.
  const expertTimeline =
    playback.expert.timeline.length === playback.timeline.length ? playback.expert.timeline : null
  const onExpertAxis = viewMode === 'expert' && expertTimeline !== null
  const expertAxisPosition = useCallback(
    (seconds: number) => clamp((seconds - expertMotionStart) / expertMotionSpan),
    [expertMotionSpan, expertMotionStart]
  )

  const checkpoints = useMemo(() => {
    // The expert defines the canonical phase sequence. A learner can perform
    // checkpoints late, early, or out of order; the student timestamps below
    // are seek targets only and must never reorder the UI.
    const source = expertTimeline ?? playback.timeline
    return source
      .map(marker => {
        const studentMarker = playback.timeline.find(candidate => candidate.id === marker.id)
        const expertMarker = playback.expert.timeline.find(candidate => candidate.id === marker.id)
        return {
          id: marker.id,
          label: marker.label,
          position: clamp(studentMarker?.normalized_position ?? marker.normalized_position),
          studentPosition: clamp(studentMarker?.normalized_position ?? marker.normalized_position),
          studentSeconds: studentMarker?.timestamp_seconds,
          expertSeconds: expertMarker?.timestamp_seconds,
          expertPosition: expertMarker
            ? expertAxisPosition(expertMarker.timestamp_seconds)
            : clamp(marker.normalized_position),
          expertOrderSeconds: expertMarker?.timestamp_seconds ?? marker.timestamp_seconds
        }
      })
      .sort((left, right) => left.expertOrderSeconds - right.expertOrderSeconds)
  }, [expertAxisPosition, expertTimeline, playback.expert.timeline, playback.timeline])

  const seekCheckpoint = useCallback(
    (marker: (typeof checkpoints)[number]) => {
      const student = studentRef.current
      const expert = expertRef.current
      if (student && marker.studentSeconds !== undefined) {
        student.currentTime = Math.min(studentDuration, Math.max(0, marker.studentSeconds))
      }
      if (expert && marker.expertSeconds !== undefined) {
        expert.currentTime = Math.min(
          expertMotionEnd,
          Math.max(expertMotionStart, marker.expertSeconds)
        )
      }
      setProgress(marker.studentPosition)
      setExpertProgress(marker.expertPosition)
      setCaption({ title: marker.label, body: '', live: false })
    },
    [checkpoints, expertMotionEnd, expertMotionStart, studentDuration]
  )

  const seekExpertTrack = useCallback(
    (position: number) => {
      const seconds = expertMotionStart + clamp(position) * expertMotionSpan
      const expert = expertRef.current
      if (expert) {
        expert.playbackRate = 1
        expert.currentTime = seconds
      }
      setExpertProgress(clamp(position))
      if (expertOnly) updateExpertCaption(seconds)
    },
    [expertMotionSpan, expertMotionStart, expertOnly, updateExpertCaption]
  )

  useEffect(() => {
    // Metadata changes (especially the expert's exact duration arriving after
    // Play) also rebuild the alignment anchors. They must not be treated as a
    // tab switch: doing so paused both videos immediately after playback began.
    if (previousViewModeRef.current === viewMode) return
    previousViewModeRef.current = viewMode
    const student = studentRef.current
    const expert = expertRef.current
    student?.pause()
    expert?.pause()
    playingRef.current = false
    setPlaying(false)
    if (expertOnly && expert) {
      const seconds = Math.min(
        expertMotionEnd,
        Math.max(expertMotionStart, expert.currentTime || expertMotionStart)
      )
      expert.playbackRate = 1
      expert.currentTime = seconds
      setProgress(expertAxisPosition(seconds))
      setExpertProgress(expertAxisPosition(seconds))
      updateExpertCaption(seconds)
    }
  }, [
    expertAxisPosition,
    expertMotionEnd,
    expertMotionStart,
    expertOnly,
    updateExpertCaption,
    viewMode
  ])

  const reachedCheckpointPosition = (
    current: number,
    positionOf: (marker: (typeof checkpoints)[number]) => number
  ) => {
    const reached = checkpoints.reduce((latest, marker) => {
      const position = positionOf(marker)
      return position <= current + CHECKPOINT_REACHED_EPSILON ? Math.max(latest, position) : latest
    }, Number.NEGATIVE_INFINITY)
    return Number.isFinite(reached) ? reached : checkpoints[0] ? positionOf(checkpoints[0]) : 0
  }
  const studentCheckpointPosition = reachedCheckpointPosition(
    progress,
    marker => marker.studentPosition
  )
  const expertCheckpointPosition = reachedCheckpointPosition(
    expertProgress,
    marker => marker.expertPosition
  )
  const isCurrentCheckpoint = (marker: (typeof checkpoints)[number]) => {
    const position = expertOnly ? marker.expertPosition : marker.studentPosition
    const current = expertOnly ? expertCheckpointPosition : studentCheckpointPosition
    return Math.abs(position - current) <= CHECKPOINT_REACHED_EPSILON
  }
  const checkpointDifferenceLabel = (marker: (typeof checkpoints)[number]) => {
    const difference = marker.studentPosition - marker.expertPosition
    if (Math.abs(difference) < 0.015) return '時機接近專家'
    return `學員${difference > 0 ? '較晚' : '較早'} ${Math.round(Math.abs(difference) * 100)}%`
  }

  return (
    // The player is a panel like any other: same surface, same border, same
    // radius. Only the video frames themselves are black, because footage needs
    // a neutral backing — everything around them belongs to the page.
    <section className="glass overflow-hidden rounded-xl text-card-foreground">
      <div className="flex items-start justify-between gap-3 p-4">
        <div className="min-w-0">
          <h2 className="text-base font-semibold tracking-tight">動作同步比較</h2>
          <p className="mt-1 truncate text-xs text-muted-foreground">
            {playback.handedness === 'left' ? '左手' : '右手'} · 專家 {playback.expert.display_name}{' '}
            · 骨架距離 {playback.expert.correction_distance.toFixed(3)}
          </p>
        </div>
        <div className="shrink-0 text-right">
          <span className="text-2xl font-semibold tabular-nums text-primary">
            {playback.grade.total_grade.toFixed(1)}
          </span>
          <span className="ml-1 text-xs text-muted-foreground">分</span>
        </div>
      </div>

      <Segmented
        label="畫面模式"
        size="sm"
        options={VIEW_OPTIONS}
        value={viewMode}
        onChange={setViewMode}
        className="mx-4 mb-3"
      />

      {/* Both clips stay side by side at every width — comparing is the whole
          point, and stacking them on a phone puts the two halves of the
          comparison a scroll apart. Each frame takes its own video's aspect
          ratio, so the footage fills it exactly with no letterboxing. */}
      <AutoHeight className="mx-3">
        <div
          className={`grid gap-1.5 ${showStudent && showExpert ? 'grid-cols-2' : 'grid-cols-1'}`}
        >
          <div className={showStudent ? 'relative overflow-hidden rounded-lg' : 'hidden'}>
            <span className="absolute left-1.5 top-1.5 z-10 rounded bg-black/65 px-1.5 py-0.5 text-[11px] font-medium text-white backdrop-blur-sm">
              學員
            </span>
            <video
              ref={studentRef}
              src={playback.student_video.signed_url}
              className="w-full object-contain"
              style={{ aspectRatio: studentRatio }}
              playsInline
              muted
              preload="metadata"
              onLoadedMetadata={event => {
                setStudentDuration(event.currentTarget.duration)
                setStudentRatio(videoRatio(event.currentTarget, studentRatio))
              }}
              onTimeUpdate={() => {
                if (!expertOnly) followPlayhead(true)
              }}
              onEnded={() => setPlayback(false)}
              onClick={() => setPlayback(!playingRef.current)}
            />
          </div>
          <div className={showExpert ? 'relative overflow-hidden rounded-lg' : 'hidden'}>
            <span className="absolute left-1.5 top-1.5 z-10 rounded bg-black/65 px-1.5 py-0.5 text-[11px] font-medium text-white backdrop-blur-sm">
              專家
            </span>
            <video
              ref={expertRef}
              src={playback.expert.video.signed_url}
              className="w-full object-contain"
              style={{ aspectRatio: expertRatio }}
              playsInline
              muted
              preload="metadata"
              onLoadedMetadata={event => {
                setExpertDuration(event.currentTarget.duration)
                setExpertRatio(videoRatio(event.currentTarget, expertRatio))
                event.currentTarget.currentTime = Math.min(
                  event.currentTarget.duration,
                  Math.max(0, playback.expert.motion_start_seconds)
                )
              }}
              onTimeUpdate={() => {
                if (expertOnly) followPlayhead(true)
              }}
              onEnded={() => {
                if (expertOnly) setPlayback(false)
              }}
              onClick={() => setPlayback(!playingRef.current)}
            />
          </div>
        </div>
      </AutoHeight>

      {/* The caption carries the AI's correction for the moment on screen. It
          sits under the frames rather than over them: at half a phone's width
          an overlay would cover the very joints it is talking about. */}
      <AutoHeight className="mx-3">
        {captionsOn ? (
          <div className="mt-1.5 rounded-lg bg-neutral-900 px-3 py-2.5 text-white">
            {caption ? (
              <>
                <p className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-white/60">
                  <span
                    className={`h-1.5 w-1.5 rounded-full ${caption.live ? 'bg-destructive' : 'bg-white/40'}`}
                  />
                  {caption.title}
                </p>
                {caption.body && <p className="mt-1 text-[15px] leading-6">{caption.body}</p>}
              </>
            ) : (
              <p className="text-[13px] leading-6 text-white/55">
                {expertOnly
                  ? '播放後會在這裡顯示專家技術檢核點'
                  : playback.coaching_cues.length > 0
                    ? '播放後會在這裡顯示 AI 提醒'
                    : '這次分析沒有需要修正的地方'}
              </p>
            )}
          </div>
        ) : null}
      </AutoHeight>

      <div className="p-4">
        <div className="relative space-y-3">
          {showStudent && showExpert && (
            <svg
              aria-hidden="true"
              viewBox="0 0 100 48"
              preserveAspectRatio="none"
              className="pointer-events-none absolute left-14 right-0 top-4 z-0 h-12 w-[calc(100%-3.5rem)] overflow-visible"
            >
              {checkpoints.map(marker => (
                <line
                  key={marker.id}
                  x1={marker.studentPosition * 100}
                  y1="0"
                  x2={marker.expertPosition * 100}
                  y2="48"
                  vectorEffect="non-scaling-stroke"
                  className="stroke-primary/30"
                  strokeWidth="1.5"
                  strokeDasharray="3 3"
                />
              ))}
            </svg>
          )}

          {showStudent && (
            <div className="relative z-10 flex h-9 items-center gap-2">
              <span className="w-12 shrink-0 text-xs font-semibold text-primary">學員</span>
              <div className="relative h-8 flex-1">
                <input
                  aria-label="學員動作時間軸"
                  type="range"
                  min="0"
                  max="1000"
                  value={Math.round(progress * 1000)}
                  onChange={event => seek(Number(event.target.value) / 1000)}
                  className="absolute inset-x-0 top-2 h-2 w-full cursor-pointer accent-primary"
                />
                {playback.coaching_cues.map((cue, index) => (
                  <button
                    key={`${cue.normalized_frame}-${index}`}
                    type="button"
                    title={cue.title}
                    aria-label={`前往問題：${cue.title}`}
                    onClick={() => seek(cue.normalized_position, cue)}
                    className="absolute top-0 h-4 w-4 -translate-x-2 rounded-full border-2 border-card bg-destructive"
                    style={{ left: `${clamp(cue.normalized_position) * 100}%` }}
                  />
                ))}
                {checkpoints.map((marker, index) => (
                  <button
                    key={marker.id}
                    type="button"
                    title={`學員：${marker.label}`}
                    aria-label={`前往學員與專家的${marker.label}`}
                    onClick={() => seekCheckpoint(marker)}
                    className="absolute top-0 z-10 flex h-5 w-5 -translate-x-1/2 items-center justify-center rounded-full border-2 border-card bg-primary text-[10px] font-semibold text-primary-foreground shadow-sm"
                    style={{ left: `${marker.studentPosition * 100}%` }}
                  >
                    {index + 1}
                  </button>
                ))}
              </div>
            </div>
          )}

          {showExpert && (
            <div className="relative z-10 flex h-9 items-center gap-2">
              <span className="w-12 shrink-0 text-xs font-semibold text-success">專家</span>
              <div className="relative h-8 flex-1">
                <input
                  aria-label="專家動作時間軸"
                  type="range"
                  min="0"
                  max="1000"
                  value={Math.round(expertProgress * 1000)}
                  onChange={event => seekExpertTrack(Number(event.target.value) / 1000)}
                  className="absolute inset-x-0 top-2 h-2 w-full cursor-pointer accent-success"
                />
                {checkpoints.map((marker, index) => (
                  <button
                    key={marker.id}
                    type="button"
                    title={`專家：${marker.label}`}
                    aria-label={`前往學員與專家的${marker.label}`}
                    onClick={() => seekCheckpoint(marker)}
                    className="absolute top-0 z-10 flex h-5 w-5 -translate-x-1/2 items-center justify-center rounded-full border-2 border-card bg-success text-[10px] font-semibold text-white shadow-sm"
                    style={{ left: `${marker.expertPosition * 100}%` }}
                  >
                    {index + 1}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="icon"
            title={playing ? '暫停' : '播放'}
            aria-label={playing ? '暫停' : '播放'}
            onClick={() => setPlayback(!playing)}
          >
            {playing ? <Pause size={18} /> : <Play size={18} />}
          </Button>
          <Button
            variant="outline"
            size="icon"
            title="重新播放"
            aria-label="重新播放"
            onClick={() => seek(0)}
          >
            <RotateCcw size={17} />
          </Button>
          <span className="ml-1 text-xs tabular-nums text-muted-foreground">
            {onExpertAxis
              ? `${formatTime(expertProgress * expertMotionSpan)} / ${formatTime(expertMotionSpan)}`
              : `${formatTime(studentTimeFromMotionProgress(progress))} / ${formatTime(studentDuration)}`}
          </span>
          <Button
            variant={captionsOn ? 'primary' : 'outline'}
            size="icon"
            className="ml-auto"
            title={captionsOn ? '關閉 AI 字幕' : '開啟 AI 字幕'}
            aria-label={captionsOn ? '關閉 AI 字幕' : '開啟 AI 字幕'}
            aria-pressed={captionsOn}
            onClick={() => setCaptionsOn(value => !value)}
          >
            <Captions size={17} />
          </Button>
          <Button
            variant="outline"
            size="icon"
            title={expertOnly ? '全螢幕查看專家影片' : '全螢幕查看學員影片'}
            aria-label={expertOnly ? '全螢幕查看專家影片' : '全螢幕查看學員影片'}
            onClick={() =>
              void (expertOnly ? expertRef.current : studentRef.current)?.requestFullscreen?.()
            }
          >
            <Maximize2 size={17} />
          </Button>
        </div>

        <div className="mt-3">
          <p className="mb-2 text-xs font-medium text-foreground">
            技術檢核點{onExpertAxis ? '（專家）' : ''}
          </p>
          <div className="flex snap-x gap-1 overflow-x-auto pb-2" aria-label="技術檢核點">
            {checkpoints.map((marker, index) => (
              <button
                key={marker.id}
                type="button"
                onClick={() => seekCheckpoint(marker)}
                className={`flex min-w-[9.5rem] snap-start items-center gap-2 border-b-2 px-1 py-2 text-left text-xs transition-colors ${
                  isCurrentCheckpoint(marker)
                    ? 'border-primary text-foreground'
                    : 'border-transparent text-muted-foreground'
                }`}
              >
                <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-success text-[10px] font-semibold text-white">
                  {index + 1}
                </span>
                <span className="leading-4">
                  <span className="block">{marker.label}</span>
                  <span className="mt-0.5 block text-[10px] font-normal text-muted-foreground">
                    {checkpointDifferenceLabel(marker)}
                  </span>
                </span>
              </button>
            ))}
          </div>
          {!onExpertAxis && pauses.length > 0 && (
            <span className="mt-1 flex items-center gap-1.5 text-[11px] text-muted-foreground">
              <span className="h-2 w-2 rounded-full bg-destructive" /> GPT 暫停點
            </span>
          )}
        </div>

        {/* Every correction as a full, tappable entry. The caption shows one at
            a time while playing; this is where a student reads them all without
            scrubbing the timeline to find each one. */}
        {!onExpertAxis && playback.coaching_cues.length > 0 && (
          <div className="mt-4 border-t border-border pt-4">
            <p className="eyebrow mb-2">AI 修正建議</p>
            <ul className="space-y-2">
              {playback.coaching_cues.map((cue, index) => {
                const isActive = activeCue === cue
                return (
                  <li key={`${cue.normalized_frame}-${index}`}>
                    <button
                      type="button"
                      onClick={() => seek(cue.normalized_position, cue)}
                      className={`w-full rounded-lg border p-3 text-left transition-colors ${
                        isActive ? 'border-destructive bg-destructive/5' : 'border-border'
                      }`}
                    >
                      <span className="flex items-baseline gap-2">
                        <span className="mt-0.5 h-2 w-2 shrink-0 rounded-full bg-destructive" />
                        <span className="text-sm font-semibold">{cue.title}</span>
                        <span className="ml-auto shrink-0 text-[11px] tabular-nums text-muted-foreground">
                          {formatTime(cue.student_timestamp_seconds)}
                        </span>
                      </span>
                      <span className="mt-1 block pl-4 text-sm leading-6 text-muted-foreground">
                        {cue.feedback}
                      </span>
                    </button>
                  </li>
                )
              })}
            </ul>
          </div>
        )}
      </div>
    </section>
  )
}
