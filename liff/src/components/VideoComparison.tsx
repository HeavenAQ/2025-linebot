'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Captions, Maximize2, Pause, Play, RotateCcw } from 'lucide-react'

import AutoHeight from '@/components/ui/auto-height'
import { Button } from '@/components/ui/button'
import { Segmented } from '@/components/ui/segmented'
import {
  buildAlignmentAnchors,
  expertMotionWindow,
  expertRateAt,
  expertTimeAt,
  progressAtExpertTime
} from '@/lib/expertAlignment'
import type { PhaseMarker, PlaybackResponse } from '@/types'

type ViewMode = 'both' | 'student' | 'expert'

const VIEW_OPTIONS = [
  { value: 'both', label: '雙畫面' },
  { value: 'student', label: '學員' },
  { value: 'expert', label: '專家' }
] as const satisfies readonly { value: ViewMode; label: string }[]

interface VideoComparisonProps {
  playback: PlaybackResponse
}

/** What the caption is saying right now: the checkpoint the playhead reached. */
interface Caption {
  title: string
}

const clamp = (value: number) => Math.min(1, Math.max(0, value))

/**
 * Drift, in expert-video seconds, past which the expert is seeked outright
 * rather than eased back. A visible jump beats a long, obviously-out-of-step
 * convergence; below it, trimming the rate is invisible where a seek stutters.
 */
const HARD_SEEK_DRIFT = 0.25

/** Wall-clock seconds the rate trim aims to close a small drift over. */
const DRIFT_CORRECTION_WINDOW = 0.5

/** Slowest and fastest playback a browser will honour smoothly. */
const clampRate = (rate: number) => Math.min(4, Math.max(0.25, rate))

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
  const [playing, setPlaying] = useState(false)
  const [progress, setProgress] = useState(0)
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
  // The render runs straight through, so a position maps onto the student clip
  // by proportion alone. The floor only keeps a clip whose metadata has not
  // loaded yet from dividing by zero.
  const motionDuration = Math.max(0.01, studentDuration)
  const { start: expertMotionStart, end: expertMotionEnd } = expertMotionWindow(
    expertDuration,
    playback.expert.motion_start_seconds,
    playback.expert.motion_end_seconds
  )
  // Map from the student's progress to the expert's clock: the analysis's
  // warped samples where it has them, the checkpoints alone where it does not,
  // and the plain window stretch when it carries neither.
  const alignmentAnchors = useMemo(
    () =>
      buildAlignmentAnchors(
        playback.timeline,
        playback.expert.timeline,
        expertMotionStart,
        expertMotionEnd,
        playback.expert.alignment
      ),
    [
      playback.timeline,
      playback.expert.timeline,
      playback.expert.alignment,
      expertMotionStart,
      expertMotionEnd
    ]
  )

  const expertTimeFromMotionProgress = useCallback(
    (position: number) => expertTimeAt(alignmentAnchors, position),
    [alignmentAnchors]
  )

  // The caption says where in the stroke the playhead is: the checkpoint it has
  // most recently reached, so it reads as a commentary that runs to the end of
  // the motion.
  const updateCaption = useCallback(
    (position: number) => {
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
      setCaption(reached ? { title: reached.label } : null)
    },
    [playback.timeline]
  )

  const motionProgressFromStudentTime = useCallback(
    (time: number) => clamp(time / motionDuration),
    [motionDuration]
  )

  const studentTimeFromMotionProgress = useCallback(
    (position: number) => Math.min(studentDuration, clamp(position) * motionDuration),
    [motionDuration, studentDuration]
  )

  const syncExpert = useCallback(
    (position: number) => {
      const expert = expertRef.current
      // Gated on the motion window rather than the duration: the window is what
      // playback actually needs, and it is known before the clip has loaded.
      if (!expert || expertMotionEnd <= expertMotionStart) return
      const target = expertTimeFromMotionProgress(position)
      // Each segment has its own tempo relative to the student -- the two
      // performances reach the same checkpoint at different points in their own
      // clips -- so the expert runs at that segment's rate rather than 1x.
      const base = expertRateAt(alignmentAnchors, position, motionDuration)
      const drift = expert.currentTime - target
      if (Math.abs(drift) > HARD_SEEK_DRIFT) {
        expert.currentTime = target
        if (Math.abs(expert.playbackRate - base) > 0.01) expert.playbackRate = base
      } else {
        // Residual drift is absorbed into the rate instead of a seek: running
        // fractionally slow or fast for half a second closes it without the
        // frame-skip a currentTime write causes mid-play.
        const trimmed = clampRate(base - drift / DRIFT_CORRECTION_WINDOW)
        if (Math.abs(expert.playbackRate - trimmed) > 0.01) expert.playbackRate = trimmed
      }
      if (playingRef.current && expert.paused) {
        void expert.play().catch(() => undefined)
      }
    },
    [
      alignmentAnchors,
      expertMotionEnd,
      expertMotionStart,
      expertTimeFromMotionProgress,
      motionDuration
    ]
  )

  const seek = useCallback(
    (position: number) => {
      const next = clamp(position)
      const student = studentRef.current
      const expert = expertRef.current
      if (student) student.currentTime = studentTimeFromMotionProgress(next)
      if (expert && expertMotionEnd > expertMotionStart) {
        expert.currentTime = expertTimeFromMotionProgress(next)
      }
      setProgress(next)
      updateCaption(next)
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
      if (!student || !expert) return
      playingRef.current = shouldPlay
      setPlaying(shouldPlay)
      if (!shouldPlay) {
        student.pause()
        expert.pause()
        return
      }
      if (student.ended || progress >= 0.999) seek(0)
      void student.play().catch(() => {
        playingRef.current = false
        setPlaying(false)
      })
      void expert.play().catch(() => undefined)
    },
    [progress, seek]
  )

  // Following the playhead on `timeupdate` alone is too coarse to hold two
  // clips together: browsers fire it about four times a second, so the expert
  // spends most of playback correcting a drift it only just noticed. While
  // playing, sync runs every frame instead, and the scrubber's React state is
  // throttled separately so re-rendering does not ride at 60fps.
  const followPlayhead = useCallback(
    (force: boolean) => {
      const student = studentRef.current
      if (!student) return
      const next = motionProgressFromStudentTime(student.currentTime)
      const now = performance.now()
      if (force || now - lastProgressAtRef.current >= PROGRESS_STATE_INTERVAL) {
        lastProgressAtRef.current = now
        setProgress(next)
      }
      syncExpert(next)
      updateCaption(next)
    },
    [motionProgressFromStudentTime, syncExpert, updateCaption]
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
    setStudentDuration(playback.student_video.duration_seconds)
    setExpertDuration(playback.expert.video.duration_seconds)
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
  const expertMotionSpan = Math.max(0.01, expertMotionEnd - expertMotionStart)
  const expertAxisPosition = useCallback(
    (seconds: number) => clamp((seconds - expertMotionStart) / expertMotionSpan),
    [expertMotionSpan, expertMotionStart]
  )

  const checkpoints = useMemo(() => {
    const source = onExpertAxis && expertTimeline ? expertTimeline : playback.timeline
    return source.map((marker, index) => ({
      id: marker.id,
      label: marker.label,
      // Where this checkpoint sits on the axis currently drawn.
      position:
        onExpertAxis && expertTimeline
          ? expertAxisPosition(marker.timestamp_seconds)
          : clamp(marker.normalized_position),
      // Seeking always speaks student progress: the student video drives
      // playback, and its position for checkpoint i lands the expert on that
      // very same checkpoint.
      seekTo: clamp(playback.timeline[index].normalized_position)
    }))
  }, [expertAxisPosition, expertTimeline, onExpertAxis, playback.timeline])

  // The playhead, expressed on whichever axis is on screen.
  const axisProgress = onExpertAxis
    ? expertAxisPosition(expertTimeFromMotionProgress(progress))
    : progress

  const seekOnAxis = useCallback(
    (position: number) => {
      if (!onExpertAxis) {
        seek(position)
        return
      }
      const seconds = expertMotionStart + clamp(position) * expertMotionSpan
      seek(progressAtExpertTime(alignmentAnchors, seconds))
    },
    [alignmentAnchors, expertMotionSpan, expertMotionStart, onExpertAxis, seek]
  )

  // Criteria can share an instant -- serve marks both 髖關節前旋 and 肩膀旋轉朝前
  // at the end of the motion -- so the nearest position can belong to more than
  // one checkpoint, and all of them are current. Singling one out meant the
  // other could never light up no matter where the playhead was.
  const nearestCheckpointDistance = checkpoints.reduce(
    (nearest, marker) => Math.min(nearest, Math.abs(marker.position - axisProgress)),
    Number.POSITIVE_INFINITY
  )
  const isCurrentCheckpoint = (position: number) =>
    Math.abs(position - axisProgress) === nearestCheckpointDistance

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
              onTimeUpdate={() => followPlayhead(true)}
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
              onClick={() => setPlayback(!playingRef.current)}
            />
          </div>
        </div>
      </AutoHeight>

      {/* The caption names the technical checkpoint on screen. It sits under
          the frames rather than over them: at half a phone's width an overlay
          would cover the very joints it is naming. */}
      <AutoHeight className="mx-3">
        {captionsOn ? (
          <div className="mt-1.5 rounded-lg bg-neutral-900 px-3 py-2.5 text-white">
            {caption ? (
              <p className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-white/60">
                <span className="h-1.5 w-1.5 rounded-full bg-white/40" />
                {caption.title}
              </p>
            ) : (
              <p className="text-[13px] leading-6 text-white/55">播放後會顯示目前的技術檢核點</p>
            )}
          </div>
        ) : null}
      </AutoHeight>

      <div className="p-4">
        <div className="relative h-9">
          <input
            aria-label={onExpertAxis ? '專家動作時間軸' : '動作時間軸'}
            type="range"
            min="0"
            max="1000"
            value={Math.round(axisProgress * 1000)}
            onChange={event => seekOnAxis(Number(event.target.value) / 1000)}
            className="absolute inset-x-0 top-2 h-2 w-full cursor-pointer accent-primary"
          />
          {checkpoints.map((marker, index) => (
            <button
              key={marker.id}
              type="button"
              title={marker.label}
              aria-label={`前往${marker.label}`}
              onClick={() => seek(marker.seekTo)}
              className="absolute top-0 z-10 flex h-5 w-5 -translate-x-1/2 items-center justify-center rounded-full border-2 border-card bg-success text-[10px] font-semibold text-white shadow-sm transition-[left] duration-200"
              style={{ left: `${marker.position * 100}%` }}
            >
              {index + 1}
            </button>
          ))}
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
              ? `${formatTime(axisProgress * expertMotionSpan)} / ${formatTime(expertMotionSpan)}`
              : `${formatTime(studentTimeFromMotionProgress(progress))} / ${formatTime(studentDuration)}`}
          </span>
          <Button
            variant={captionsOn ? 'primary' : 'outline'}
            size="icon"
            className="ml-auto"
            title={captionsOn ? '關閉字幕' : '開啟字幕'}
            aria-label={captionsOn ? '關閉字幕' : '開啟字幕'}
            aria-pressed={captionsOn}
            onClick={() => setCaptionsOn(value => !value)}
          >
            <Captions size={17} />
          </Button>
          <Button
            variant="outline"
            size="icon"
            title="全螢幕查看學員影片"
            aria-label="全螢幕查看學員影片"
            onClick={() => void studentRef.current?.requestFullscreen?.()}
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
                onClick={() => seek(marker.seekTo)}
                className={`flex min-w-[9.5rem] snap-start items-center gap-2 border-b-2 px-1 py-2 text-left text-xs transition-colors ${
                  isCurrentCheckpoint(marker.position)
                    ? 'border-primary text-foreground'
                    : 'border-transparent text-muted-foreground'
                }`}
              >
                <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-success text-[10px] font-semibold text-white">
                  {index + 1}
                </span>
                <span className="leading-4">{marker.label}</span>
              </button>
            ))}
          </div>
        </div>

      </div>
    </section>
  )
}
