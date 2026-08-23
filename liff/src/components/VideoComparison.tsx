'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Captions, Maximize2, Pause, Play, RotateCcw } from 'lucide-react'

import AutoHeight from '@/components/ui/auto-height'
import { Button } from '@/components/ui/button'
import { Segmented } from '@/components/ui/segmented'
import {
  buildAlignmentAnchors,
  expertRateAt,
  expertTimeAt,
  progressAtExpertTime
} from '@/lib/expertAlignment'
import type { CoachingCue, PlaybackResponse } from '@/types'

type ViewMode = 'both' | 'student' | 'expert'

const VIEW_OPTIONS = [
  { value: 'both', label: '雙畫面' },
  { value: 'student', label: '學員' },
  { value: 'expert', label: '專家' }
] as const satisfies readonly { value: ViewMode; label: string }[]

interface VideoComparisonProps {
  playback: PlaybackResponse
}

interface PauseInterval {
  start: number
  end: number
  duration: number
  position: number
  cue: CoachingCue
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
  const [captionCue, setCaptionCue] = useState<CoachingCue | null>(null)
  const [captionLive, setCaptionLive] = useState(false)
  const [studentRatio, setStudentRatio] = useState(() =>
    metadataRatio(playback.student_video.width, playback.student_video.height)
  )
  const [expertRatio, setExpertRatio] = useState(() =>
    metadataRatio(playback.expert.video.width, playback.expert.video.height)
  )
  const [activeCue, setActiveCue] = useState<CoachingCue | null>(playback.coaching_cues[0] ?? null)

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
  const expertMotionStart = Math.min(
    expertDuration,
    Math.max(0, playback.expert.motion_start_seconds)
  )
  const configuredExpertEnd = playback.expert.motion_end_seconds
  const expertMotionEnd = Math.min(
    expertDuration,
    configuredExpertEnd > expertMotionStart ? configuredExpertEnd : expertDuration
  )
  // Checkpoint-anchored map from the student's progress to the expert's clock.
  // Falls back to the plain window stretch when an analysis carries no expert
  // checkpoints.
  const alignmentAnchors = useMemo(
    () =>
      buildAlignmentAnchors(
        playback.timeline,
        playback.expert.timeline,
        expertMotionStart,
        expertMotionEnd
      ),
    [playback.timeline, playback.expert.timeline, expertMotionStart, expertMotionEnd]
  )

  const expertTimeFromMotionProgress = useCallback(
    (position: number) => expertTimeAt(alignmentAnchors, position),
    [alignmentAnchors]
  )

  const pauseAtTime = useCallback(
    (time: number) => pauses.find(pause => time >= pause.start && time < pause.end),
    [pauses]
  )

  // Captions hold the last correction the playhead reached rather than clearing
  // between cues: each one only occupies its two-second pause, and text that
  // blinks in and out is harder to read than text that stays put.
  const updateCaption = useCallback(
    (studentTime: number) => {
      const live = pauseAtTime(studentTime)
      if (live) {
        setCaptionCue(live.cue)
        setCaptionLive(true)
        return
      }
      setCaptionLive(false)
      const passed = pauses.filter(pause => studentTime >= pause.start)
      setCaptionCue(passed.length > 0 ? passed[passed.length - 1].cue : null)
    },
    [pauseAtTime, pauses]
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
    (position: number, studentTime: number) => {
      const expert = expertRef.current
      if (!expert || !Number.isFinite(expertDuration) || expertDuration <= 0) return
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
      if (pauseAtTime(studentTime)) {
        expert.pause()
      } else if (playingRef.current && expert.paused) {
        void expert.play().catch(() => undefined)
      }
    },
    [alignmentAnchors, expertDuration, expertTimeFromMotionProgress, motionDuration, pauseAtTime]
  )

  const seek = useCallback(
    (position: number, cue?: CoachingCue) => {
      const next = clamp(position)
      const student = studentRef.current
      const expert = expertRef.current
      const studentTime = cue?.student_timestamp_seconds ?? studentTimeFromMotionProgress(next)
      if (student) student.currentTime = studentTime
      if (expert && expertDuration > 0) {
        expert.currentTime = expertTimeFromMotionProgress(next)
      }
      setProgress(next)
      if (cue) {
        setActiveCue(cue)
        setCaptionCue(cue)
        setCaptionLive(true)
      } else {
        updateCaption(studentTime)
      }
    },
    [expertDuration, expertTimeFromMotionProgress, studentTimeFromMotionProgress, updateCaption]
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
      if (!pauseAtTime(student.currentTime)) void expert.play().catch(() => undefined)
    },
    [pauseAtTime, progress, seek]
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
      syncExpert(next, student.currentTime)
      updateCaption(student.currentTime)
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
    setActiveCue(playback.coaching_cues[0] ?? null)
    setCaptionCue(null)
    setCaptionLive(false)
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

  const activeCheckpoint =
    checkpoints.length === 0
      ? -1
      : checkpoints.reduce(
          (nearest, marker, index) =>
            Math.abs(marker.position - axisProgress) <
            Math.abs(checkpoints[nearest].position - axisProgress)
              ? index
              : nearest,
          0
        )

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

      {/* The caption carries the AI's correction for the moment on screen. It
          sits under the frames rather than over them: at half a phone's width
          an overlay would cover the very joints it is talking about. */}
      <AutoHeight className="mx-3">
        {captionsOn ? (
          <div className="mt-1.5 rounded-lg bg-neutral-900 px-3 py-2.5 text-white">
            {captionCue ? (
              <>
                <p className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-white/60">
                  <span
                    className={`h-1.5 w-1.5 rounded-full ${captionLive ? 'bg-destructive' : 'bg-white/40'}`}
                  />
                  {captionCue.title}
                </p>
                <p className="mt-1 text-[15px] leading-6">{captionCue.feedback}</p>
              </>
            ) : (
              <p className="text-[13px] leading-6 text-white/55">
                {playback.coaching_cues.length > 0
                  ? '播放後會在這裡顯示 AI 提醒'
                  : '這次分析沒有需要修正的地方'}
              </p>
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
              className="absolute top-0 flex h-5 w-5 -translate-x-1/2 items-center justify-center rounded-full border-2 border-card bg-success text-[10px] font-semibold text-white shadow-sm transition-[left] duration-200"
              style={{ left: `${marker.position * 100}%` }}
            >
              {index + 1}
            </button>
          ))}
          {playback.coaching_cues.map((cue, index) => (
            <button
              key={`${cue.normalized_frame}-${index}`}
              type="button"
              title={cue.title}
              aria-label={`前往問題：${cue.title}`}
              onClick={() => seek(cue.normalized_position, cue)}
              className="absolute top-0 h-4 w-4 -translate-x-2 rounded-full border-2 border-card bg-destructive transition-[left] duration-200"
              style={{
                left: `${
                  (onExpertAxis
                    ? expertAxisPosition(expertTimeFromMotionProgress(cue.normalized_position))
                    : clamp(cue.normalized_position)) * 100
                }%`
              }}
            />
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
                  index === activeCheckpoint
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
          {pauses.length > 0 && (
            <span className="mt-1 flex items-center gap-1.5 text-[11px] text-muted-foreground">
              <span className="h-2 w-2 rounded-full bg-destructive" /> GPT 暫停點
            </span>
          )}
        </div>

        {/* Every correction as a full, tappable entry. The caption shows one at
            a time while playing; this is where a student reads them all without
            scrubbing the timeline to find each one. */}
        {playback.coaching_cues.length > 0 && (
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
