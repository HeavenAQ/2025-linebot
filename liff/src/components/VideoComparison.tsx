'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Maximize2, Pause, Play, RotateCcw } from 'lucide-react'

import type { CoachingCue, PlaybackResponse } from '@/types'

type ViewMode = 'both' | 'student' | 'expert'

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

const formatTime = (seconds: number) => {
  if (!Number.isFinite(seconds)) return '0:00'
  const rounded = Math.max(0, Math.floor(seconds))
  return `${Math.floor(rounded / 60)}:${String(rounded % 60).padStart(2, '0')}`
}

export default function VideoComparison({ playback }: VideoComparisonProps) {
  const studentRef = useRef<HTMLVideoElement>(null)
  const expertRef = useRef<HTMLVideoElement>(null)
  const playingRef = useRef(false)
  const [playing, setPlaying] = useState(false)
  const [progress, setProgress] = useState(0)
  const [studentDuration, setStudentDuration] = useState(
    playback.student_video.duration_seconds
  )
  const [expertDuration, setExpertDuration] = useState(playback.expert.video.duration_seconds)
  const [viewMode, setViewMode] = useState<ViewMode>('both')
  const [activeCue, setActiveCue] = useState<CoachingCue | null>(
    playback.coaching_cues[0] ?? null
  )

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

  const pauseAtTime = useCallback(
    (time: number) => pauses.find(pause => time >= pause.start && time < pause.end),
    [pauses]
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
      const target = clamp(position) * expertDuration
      if (Math.abs(expert.currentTime - target) > 0.12) expert.currentTime = target
      if (pauseAtTime(studentTime)) {
        expert.pause()
      } else if (playingRef.current && expert.paused) {
        void expert.play().catch(() => undefined)
      }
    },
    [expertDuration, pauseAtTime]
  )

  const seek = useCallback(
    (position: number, cue?: CoachingCue) => {
      const next = clamp(position)
      const student = studentRef.current
      const expert = expertRef.current
      const studentTime = cue?.student_timestamp_seconds ?? studentTimeFromMotionProgress(next)
      if (student) student.currentTime = studentTime
      if (expert && expertDuration > 0) expert.currentTime = next * expertDuration
      setProgress(next)
      if (cue) setActiveCue(cue)
    },
    [expertDuration, studentTimeFromMotionProgress]
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

  const onStudentTimeUpdate = () => {
    const student = studentRef.current
    if (!student) return
    const next = motionProgressFromStudentTime(student.currentTime)
    setProgress(next)
    syncExpert(next, student.currentTime)
  }

  useEffect(() => {
    playingRef.current = false
    setPlaying(false)
    setProgress(0)
    setStudentDuration(playback.student_video.duration_seconds)
    setExpertDuration(playback.expert.video.duration_seconds)
    setActiveCue(playback.coaching_cues[0] ?? null)
  }, [playback])

  const showStudent = viewMode !== 'expert'
  const showExpert = viewMode !== 'student'

  return (
    <section className="mx-auto mt-5 w-full max-w-5xl border-y bg-zinc-950 text-white sm:border sm:border-zinc-800">
      <div className="flex items-start justify-between gap-3 px-4 py-3">
        <div className="min-w-0">
          <h2 className="text-base font-semibold">動作同步比較</h2>
          <p className="mt-1 truncate text-xs text-zinc-400">
            專家 {playback.expert.display_name} · 骨架距離{' '}
            {playback.expert.euclidean_distance.toFixed(3)}
          </p>
        </div>
        <div className="shrink-0 text-right">
          <span className="text-2xl font-semibold tabular-nums">
            {playback.grade.total_grade.toFixed(1)}
          </span>
          <span className="ml-1 text-xs text-zinc-400">分</span>
        </div>
      </div>

      <div className="mx-4 mb-3 grid h-9 grid-cols-3 border border-zinc-700 bg-zinc-900 p-0.5 text-xs">
        {(
          [
            ['both', '雙畫面'],
            ['student', '學員'],
            ['expert', '專家']
          ] as const
        ).map(([mode, label]) => (
          <button
            key={mode}
            type="button"
            aria-pressed={viewMode === mode}
            onClick={() => setViewMode(mode)}
            className={viewMode === mode ? 'bg-white font-medium text-zinc-950' : 'text-zinc-300'}
          >
            {label}
          </button>
        ))}
      </div>

      <div className={`grid bg-black ${showStudent && showExpert ? 'md:grid-cols-2' : ''}`}>
        <div className={showStudent ? 'relative' : 'hidden'}>
          <div className="absolute left-2 top-2 z-10 bg-black/80 px-2 py-1 text-xs">學員修正</div>
          <video
            ref={studentRef}
            src={playback.student_video.signed_url}
            className="aspect-[4/3] w-full bg-black object-contain"
            playsInline
            muted
            preload="metadata"
            onLoadedMetadata={event => setStudentDuration(event.currentTarget.duration)}
            onTimeUpdate={onStudentTimeUpdate}
            onEnded={() => setPlayback(false)}
            onClick={() => setPlayback(!playingRef.current)}
          />
        </div>
        <div className={showExpert ? 'relative border-t border-zinc-800 md:border-l md:border-t-0' : 'hidden'}>
          <div className="absolute left-2 top-2 z-10 bg-black/80 px-2 py-1 text-xs">最近專家</div>
          <video
            ref={expertRef}
            src={playback.expert.video.signed_url}
            className="aspect-[4/3] w-full bg-black object-contain"
            playsInline
            muted
            preload="metadata"
            onLoadedMetadata={event => setExpertDuration(event.currentTarget.duration)}
            onClick={() => setPlayback(!playingRef.current)}
          />
        </div>
      </div>

      <div className="px-4 pb-4 pt-3">
        <div className="relative h-8">
          <input
            aria-label="動作時間軸"
            type="range"
            min="0"
            max="1000"
            value={Math.round(progress * 1000)}
            onChange={event => seek(Number(event.target.value) / 1000)}
            className="absolute inset-x-0 top-2 h-2 w-full cursor-pointer accent-red-500"
          />
          {playback.timeline.map(marker => (
            <button
              key={marker.id}
              type="button"
              title={marker.label}
              aria-label={`前往${marker.label}`}
              onClick={() => seek(marker.normalized_position)}
              className="absolute top-0 h-4 w-px bg-emerald-400"
              style={{ left: `${clamp(marker.normalized_position) * 100}%` }}
            />
          ))}
          {playback.coaching_cues.map((cue, index) => (
            <button
              key={`${cue.normalized_frame}-${index}`}
              type="button"
              title={cue.title}
              aria-label={`前往問題：${cue.title}`}
              onClick={() => seek(cue.normalized_position, cue)}
              className="absolute top-0 h-4 w-4 -translate-x-2 border-2 border-white bg-red-500"
              style={{ left: `${clamp(cue.normalized_position) * 100}%` }}
            />
          ))}
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            title={playing ? '暫停' : '播放'}
            aria-label={playing ? '暫停' : '播放'}
            onClick={() => setPlayback(!playing)}
            className="grid h-10 w-10 place-items-center border border-zinc-600 bg-zinc-900"
          >
            {playing ? <Pause size={18} /> : <Play size={18} />}
          </button>
          <button
            type="button"
            title="重新播放"
            aria-label="重新播放"
            onClick={() => seek(0)}
            className="grid h-10 w-10 place-items-center border border-zinc-600 bg-zinc-900"
          >
            <RotateCcw size={17} />
          </button>
          <span className="ml-1 text-xs tabular-nums text-zinc-400">
            {formatTime(studentTimeFromMotionProgress(progress))} / {formatTime(studentDuration)}
          </span>
          <button
            type="button"
            title="全螢幕查看學員影片"
            aria-label="全螢幕查看學員影片"
            onClick={() => void studentRef.current?.requestFullscreen?.()}
            className="ml-auto grid h-10 w-10 place-items-center border border-zinc-600 bg-zinc-900"
          >
            <Maximize2 size={17} />
          </button>
        </div>

        <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-zinc-400">
          {playback.timeline.map(marker => (
            <span key={marker.id} className="flex items-center gap-1">
              <span className="h-2 w-2 bg-emerald-400" /> {marker.label}
            </span>
          ))}
          {pauses.length > 0 && (
            <span className="flex items-center gap-1">
              <span className="h-2 w-2 bg-red-500" /> GPT 暫停點
            </span>
          )}
        </div>

        {activeCue && (
          <div className="mt-4 border-l-4 border-red-500 pl-3">
            <p className="text-sm font-semibold text-red-300">{activeCue.title}</p>
            <p className="mt-1 text-sm leading-6 text-zinc-200">{activeCue.feedback}</p>
          </div>
        )}
      </div>
    </section>
  )
}
