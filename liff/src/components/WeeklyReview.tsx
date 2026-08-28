'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { MessageCircleQuestion, Play } from 'lucide-react'

import type { PlaybackResponse, UserData } from '@/types'
import { Alert } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Segmented } from '@/components/ui/segmented'
import Spinner from '@/components/ui/spinner'
import Toast from '@/components/ui/toast'
import VideoComparison from '@/components/VideoComparison'
import { fetchPlayback } from '@/lib/api/fetchPlayback'
import {
  fetchWeeklyReflections,
  saveWeeklyNote,
  type WeeklyNoteField,
  type WeeklyReflection
} from '@/lib/api/weeklyReflections'
import { SkillNameMap, type Skill } from '@/lib/types'
import { authorizedFetch } from '@/lib/api/client'
import { SCORE_RECORD_LABEL, type ChatMessage } from '@/lib/useSkillSummary'
import { formatWeekRange, isoWeek, parseWorkDate } from '@/lib/week'
import { REVIEW_SECTIONS, type ReviewSection, type WorkFocus } from '@/lib/workLink'

interface WeeklyReviewProps {
  userId: string
  userData: UserData
  /**
   * One attempt to open on arrival, named by the link the student followed from
   * LINE. Passed in rather than read from the URL here so this stays a
   * component that shows what it is given.
   */
  focusWork?: WorkFocus | null
  /**
   * The sub-tab to open on arrival, named by the same link. Null leaves the
   * student on 反思, which is where most of them are heading.
   */
  focusSection?: ReviewSection | null
}

interface WeekEntry {
  skill: Skill
  workDate: string
  at: Date
  totalGrade: number
}

interface QuestionPair {
  question: string
  answer: string
  skill: string
  at: Date
}

const SKILLS = Object.keys(SkillNameMap) as Skill[]

/**
 * The two notes a week holds, and how each one is presented. They are the same
 * week from either end -- what the student meant to work on, and how it went --
 * so they share a week picker and a save path and differ only in wording.
 */
const SECTIONS: Record<
  ReviewSection,
  {
    field: WeeklyNoteField
    label: string
    heading: string
    placeholder: string
    saveLabel: string
    savedToast: string
  }
> = {
  reflection: {
    field: 'note',
    label: '反思',
    heading: '本週反思',
    placeholder: '這週練習下來，哪裡進步了？哪裡還要加強？下週想先做什麼？',
    saveLabel: '儲存反思',
    savedToast: '本週反思已儲存'
  },
  preview: {
    field: 'preview',
    label: '預習',
    heading: '課前檢視要點',
    placeholder: '下次上課前想先盯住哪些重點？例如：引拍高度、擊球點、重心轉移。',
    saveLabel: '儲存預習',
    savedToast: '課前檢視要點已儲存'
  }
}

const SECTION_OPTIONS = REVIEW_SECTIONS.map(value => ({ value, label: SECTIONS[value].label }))

const formatDay = (date: Date) => `${date.getMonth() + 1}/${date.getDate()}`

/** Groups every graded attempt into the ISO week it was recorded in. */
function entriesByWeek(userData: UserData): Map<string, WeekEntry[]> {
  const weeks = new Map<string, WeekEntry[]>()
  for (const skill of SKILLS) {
    for (const [workDate, work] of Object.entries(userData.portfolio[skill] ?? {})) {
      const at = parseWorkDate(workDate)
      if (!at) continue
      const label = isoWeek(at)
      const entry: WeekEntry = {
        skill,
        workDate,
        at,
        totalGrade: work.grading_outcome.total_grade
      }
      weeks.set(label, [...(weeks.get(label) ?? []), entry])
    }
  }
  for (const entries of weeks.values()) entries.sort((a, b) => b.at.getTime() - a.at.getTime())
  return weeks
}

/**
 * Pairs the learner's questions with the coach's answers for each week.
 *
 * History is stored as a flat run of turns, so an answer is whatever assistant
 * turn follows a question. A question with nothing after it is still shown —
 * it is a thing the student asked.
 */
function questionsByWeek(messages: readonly ChatMessage[]): Map<string, QuestionPair[]> {
  const weeks = new Map<string, QuestionPair[]>()
  messages.forEach((message, index) => {
    if (message.role !== 'user' || !message.text) return
    // The score record the bot filed on the learner's behalf is not a question.
    if (message.text === SCORE_RECORD_LABEL) return
    const at = message.timestamp ? new Date(message.timestamp) : null
    if (!at || Number.isNaN(at.getTime())) return
    const next = messages[index + 1]
    const label = isoWeek(at)
    const pair: QuestionPair = {
      question: message.text,
      answer: next && next.role === 'assistant' ? next.text : '',
      skill: message.skill,
      at
    }
    weeks.set(label, [...(weeks.get(label) ?? []), pair])
  })
  return weeks
}

export default function WeeklyReview({
  userId,
  userData,
  focusWork,
  focusSection
}: WeeklyReviewProps) {
  // Every skill's questions, not just the one selected on the page above: a
  // week's review covers whatever the student practised that week.
  const [messages, setMessages] = useState<ChatMessage[]>([])

  useEffect(() => {
    let cancelled = false
    const query = new URLSearchParams({ user_id: userId })
    authorizedFetch(`/api/chat/history?${query}`)
      .then(response => (response.ok ? response.json() : { data: [] }))
      .then(json => {
        if (!cancelled) setMessages(Array.isArray(json.data) ? json.data : [])
      })
      .catch(() => undefined)
    return () => {
      cancelled = true
    }
  }, [userId])

  const weeks = useMemo(() => entriesByWeek(userData), [userData])
  const questions = useMemo(() => questionsByWeek(messages), [messages])

  // Every week that has anything in it, newest first.
  const weekLabels = useMemo(() => {
    const labels = new Set([...weeks.keys(), ...questions.keys()])
    return [...labels].sort().reverse()
  }, [questions, weeks])

  const [selectedWeek, setSelectedWeek] = useState('')
  const [openWork, setOpenWork] = useState<WeekEntry | null>(null)
  const [playback, setPlayback] = useState<PlaybackResponse | null>(null)
  const [playbackError, setPlaybackError] = useState('')
  const [playbackLoading, setPlaybackLoading] = useState(false)
  const [reflections, setReflections] = useState<Record<string, WeeklyReflection>>({})
  const [section, setSection] = useState<ReviewSection>('reflection')
  // What the student has typed but not saved, per note, for the week on screen.
  // Held as overrides on top of the stored record rather than as copies of it,
  // so saving one note leaves an edit in progress on the other one alone.
  const [drafts, setDrafts] = useState<Partial<Record<WeeklyNoteField, string>>>({})
  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState('')
  const [toast, setToast] = useState('')

  const week = selectedWeek || weekLabels[0] || ''
  const copy = SECTIONS[section]
  const stored = reflections[week]?.[copy.field] ?? ''
  const draft = drafts[copy.field] ?? stored
  const dirty = draft !== stored

  useEffect(() => {
    let cancelled = false
    fetchWeeklyReflections(userId)
      .then(value => {
        if (!cancelled) setReflections(value)
      })
      .catch(() => undefined)
    return () => {
      cancelled = true
    }
  }, [userId])

  // Both editors start again from what is stored whenever the week changes.
  useEffect(() => {
    setDrafts({})
    setSaveError('')
  }, [week])

  // A link can name the sub-tab as well as the week, so a learner sent here to
  // plan the next lesson lands on 預習. Only its arrival moves them; from then
  // on they switch freely.
  useEffect(() => {
    if (focusSection) setSection(focusSection)
  }, [focusSection])

  // The week a deep link opened, so the reset below can tell that week's video
  // being opened for the student apart from the student leaving the week.
  const focusedWeek = useRef('')

  // A link from a portfolio card names one attempt: show the week it belongs to
  // with that video already expanded, so the feedback and the reflection box
  // are on screen together. Only its own arrival triggers this — from then on
  // the student navigates freely.
  useEffect(() => {
    if (!focusWork) return
    const at = parseWorkDate(focusWork.date)
    if (!at) return
    const label = isoWeek(at)
    const entry = weeks
      .get(label)
      ?.find(
        candidate => candidate.skill === focusWork.skill && candidate.workDate === focusWork.date
      )
    if (!entry) return
    focusedWeek.current = label
    setSelectedWeek(label)
    setOpenWork(entry)
  }, [focusWork, weeks])

  // Changing week puts the list back to its collapsed state.
  useEffect(() => {
    if (focusedWeek.current === week) return
    setOpenWork(null)
    setPlayback(null)
  }, [week])

  useEffect(() => {
    if (!openWork) {
      setPlayback(null)
      return
    }
    let cancelled = false
    setPlaybackLoading(true)
    setPlaybackError('')
    fetchPlayback(userId, openWork.skill, openWork.workDate)
      .then(value => {
        if (!cancelled) setPlayback(value)
      })
      .catch(error => {
        if (!cancelled) {
          setPlayback(null)
          setPlaybackError(error instanceof Error ? error.message : '無法載入影片')
        }
      })
      .finally(() => {
        if (!cancelled) setPlaybackLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [openWork, userId])

  const onSave = useCallback(async () => {
    if (!week) return
    setSaving(true)
    setSaveError('')
    try {
      // The answer carries the whole week back, including the note this save
      // did not touch, so the other editor keeps showing what is stored.
      const saved = await saveWeeklyNote(userId, week, copy.field, draft)
      setReflections(current => ({ ...current, [week]: saved }))
      setDrafts(current => ({ ...current, [copy.field]: undefined }))
      setToast(copy.savedToast)
    } catch (error) {
      setSaveError(error instanceof Error ? error.message : '儲存失敗')
    } finally {
      setSaving(false)
    }
  }, [copy, draft, userId, week])

  if (weekLabels.length === 0) {
    return (
      <Alert title="還沒有可回顧的紀錄">
        上傳練習影片或與教練機器人對話後，這裡會以每週為單位整理你的學習。
      </Alert>
    )
  }

  const entries = weeks.get(week) ?? []
  const weekQuestions = questions.get(week) ?? []
  return (
    <div className="space-y-5">
      {/* Weeks scroll sideways so a whole semester stays reachable with a thumb
          instead of pushing the content off the screen. */}
      <div className="-mx-4 flex snap-x gap-2 overflow-x-auto px-4 pb-1" aria-label="選擇週次">
        {weekLabels.map(label => {
          const isActive = label === week
          return (
            <button
              key={label}
              type="button"
              onClick={() => setSelectedWeek(label)}
              aria-current={isActive}
              className={`shrink-0 snap-start rounded-full border px-3.5 py-2 text-[13px] transition-colors ${
                isActive
                  ? 'border-primary bg-primary text-primary-foreground'
                  : 'border-border text-muted-foreground'
              }`}
            >
              {formatWeekRange(label)}
              {reflections[label]?.note || reflections[label]?.preview ? (
                <span className={isActive ? 'ml-1.5' : 'ml-1.5 text-primary'}>·</span>
              ) : null}
            </button>
          )
        })}
      </div>

      <section>
        <h3 className="eyebrow mb-2">本週練習</h3>
        {entries.length === 0 ? (
          <p className="text-[13px] text-muted-foreground">這週沒有上傳影片。</p>
        ) : (
          <ul className="space-y-2">
            {entries.map(entry => {
              const isOpen =
                openWork?.workDate === entry.workDate && openWork?.skill === entry.skill
              return (
                <li key={`${entry.skill}-${entry.workDate}`}>
                  <button
                    type="button"
                    onClick={() => setOpenWork(isOpen ? null : entry)}
                    aria-expanded={isOpen}
                    className={`flex w-full items-center gap-3 rounded-lg border p-3 text-left transition-colors ${
                      isOpen ? 'border-primary' : 'border-border'
                    }`}
                  >
                    <Play size={15} className="shrink-0 text-primary" aria-hidden />
                    <span className="text-sm font-medium">{SkillNameMap[entry.skill]}</span>
                    <span className="text-[13px] text-muted-foreground">{formatDay(entry.at)}</span>
                    <span className="num ml-auto font-data text-sm font-semibold">
                      {entry.totalGrade.toFixed(1)}
                    </span>
                  </button>

                  {isOpen && (
                    <div className="mt-2">
                      {playbackLoading && <Spinner />}
                      {!playbackLoading && playback && <VideoComparison playback={playback} />}
                      {!playbackLoading && playbackError && (
                        <Alert variant="warning" title="無法載入影片">
                          {playbackError}
                        </Alert>
                      )}
                    </div>
                  )}
                </li>
              )
            })}
          </ul>
        )}
      </section>

      <section>
        <h3 className="eyebrow mb-2 flex items-center gap-1.5">
          <MessageCircleQuestion size={15} className="text-primary" aria-hidden />
          本週向 AI 提問
          {weekQuestions.length > 0 && (
            <span className="text-muted-foreground">（{weekQuestions.length}）</span>
          )}
        </h3>
        {weekQuestions.length === 0 ? (
          <p className="text-[13px] text-muted-foreground">這週沒有提問紀錄。</p>
        ) : (
          <ul className="space-y-3">
            {weekQuestions.map((pair, index) => (
              <li
                key={`${pair.at.toISOString()}-${index}`}
                className="rounded-lg border border-border p-3"
              >
                <p className="text-sm font-medium leading-6">{pair.question}</p>
                {pair.answer && (
                  <p className="mt-1.5 whitespace-pre-line border-t border-border pt-1.5 text-[13px] leading-6 text-muted-foreground">
                    {pair.answer}
                  </p>
                )}
                <p className="mt-1.5 text-[11px] text-muted-foreground">
                  {SkillNameMap[pair.skill as Skill] || pair.skill} · {formatDay(pair.at)}
                </p>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="space-y-3">
        <Segmented
          role="tablist"
          label="本週筆記"
          size="sm"
          options={SECTION_OPTIONS}
          value={section}
          onChange={setSection}
        />
        <div role="tabpanel">
          <h3 className="eyebrow mb-2">{copy.heading}</h3>
          <textarea
            value={draft}
            onChange={event =>
              setDrafts(current => ({ ...current, [copy.field]: event.target.value }))
            }
            rows={5}
            maxLength={4000}
            placeholder={copy.placeholder}
            className="w-full rounded-lg border border-border bg-background p-3 text-[15px] leading-7 outline-none focus:border-primary"
          />
          <div className="mt-2 flex items-center gap-3">
            <Button onClick={onSave} disabled={saving || !dirty}>
              {saving ? '儲存中…' : copy.saveLabel}
            </Button>
            {dirty && !saving && (
              <span className="text-[13px] text-muted-foreground">尚未儲存</span>
            )}
            <span className="ml-auto text-[11px] text-muted-foreground">{draft.length} / 4000</span>
          </div>
          {saveError && (
            <p className="mt-2 text-[13px] text-destructive" role="alert">
              {saveError}
            </p>
          )}
        </div>
      </section>

      <Toast message={toast} onDismiss={() => setToast('')} />
    </div>
  )
}
