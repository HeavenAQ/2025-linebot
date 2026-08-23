/**
 * ISO week labels, matching the `YYYY-Www` form the Go backend writes with
 * `time.ISOWeek`. Reflections are stored under this label, so the two sides
 * have to agree on which week a date belongs to — including the turn of the
 * year, where the ISO year is not always the calendar year.
 */

/** Parses a portfolio key ("2026-08-19-14-30") into a local Date. */
export function parseWorkDate(key: string): Date | null {
  const [year, month, day, hour, minute] = key.split('-').map(Number)
  if (!year || !month || !day) return null
  const date = new Date(year, month - 1, day, hour || 0, minute || 0)
  return Number.isNaN(date.getTime()) ? null : date
}

/** The ISO week a date falls in, as { year, week }. */
export function isoWeekParts(date: Date): { year: number; week: number } {
  // Shift to the Thursday of this week: the ISO year is whichever year that
  // Thursday lands in, which is what makes 1 Jan sometimes belong to week 52
  // or 53 of the year before.
  const shifted = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()))
  const weekday = shifted.getUTCDay() || 7
  shifted.setUTCDate(shifted.getUTCDate() + 4 - weekday)
  const yearStart = new Date(Date.UTC(shifted.getUTCFullYear(), 0, 1))
  const week = Math.ceil(((shifted.getTime() - yearStart.getTime()) / 86_400_000 + 1) / 7)
  return { year: shifted.getUTCFullYear(), week }
}

/** The label a week is stored under, e.g. "2026-W34". */
export function isoWeek(date: Date): string {
  const { year, week } = isoWeekParts(date)
  return `${year}-W${String(week).padStart(2, '0')}`
}

/** Monday and Sunday of an ISO week label, for showing a human date range. */
export function weekRange(label: string): { start: Date; end: Date } | null {
  const match = /^(\d{4})-W(\d{2})$/.exec(label)
  if (!match) return null
  const year = Number(match[1])
  const week = Number(match[2])
  // 4 January is always in ISO week 1, so walk from the Monday of that week.
  const fourth = new Date(Date.UTC(year, 0, 4))
  const firstMonday = new Date(fourth)
  firstMonday.setUTCDate(fourth.getUTCDate() - ((fourth.getUTCDay() || 7) - 1))
  const start = new Date(firstMonday)
  start.setUTCDate(firstMonday.getUTCDate() + (week - 1) * 7)
  const end = new Date(start)
  end.setUTCDate(start.getUTCDate() + 6)
  return { start, end }
}

/** "8/17 – 8/23", the way a student recognises a week. */
export function formatWeekRange(label: string): string {
  const range = weekRange(label)
  if (!range) return label
  const format = (date: Date) => `${date.getUTCMonth() + 1}/${date.getUTCDate()}`
  return `${format(range.start)} – ${format(range.end)}`
}
