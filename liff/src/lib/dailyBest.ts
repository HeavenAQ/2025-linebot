/**
 * One point per practice day for the score trend: that day's best attempt.
 *
 * A student often uploads several attempts in one session, and plotting every
 * attempt turns a semester into an unreadable zigzag. The best of the day is
 * what they reached that day, so it is the progress worth charting.
 */

export interface DailyBest {
  /** "YYYY-MM-DD" */
  day: string
  totalGrade: number
}

/** Portfolio keys start with "YYYY-MM-DD", whatever time or suffix follows. */
const dayOf = (workDate: string): string | null => {
  const match = /^(\d{4})-(\d{1,2})-(\d{1,2})(?:-|$)/.exec(workDate)
  if (!match) return null
  return `${match[1]}-${match[2].padStart(2, '0')}-${match[3].padStart(2, '0')}`
}

/** Oldest day first. Attempts without a readable date are left out. */
export function dailyBestScores(
  attempts: readonly { date: string; totalGrade: number }[]
): DailyBest[] {
  const best = new Map<string, number>()
  for (const { date, totalGrade } of attempts) {
    const day = dayOf(date)
    if (!day || !Number.isFinite(totalGrade)) continue
    const current = best.get(day)
    if (current === undefined || totalGrade > current) best.set(day, totalGrade)
  }
  return [...best.entries()]
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
    .map(([day, totalGrade]) => ({ day, totalGrade }))
}
