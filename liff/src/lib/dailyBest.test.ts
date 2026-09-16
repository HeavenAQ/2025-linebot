import assert from 'node:assert/strict'
import test from 'node:test'

import { dailyBestScores } from './dailyBest.ts'

test('keeps only the highest score of each day, oldest day first', () => {
  assert.deepEqual(
    dailyBestScores([
      { date: '2026-09-14-14-20', totalGrade: 61 },
      { date: '2026-09-07-14-05', totalGrade: 40 },
      { date: '2026-09-14-14-05-31-8f3a2c1d', totalGrade: 72.5 },
      { date: '2026-09-14-15-40', totalGrade: 55 },
      { date: '2026-09-07-15-00', totalGrade: 38 }
    ]),
    [
      { day: '2026-09-07', totalGrade: 40 },
      { day: '2026-09-14', totalGrade: 72.5 }
    ]
  )
})

test('leaves out attempts without a readable date or score', () => {
  assert.deepEqual(
    dailyBestScores([
      { date: 'legacy', totalGrade: 90 },
      { date: '2026-09-14-14-20', totalGrade: Number.NaN },
      { date: '2026-09-15-10-00', totalGrade: 0 }
    ]),
    [{ day: '2026-09-15', totalGrade: 0 }]
  )
})
