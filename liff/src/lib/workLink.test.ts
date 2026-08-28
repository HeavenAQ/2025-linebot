import assert from 'node:assert/strict'
import test from 'node:test'

import { resolveWorkFocus } from './workLink.ts'
import type { Portfolios, Work } from '@/schemas/userData.schema.ts'

const work = { date: '2026-08-19-14-30' } as Work

const portfolio = {
  serve: { '2026-08-19-14-30': work },
  smash: {},
  clear: {},
  lift: {}
} as unknown as Portfolios

test('opens the attempt the link names', () => {
  assert.deepEqual(resolveWorkFocus('?tab=review&skill=serve&date=2026-08-19-14-30', portfolio), {
    skill: 'serve',
    date: '2026-08-19-14-30'
  })
})

test('ignores a link with no work on it', () => {
  assert.equal(resolveWorkFocus('?tab=review', portfolio), null)
  assert.equal(resolveWorkFocus('?tab=review&skill=serve', portfolio), null)
  assert.equal(resolveWorkFocus('?tab=review&date=2026-08-19-14-30', portfolio), null)
})

// The card outlives the record: the work may have been removed, or the link
// may name a skill this student never practised.
test('falls back when the attempt is not in the portfolio', () => {
  assert.equal(resolveWorkFocus('?skill=serve&date=2020-01-01-00-00', portfolio), null)
  assert.equal(resolveWorkFocus('?skill=smash&date=2026-08-19-14-30', portfolio), null)
})

test('rejects a skill that is not a skill', () => {
  assert.equal(resolveWorkFocus('?skill=badminton&date=2026-08-19-14-30', portfolio), null)
  assert.equal(resolveWorkFocus('?skill=constructor&date=2026-08-19-14-30', portfolio), null)
})

test('rejects an inherited property posing as a work date', () => {
  assert.equal(resolveWorkFocus('?skill=serve&date=constructor', portfolio), null)
})
