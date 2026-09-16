import assert from 'node:assert/strict'
import test from 'node:test'

import { expertMotionWindow } from './expertAlignment.ts'

test('keeps the expert motion window while the clip duration is unknown', () => {
  // A bank-matched expert arrives with duration 0; clamping to it collapsed the
  // window to a point and the expert sat on its first frame.
  assert.deepEqual(expertMotionWindow(0, 0.767, 2.267), { start: 0.767, end: 2.267 })
  assert.deepEqual(expertMotionWindow(Number.NaN, 0.767, 2.267), { start: 0.767, end: 2.267 })
})

test('bounds the expert motion window by the clip once it is known', () => {
  assert.deepEqual(expertMotionWindow(3.333, 0.767, 2.267), { start: 0.767, end: 2.267 })
  // A window claiming more than the clip holds is trimmed to the clip.
  assert.deepEqual(expertMotionWindow(2.0, 0.767, 2.267), { start: 0.767, end: 2.0 })
})

test('falls back to the whole clip when no expert window was recorded', () => {
  assert.deepEqual(expertMotionWindow(3.333, 0, 0), { start: 0, end: 3.333 })
  // Nothing known at all: an empty window, which callers treat as "not ready".
  assert.deepEqual(expertMotionWindow(0, 0, 0), { start: 0, end: 0 })
})
