import assert from 'node:assert/strict'
import test from 'node:test'

import {
  buildAlignmentAnchors,
  expertRateAt,
  expertTimeAt,
  progressAtExpertTime
} from './expertAlignment.ts'
import type { PhaseMarker } from '@/schemas/userData.schema.ts'

const marker = (id: string, position: number, seconds: number): PhaseMarker => ({
  id,
  label: id,
  normalized_frame: Math.round(position * 63),
  normalized_position: position,
  timestamp_seconds: seconds
})

// The student reaches impact at 60% of the motion; the expert reaches it at
// 2.0s of a 1.0-3.0s window, which is 50%.
const studentTimeline = [marker('ready', 0.2, 0), marker('impact', 0.6, 0)]
const expertTimeline = [marker('ready', 0.25, 1.5), marker('impact', 0.5, 2.0)]

test('anchors the expert clock on matching checkpoints', () => {
  const anchors = buildAlignmentAnchors(studentTimeline, expertTimeline, 1, 3)

  assert.deepEqual(anchors, [
    { position: 0, seconds: 1 },
    { position: 0.2, seconds: 1.5 },
    { position: 0.6, seconds: 2 },
    { position: 1, seconds: 3 }
  ])
})

test('puts the expert on its own checkpoint, not the proportional one', () => {
  const anchors = buildAlignmentAnchors(studentTimeline, expertTimeline, 1, 3)

  // A uniform stretch would have played 1 + 0.6 * 2 = 2.2s at the student's
  // impact, running the expert a fifth of a stroke ahead.
  assert.equal(expertTimeAt(anchors, 0.6), 2)
  assert.equal(expertTimeAt(anchors, 0.2), 1.5)
  // Between checkpoints the expert interpolates within that segment only.
  assert.equal(expertTimeAt(anchors, 0.4), 1.75)
  assert.equal(expertTimeAt(anchors, 0.8), 2.5)
})

test('clamps progress to the motion window', () => {
  const anchors = buildAlignmentAnchors(studentTimeline, expertTimeline, 1, 3)

  assert.equal(expertTimeAt(anchors, -1), 1)
  assert.equal(expertTimeAt(anchors, 5), 3)
})

test('falls back to the whole window without expert checkpoints', () => {
  const anchors = buildAlignmentAnchors(studentTimeline, [], 1, 3)

  assert.deepEqual(anchors, [
    { position: 0, seconds: 1 },
    { position: 1, seconds: 3 }
  ])
  assert.equal(expertTimeAt(anchors, 0.5), 2)
})

test('drops checkpoints that would rewind the expert', () => {
  const student = [marker('a', 0.3, 0), marker('b', 0.6, 0)]
  const expert = [marker('a', 0.3, 2.4), marker('b', 0.6, 1.8)]

  const anchors = buildAlignmentAnchors(student, expert, 1, 3)

  assert.deepEqual(anchors, [
    { position: 0, seconds: 1 },
    { position: 0.3, seconds: 2.4 },
    { position: 1, seconds: 3 }
  ])
})

test('ignores a mismatched expert timeline', () => {
  const anchors = buildAlignmentAnchors(studentTimeline, [marker('only', 0.5, 2)], 1, 3)

  assert.deepEqual(anchors, [
    { position: 0, seconds: 1 },
    { position: 1, seconds: 3 }
  ])
})

// Serve grades the hip rotation (keyframe 4) before the wrist flick (keyframe
// 3), and pairs of criteria share a keyframe, so the timeline arrives neither
// chronological nor distinct. Anchors must still come out ordered.
test('orders a serve timeline that is listed out of stroke order', () => {
  const student = [
    marker('arms_raised', 0.19, 0),
    marker('racket_foot_weight', 0.19, 0),
    marker('weight_transfer', 0.48, 0),
    marker('hip_rotation', 1, 0),
    marker('wrist_flick', 0.75, 0),
    marker('shoulder_rotation', 1, 0)
  ]
  const expert = [
    marker('arms_raised', 0.19, 1.3),
    marker('racket_foot_weight', 0.19, 1.3),
    marker('weight_transfer', 0.48, 1.9),
    marker('hip_rotation', 1, 3),
    marker('wrist_flick', 0.75, 2.5),
    marker('shoulder_rotation', 1, 3)
  ]

  const anchors = buildAlignmentAnchors(student, expert, 1, 3)

  assert.deepEqual(anchors, [
    { position: 0, seconds: 1 },
    { position: 0.19, seconds: 1.3 },
    { position: 0.48, seconds: 1.9 },
    { position: 0.75, seconds: 2.5 },
    { position: 1, seconds: 3 }
  ])
  // The wrist flick maps to the expert's wrist flick, not to whatever the
  // expert happened to be doing three quarters of the way through.
  assert.equal(expertTimeAt(anchors, 0.75), 2.5)
})

test('runs the expert at each segment own tempo', () => {
  const anchors = buildAlignmentAnchors(studentTimeline, expertTimeline, 1, 3)
  const motionDuration = 4

  // 0.5s of expert over 0.2 * 4s of student, and 0.5s over 0.4 * 4s after.
  assert.equal(expertRateAt(anchors, 0.1, motionDuration), 0.625)
  assert.equal(expertRateAt(anchors, 0.4, motionDuration), 0.3125)
  assert.equal(expertRateAt(anchors, 0.5, 0), 1)
})

test('keeps the requested rate inside what a browser will honour', () => {
  const anchors = [
    { position: 0, seconds: 0 },
    { position: 0.01, seconds: 2 },
    { position: 1, seconds: 2.01 }
  ]

  assert.equal(expertRateAt(anchors, 0.005, 1), 4)
  assert.equal(expertRateAt(anchors, 0.9, 1), 0.25)
})

test('maps an expert timestamp back to the student progress that reaches it', () => {
  const anchors = buildAlignmentAnchors(studentTimeline, expertTimeline, 1, 3)

  // Every anchor round-trips: the expert view scrubs on these seconds and the
  // student video has to land on the same checkpoint.
  for (const anchor of anchors) {
    assert.equal(progressAtExpertTime(anchors, anchor.seconds), anchor.position)
  }
  // Halfway between two anchors in seconds is halfway between them in progress.
  assert.equal(progressAtExpertTime(anchors, 1.75), 0.4)
})

test('clamps expert times outside the motion window to its ends', () => {
  const anchors = buildAlignmentAnchors(studentTimeline, expertTimeline, 1, 3)

  assert.equal(progressAtExpertTime(anchors, 0), 0)
  assert.equal(progressAtExpertTime(anchors, 99), 1)
})

test('inverts expertTimeAt across the whole motion', () => {
  const anchors = buildAlignmentAnchors(studentTimeline, expertTimeline, 1, 3)

  for (let step = 0; step <= 10; step += 1) {
    const position = step / 10
    const roundTripped = progressAtExpertTime(anchors, expertTimeAt(anchors, position))
    assert.ok(Math.abs(roundTripped - position) < 1e-9)
  }
})
