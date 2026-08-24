import assert from 'node:assert/strict'
import test from 'node:test'

import {
  buildAlignmentAnchors,
  expertRateAt,
  expertMotionWindow,
  expertTimeAt,
  progressAtExpertTime
} from './expertAlignment.ts'
import type { AlignmentSample, PhaseMarker } from '@/schemas/userData.schema.ts'

const marker = (id: string, position: number, seconds: number): PhaseMarker => ({
  id,
  label: id,
  normalized_frame: Math.round(position * 63),
  normalized_position: position,
  timestamp_seconds: seconds
})

const sample = (position: number, seconds: number): AlignmentSample => ({
  normalized_position: position,
  expert_seconds: seconds
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

// The analysis warps the poses between checkpoints and sends the result as
// dense samples. Those already pin the checkpoints, so they stand in for the
// checkpoint pairs rather than being merged with them.
test('follows the warp when the analysis carries one', () => {
  const warp = [sample(0, 1), sample(0.25, 1.8), sample(0.5, 2), sample(0.75, 2.1), sample(1, 3)]

  const anchors = buildAlignmentAnchors(studentTimeline, expertTimeline, 1, 3, warp)

  assert.deepEqual(anchors, [
    { position: 0, seconds: 1 },
    { position: 0.25, seconds: 1.8 },
    { position: 0.5, seconds: 2 },
    { position: 0.75, seconds: 2.1 },
    { position: 1, seconds: 3 }
  ])
  // A quarter of the way in the expert is at 1.8s, not the 1.5s the checkpoint
  // pair alone would have given, and not the 1.5s of a whole-window stretch.
  assert.equal(expertTimeAt(anchors, 0.25), 1.8)
  assert.equal(expertTimeAt(anchors, 0.375), 1.9)
})

test('falls back to the checkpoints when no warp was recorded', () => {
  // Analyses that predate the warp send nothing, which has to behave exactly
  // as it did before it shipped.
  assert.deepEqual(
    buildAlignmentAnchors(studentTimeline, expertTimeline, 1, 3, []),
    buildAlignmentAnchors(studentTimeline, expertTimeline, 1, 3)
  )
})

test('drops warp samples that would stall or rewind the expert', () => {
  const warp = [sample(0, 1), sample(0.3, 1.5), sample(0.6, 1.5), sample(0.8, 1.2), sample(1, 3)]

  const anchors = buildAlignmentAnchors(studentTimeline, expertTimeline, 1, 3, warp)

  assert.deepEqual(anchors, [
    { position: 0, seconds: 1 },
    { position: 0.3, seconds: 1.5 },
    { position: 1, seconds: 3 }
  ])
})

test('holds the warp inside the expert motion window', () => {
  const warp = [sample(0, 0.2), sample(0.5, 2), sample(1, 5)]

  const anchors = buildAlignmentAnchors(studentTimeline, expertTimeline, 1, 3, warp)

  assert.deepEqual(anchors, [
    { position: 0, seconds: 1 },
    { position: 0.5, seconds: 2 },
    { position: 1, seconds: 3 }
  ])
  assert.equal(progressAtExpertTime(anchors, expertTimeAt(anchors, 0.7)), 0.7)
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
