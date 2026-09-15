import assert from 'node:assert/strict'
import test from 'node:test'
import { advanceCheckpointLoop, checkpointRange } from './checkpointPlayback.ts'

test('uses measured source-clock range, including its last frame', () => {
  assert.deepEqual(
    checkpointRange({ id: 'a', timestamp_seconds: 3, start_seconds: 1, end_seconds: 2 }, 0, 5),
    { start: 1, end: 2 + 1 / 30, inferred: false }
  )
  assert.equal(checkpointRange({ id: 'old', timestamp_seconds: 3 }, 0, 5).inferred, true)
})

test('faster video waits, then both restart at their own range starts', () => {
  const video = (time: number) => ({
    currentTime: time,
    ended: false,
    paused: false,
    seeking: false,
    readyState: 4,
    pause() {
      this.paused = true
    },
    async play() {
      this.paused = false
    }
  })
  const student = video(2),
    expert = video(11)
  const tracks = [
    { video: student, range: { start: 1, end: 2, inferred: false } },
    { video: expert, range: { start: 10, end: 13, inferred: false } }
  ]
  advanceCheckpointLoop(tracks)
  assert.equal(student.paused, true)
  assert.equal(expert.paused, false)
  expert.currentTime = 13
  advanceCheckpointLoop(tracks)
  assert.equal(student.currentTime, 1)
  assert.equal(expert.currentTime, 10)
  assert.equal(student.paused, false)
})

test('single video loops without a hidden partner and waits for seeking', () => {
  const v = {
    currentTime: 5,
    ended: true,
    paused: true,
    seeking: true,
    readyState: 4,
    pause() {},
    async play() {
      this.paused = false
    }
  }
  const tracks = [{ video: v, range: { start: 2, end: 5, inferred: false } }]
  advanceCheckpointLoop(tracks)
  assert.equal(v.currentTime, 5)
  v.seeking = false
  advanceCheckpointLoop(tracks)
  assert.equal(v.currentTime, 2)
})
